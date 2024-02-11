import os
import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,models

from dataloader_iam import Batch

# Disable eager mode
# tf.compat.v1.disable_eager_execution()


class DecoderType:
    """CTC decoder types."""
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2


class Model(tf.keras.Model):
    """Minimalistic TF model for HTR."""

    def __init__(self,
                 char_list: List[str],
                 decoder_type: str = DecoderType.BestPath,
                 must_restore: bool = False,
                 dump: bool = False) -> None:
        """Init model: add CNN, RNN and CTC and initialize TF."""
        super(Model,self).__init__()
        self.dump = dump
        self.char_list = char_list
        self.decoder_type = decoder_type
        self.must_restore = must_restore
        self.snap_ID = 0
        self.optimizer = tf.keras.optimizers.Adam(lr = 0.001)
        print("-----------------------", self.char_list)


        # Whether to use normalization over a batch or a population
        # self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')


        # setup CNN, RNN and CTC
        kernel_vals = [5, 5, 3, 3, 3]
        feature_vals = [1, 32, 64, 128, 128, 256]
        stride_vals = pool_vals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        num_layers = len(stride_vals)

        # input_imgs = layers.Input(shape=(None, None, 1), dtype=tf.float32)  # Assuming single-channel images

        # model = models.Sequential()
        # model.add(input_imgs)

        self.conv_layers = []
        # Add Conv2D, MaxPooling2D, and BatchNormalization layers based on the configurations
        for i in range(num_layers):
            self.conv_layers.append((tf.keras.layers.Conv2D(feature_vals[i+1], kernel_size = kernel_vals[i], padding='same', activation='relu')))
            self.conv_layers.append(tf.keras.layers.MaxPooling2D(pool_size=stride_vals[i],strides = stride_vals[i]))
            self.conv_layers.append(tf.keras.layers.BatchNormalization())
                                    
        # for kernel, features, strides in zip(kernel_vals, feature_vals, stride_vals):
        #     self.conv_layers.append(tf.keras.layers.Conv2D(features, (kernel, kernel), padding='same', activation='relu'))
        #     self.conv_layers.append(tf.keras.layers.MaxPooling2D(pool_size=strides))
        #     self.conv_layers.append(tf.keras.layers.BatchNormalization())

        # Flatten the output to feed into the RNN
        # self.flatten = tf.keras.layers.Flatten()

         # RNN layer
        self.rnn_layers = [tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),]

        # # Dense layer to project RNN output to character probabilities
        # self.dense = tf.keras.layers.Dense(len(char_list) + 1)  # +1 for CTC blank character
        
        # Projection layer
        self.projection_layer = tf.keras.layers.Conv2D(filters=len(char_list) + 1, kernel_size=(1, 1), padding='SAME')

        # self.build((None, None, None, 1))  # Specify input shape (dynamic shapes for height and width)
        # self.compile()  # Ensure the model is compiled (might be optional depending on your setup)

        if must_restore:
            self.restore_model()
    
    def restore_model(self):
        model_dir = '../model/'
        latest_snapshot = tf.train.latest_checkpoint(model_dir)
        
        if latest_snapshot:
            print(f'Init with stored values from {latest_snapshot}')
            self.load_weights(latest_snapshot)
        else:
            print('Init with new values or no checkpoint found in the specified directory.')

    def save_model(self):
        model_dir = '../model/'
        os.makedirs(model_dir, exist_ok=True)
        filepath = os.path.join(model_dir, "model_checkpoint")
        self.save_weights(filepath)


    @tf.function
    def call(self, inputs, training=None):
        x = inputs
        # CNN layers
        for layer in self.conv_layers:
            x = layer(x, training=training)
    
        batch_size, width, height, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [batch_size, width, height * channels])  # New shape: [batch_size, timesteps, features]
        # No need to flatten for RNN input
        
        # RNN
        for layer in self.rnn_layers:
            x = layer(x)
        
        # Projection to character probabilities
        x = tf.expand_dims(x, 2)  # Necessary for Conv2D projection layer
        x = self.projection_layer(x)
        x = tf.squeeze(x, axis=2)  # Adjust to expected output shape
        return x  
    
    
    def model(self):
        x = tf.keras.Input(shape=(None, None, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    
    @tf.function
    def compute_label_lengths_from_sparse(self,sparse_labels):
        # Get the maximum label length for each batch by looking at the last index in the sparse tensor for each batch element.
        batch_size = sparse_labels.dense_shape[0]
        # Initialize a tensor to hold the count for each batch element
        label_length = tf.zeros([batch_size], dtype=tf.int32)
        
        # Update counts based on the indices of the sparse tensor
        for i in range(tf.shape(sparse_labels.indices)[0]):
            batch_index = sparse_labels.indices[i][0]
            label_length = tf.tensor_scatter_nd_update(label_length, [[batch_index]], [sparse_labels.indices[i][1] + 1])

        return label_length

    @tf.function  # Compiles `compute_loss` into a static graph for faster execution
    def compute_loss(self, labels, logits, logit_length):
        """
        Computes the CTC loss between the labels and the predictions of the network.

        :param labels: The true labels as a dense Tensor of shape [batch_size, max_label_len].
        :param logits: The logits output from the RNN of shape [batch_size, max_time, num_classes].
        :param logit_length: The length of each logit sequence as a Tensor of shape [batch_size].
        """
        label_length =self.compute_label_lengths_from_sparse(labels)
        loss = tf.nn.ctc_loss(labels=labels,
                              logits=logits,
                              label_length=label_length,
                              logit_length=logit_length,
                              logits_time_major=False,
                              blank_index=-1)
        return tf.reduce_mean(loss)
    
    # @tf.function  # Ensures `decode` runs in graph mode, optimizing the decoding process
    def decode(self, logits, logit_length):
        """
        Decodes the logits using either greedy or beam search decoder.

        :param logits: The logits output from the RNN of shape [batch_size, max_time, num_classes].
        :param logit_length: The length of each logit sequence as a Tensor of shape [batch_size].
        """
        # Transpose needed if logits are not time major
        # self.decoded = None
        decoded = None  # Or appropriate default value based on expected type
        logits = tf.transpose(logits, perm=[1, 0, 2])
        print(f"-----------------------------------------{self.decoder_type}")
        if self.decoder_type == 0:
            decoded, _ = tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=logit_length)
        elif self.decoder_type == 1:
            decoded, _ = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=logit_length, beam_width=50)
        # Implement other decoders as needed
                # Ensure there's an `else` block if there are other decoder types or a catch-all is needed
        else:
            print("Warning: No matching decoder type found, or implement default handling here")
        return decoded
    
    def decoder_output_to_text(self, ctc_output, batch_size) -> List[str]:
        """Extract texts from output of CTC decoder."""

        # word beam search: already contains label strings
        if self.decoder_type == DecoderType.WordBeamSearch:
            label_strs = ctc_output

        # TF decoders: label strings are contained in sparse tensor
        else:
            # ctc returns tuple, first element is SparseTensor
            decoded = ctc_output[0]
            # decoded = tf.sparse.to_dense(ctc_output[0], default_value=-1).numpy()


            # contains string of labels for each batch element
            label_strs = [[] for _ in range(batch_size)]

            # go over all indices and save mapping: batch -> values
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batch_element = idx2d[0]  # index according to [b,t]
                label_strs[batch_element].append(label)

        # map labels to chars for all batch elements
        print("CHARECTER LIST IS ",self.char_list)
        self.char_list = [' ', '!', '"', '#', '&', "'", '(', ')', '*', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        return [''.join([self.char_list[c] if c < len(self.char_list) else '<UNK>' for c in labelStr]) for labelStr in label_strs]
    
    @tf.function
    def train_batch(self, batch):
        images = batch.imgs
        tensor_list = batch.imgs
        tensor_list = [tf.cast(tensor, dtype=tf.float32) for tensor in tensor_list] 
        tensor_list_with_channel = [tf.expand_dims(tensor, -1) for tensor in tensor_list]
        images_tensor = tf.stack(tensor_list_with_channel, axis=0)  # This should print: (batch_size, 128, 32, 1)
        labels = batch.gt_texts
        
        # max_text_len = images[0].shape[0] //4
        sparse_labels = self.to_sparse(labels)
        # Assuming labels are already in the sparse tensor format required by CTC loss
        # and images are the inputs to your model
        with tf.GradientTape() as tape:
            logits = self(images_tensor, training=True)
 # Obtain logits from the model (batch_size, W/4, C + 1)
            # Compute logit length based on your model's specifics, e.g., dividing image width by downsample factor
            logit_length = tf.fill([tf.shape(images)[0]], tf.shape(logits)[1])
            loss = self.compute_loss(sparse_labels, logits, logit_length)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss


    def to_sparse(self, texts: List[str]) -> Tuple[List[List[int]], List[int], List[int]]:
        """Put ground truth texts into sparse tensor for ctc_loss."""
        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for batchElement, text in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            label_str = [self.char_list.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(label_str) > shape[1]:
                shape[1] = len(label_str)
            # put each label into sparse tensor
            for i, label in enumerate(label_str):
                indices.append([batchElement, i])
                values.append(label)
        sparse_tensor_to_return = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
        return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    

    @staticmethod
    def dump_nn_output(rnn_output: np.ndarray) -> None:
        """Dump the output of the NN to CSV file(s)."""
        dump_dir = '../dump/'
        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)

        # iterate over all batch elements and create a CSV file for each one
        max_t, max_b, max_c = rnn_output.shape
        for b in range(max_b):
            csv = ''
            for t in range(max_t):
                csv += ';'.join([str(rnn_output[t, b, c]) for c in range(max_c)]) + ';\n'
            fn = dump_dir + 'rnnOutput_' + str(b) + '.csv'
            print('Write dump of NN to file: ' + fn)
            with open(fn, 'w') as f:
                f.write(csv)


    def infer_batch(self, batch: Batch, calc_probability: bool = False, probability_of_gt: bool = False):
        eval_list = []
        images = batch.imgs
        print(len(images))
        tensor_list = batch.imgs
        tensor_list = [tf.cast(tensor, dtype=tf.float32) for tensor in tensor_list] 
        tensor_list_with_channel = [tf.expand_dims(tensor, -1) for tensor in tensor_list]
        images_tensor = tf.stack(tensor_list_with_channel, axis=0)  # This should print: (batch_size, 128, 32, 1)
        labels = batch.gt_texts
        # max_text_len = images[0].shape[0] //4
        sparse_labels = self.to_sparse(labels)
        logits = self(images_tensor, training=False)
        print(f"logits shape is {logits.shape}") 
        print(f"logits 0 is {logits[0]}")
        
        logit_length = tf.fill([tf.shape(images)[0]], tf.shape(logits)[1])
        decoded = self.decode(logits, logit_length)
        print('decoded output is ', decoded) 
        # print(decoded[0])
        batch_size = logits.shape[0]  # Assuming logits is a tensor of shape [batch_size, max_time, num_classes]
        texts = model.decoder_output_to_text(ctc_output=decoded, batch_size=batch_size)
         # Convert decoded indices to text
        # if decoded is not None:  # Ensure decoded is not None
        #     # Ensure operation is outside @tf.function
        #     texts = ["".join([self.char_list[i.numpy()] for i in text.values]) for text in decoded]
        # else:
        #     texts = []
        probabilities = None
        if calc_probability:
            sparse_labels = self.to_sparse(labels if probability_of_gt else self.decode(logits,logit_length))  # Placeholder for actual conversion
            logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
            
            # Calculate loss (used as negative log probability)
            loss = self.compute_loss(sparse_labels, logits, logit_length)
            probabilities = tf.exp(-loss)  # Convert loss to probabilities
        
        return texts,probabilities

        # # sequence length depends on input image size (model downsizes width by 4)
        # max_text_len = batch.imgs[0].shape[0] // 4

        # # dict containing all tensor fed into the model
        # feed_dict = {self.input_imgs: batch.imgs, self.seq_len: [max_text_len] * num_batch_elements,
        #              self.is_train: False}


    # def infer_batch(self, batch: Batch, calc_probability: bool = False, probability_of_gt: bool = False):
    #     """Feed a batch into the NN to recognize the texts."""

    #     # decode, optionally save RNN output
    #     num_batch_elements = len(batch.imgs)

    #     # put tensors to be evaluated into list
    #     eval_list = []

    #     if self.decoder_type == DecoderType.WordBeamSearch:
    #         eval_list.append(self.wbs_input)
    #     else:
    #         eval_list.append(self.decoder)

    #     if self.dump or calc_probability:
    #         eval_list.append(self.ctc_in_3d_tbc)

    #     # sequence length depends on input image size (model downsizes width by 4)
    #     max_text_len = batch.imgs[0].shape[0] // 4

    #     # dict containing all tensor fed into the model
    #     feed_dict = {self.input_imgs: batch.imgs, self.seq_len: [max_text_len] * num_batch_elements,
    #                  self.is_train: False}

    #     # evaluate model
    #     eval_res = self.sess.run(eval_list, feed_dict)

    #     # TF decoders: decoding already done in TF graph
    #     if self.decoder_type != DecoderType.WordBeamSearch:
    #         decoded = eval_res[0]
    #     # word beam search decoder: decoding is done in C++ function compute()
    #     else:
    #         decoded = self.decoder.compute(eval_res[0])

    #     # map labels (numbers) to character string
    #     texts = self.decoder_output_to_text(decoded, num_batch_elements)

    #     # feed RNN output and recognized text into CTC loss to compute labeling probability
    #     probs = None
    #     if calc_probability:
    #         sparse = self.to_sparse(batch.gt_texts) if probability_of_gt else self.to_sparse(texts)
    #         ctc_input = eval_res[1]
    #         eval_list = self.loss_per_element
    #         feed_dict = {self.saved_ctc_input: ctc_input, self.gt_texts: sparse,
    #                      self.seq_len: [max_text_len] * num_batch_elements, self.is_train: False}
    #         loss_vals = self.sess.run(eval_list, feed_dict)
    #         probs = np.exp(-loss_vals)

    #     # dump the output of the NN to CSV file(s)
    #     if self.dump:
    #         self.dump_nn_output(eval_res[1])

    #     return texts, probs

    def save(self) -> None:
        """Save model to file."""
        self.snap_ID += 1
        self.saver.save(self.sess, '../model/snapshot', global_step=self.snap_ID)


# Example usage
char_list = ['a', 'b', 'c', 'd']  # Example character list for OCR
model = Model(char_list=char_list)

# Print model summary to verify architecture
model.build(input_shape=(None, 128, 32, 1))  # Example input shape, adjust as needed
model.summary()
