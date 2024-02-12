import argparse
import json
from typing import Tuple, List

import cv2
import editdistance
from path import Path

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor

import tensorflow as tf 
from tensorflow.keras.layers import Dense
print(tf.__version__)


class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'


def get_img_height() -> int:
    """Fixed height for NN."""
    return 80


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 640, get_img_height()
    return 128, get_img_height()


def write_summary(average_train_loss: List[float], char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'averageTrainLoss': average_train_loss, 'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)


def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())


def train(model: Model,
          loader: DataLoaderIAM,
          line_mode: bool = True,
          early_stopping: int = 25) -> None:
    """Trains NN."""
    epoch = 0  # number of training epochs since start
    summary_char_error_rates = []
    summary_word_accuracies = []

    train_loss_in_epoch = []
    average_train_loss = []

    preprocessor = Preprocessor(get_img_size(line_mode), data_augmentation=False, line_mode=line_mode,dynamic_width=True)
    best_char_error_rate = float('inf')  # best validation character error rate
    no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
    # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            images = batch.imgs
            demo_img = cv2.imread('c04-110-00(2).jpg')
            # visualize_image(demo_img, title = 'Example Image of 320,40 size')
            # visualize_image(images[0],title = 'Image before preprocessing')
            # print(f'Shape before pre-processing is {images[0].shape}')
            # visualize_image(images[0])
            batch = preprocessor.process_batch(batch)
            # visualize_image(batch.imgs[0],title = 'Image after preprocessing')
            # print(f'Shape after pre-processing is {batch.imgs[0].shape}')
            # visualize_preprocessed_opencv_images(images, num_images=1, cols=2)
            # visualize_image(images[0])
            loss = model.train_batch(batch)
            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')
            train_loss_in_epoch.append(loss)

        # validate
        char_error_rate, word_accuracy = validate(model, loader, line_mode)

        # write summary
        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        average_train_loss.append((sum(train_loss_in_epoch)) / len(train_loss_in_epoch))
        write_summary(average_train_loss, summary_char_error_rates, summary_word_accuracies)

        # reset train loss list
        train_loss_in_epoch = []

        # if best validation accuracy so far, save model parameters
        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {best_char_error_rate * 100.0}%')
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        if no_improvement_since >= early_stopping:
            print(f'No more improvement for {early_stopping} epochs. Training stopped.')
            break


def validate(model: Model, loader: DataLoaderIAM, line_mode: bool) -> Tuple[float, float]:
    """Validates NN."""
    print('Validate NN')
    loader.validation_set()
    preprocessor = Preprocessor(get_img_size(line_mode), line_mode=True,data_augmentation=False,dynamic_width=True)
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        batch = loader.get_next()
        images = batch.imgs
        visualize_image(images[0],title = 'Image before preprocessing')
        batch = preprocessor.process_batch(batch)
        images = batch.imgs
        # visualize_preprocessed_opencv_images(images, num_images=2, cols=2)
        visualize_image(images[0],title = 'Image after preprocessing')
        recognized, _ = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
    return char_error_rate, word_accuracy

import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_image(image,title):
    """
    Visualizes a single preprocessed OpenCV-compatible image.
    
    Parameters:
    - image: a preprocessed OpenCV-compatible image (numpy array).
    """
    # Rescale the image from [-0.5, 0.5] back to [0, 1] for visualization
    # image = image.astype(float)
    img_normalized = (image + 0.5).clip(0, 1)
    if img_normalized is None:
        print(f"Failed to load image")
    
    # if img_normalized.ndim == 3:
    #     # If the image has 3 channels, convert from BGR to RGB
    #     img_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
    # elif img_normalized.ndim == 2:
    #     # For grayscale images, ensure correct dimensions for imshow
    #     img_normalized = np.squeeze(img_normalized)
    # if img_normalized.dtype != 'uint8':
        # img_normalized = img_normalized.astype('uint8')

    
    # Display the image
    plt.imshow(image)
    
    # plt.imshow(image)
    plt.axis('off')  # Hide the axis
    plt.title(title)
    plt.show()

def visualize_preprocessed_opencv_images(images, num_images=5, cols=5):
    """
    Visualizes a subset of preprocessed OpenCV-compatible images.
    
    Parameters:
    - images: a list of preprocessed OpenCV-compatible images (numpy arrays).
    - num_images: the number of images to display.
    - cols: the number of columns in the plot.
    """
    # Determine the number of rows needed
    num_images = min(len(images), num_images)
    rows = np.ceil(num_images / cols).astype(int)
    
    # Set up the matplotlib figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    
    for i in range(num_images):
        img = images[i]
        # Rescale images from [-0.5, 0.5] back to [0, 1] for visualization
        img = (img + 0.5).clip(0, 1)
        
        if img.ndim == 3:
            # If the image has 3 channels, convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 2:
            # For grayscale images, ensure correct dimensions for imshow
            img = np.squeeze(img)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    
    # Turn off axes for any unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()




def infer(model: Model, fn_img: Path) -> None:
    """Recognizes text in image provided by file path."""
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=False, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    print(f'Recognized: "{recognized[0]}"')
    print(f'Probability: {probability[0]}')


def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
    parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/word.png')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

    return parser.parse_args()


def main():
    """Main function."""

    # parse arguments and set CTC decoder
    args = parse_args()
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[args.decoder]

    # train the model
    if args.mode == 'train':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)

        # when in line mode, take care to have a whitespace in the char list
        char_list = loader.char_list
        print("value of charecter list in the main is ",char_list)
        if args.line_mode and ' ' not in char_list:
            char_list = [' '] + char_list

        # save characters and words
        with open(FilePaths.fn_char_list, 'w') as f:
            f.write(''.join(char_list))

        with open(FilePaths.fn_corpus, 'w') as f:
            f.write(' '.join(loader.train_words + loader.validation_words))

        model = Model(char_list, decoder_type)
        train(model, loader, line_mode=args.line_mode, early_stopping=args.early_stopping)

    # evaluate it on the validation set
    elif args.mode == 'validate':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
        model = Model(char_list_from_file(), decoder_type, must_restore=True)
        validate(model, loader, args.line_mode)

    # infer text on test image
    elif args.mode == 'infer':
        model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)
        infer(model, args.img_file)


if __name__ == '__main__':
    main()
