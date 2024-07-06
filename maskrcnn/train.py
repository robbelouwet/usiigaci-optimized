"""
Mask-RCNN training script.

Usage:
    train.py --train-dir=<dataset_dir> --val-dir=<val_dir>
             --out-dir=<out_dir> [--weights=<weights_path>]
"""

import os
from datetime import datetime
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# Root directory of the project
from maskrcnn.mrcnn.utils import findBatchSize

# Import Mask RCNN
import cv2
import numpy as np
from imgaug import augmenters as iaa
from maskrcnn.mrcnn import model as modellib
from maskrcnn.mrcnn import utils
from maskrcnn.mrcnn.config import Config
from project_utils import ROOT_DIR
import tensorflow as tf
from notify_run import Notify
from maskrcnn.evaluate_predictions import evaluate, print_evaluation_metrics
import matplotlib.pyplot as plt

INPUT_CHANNEL = 'raw.tif'

# Create a TensorBoard callback
logdir = ROOT_DIR + "/logs"
# Create a TensorBoard callback
logs = ROOT_DIR + "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch='500,520')


# https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py
class CrystalConfig(Config):
    NAME = "Crysco GPU Stack config"

    # Adjust to GPU memory A 12GB GPU can typically handle 4 images of 1024x1024px.
    IMAGES_PER_GPU = 2

    # used in maskrcnn/mrcnn/model.py::2050 MaskRCNN build(), must be >1
    GPU_COUNT = 4

    # the batch size of the model => IMAGES_PER_GPU * GPU_COUNT
    # Optimal value found with findBatchSize(keras_model)
    # Pass explicitly to override the default auto generated
    # BATCH_SIZE = 16

    # n° head epochs
    EPOCHS_HEAD = 313

    # n° all epochs
    EPOCHS_ALL = 513

    NUM_CLASSES = 2  # Background + cell

    # STEPS will be automatically calculated, pass explicitly to override
    # STEPS_PER_EPOCH = 120
    # VALIDATION_STEPS = 45

    LEARNING_RATE = 0.007

    SCHEDULER_LR_DECREASER = 0.995

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.5
    # DETECTION_MIN_CONFIDENCE = 0.5

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    # BACKBONE = "resnet50"
    BACKBONE = "resnet101"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more proposals.
    # RPN_NMS_THRESHOLD = 0.9
    RPN_NMS_THRESHOLD = 0.99

    # How many anchors per image to use for RPN training
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128

    # Image mean (RGB)
    # MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    MEAN_PIXEL = np.array([126, 126, 126])
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    # MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    MINI_MASK_SHAPE = (100, 100)

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    # TRAIN_ROIS_PER_IMAGE = 128
    TRAIN_ROIS_PER_IMAGE = 256

    # Maximum number of ground truth instances to use in one image
    # MAX_GT_INSTANCES = 200
    MAX_GT_INSTANCES = 500
    # Max number of final detections per image
    # DETECTION_MAX_INSTANCES = 400
    DETECTION_MAX_INSTANCES = 1000


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("log.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


############################################################
#  Dataset
############################################################


class cellDataset(utils.Dataset):

    def load_cell(self, dataset_dir):
        """
        Load dataset
        :param dataset_dir: Root directory to dataset
        :return:
        """
        self.add_class("cell", 1, "cell")

        image_ids = os.listdir(dataset_dir)

        for image_id in image_ids:
            self.add_image('cell', image_id=image_id, path=os.path.join(dataset_dir, image_id, INPUT_CHANNEL))

        print('[DATASET]', dataset_dir, len(self.image_ids))

    def load_mask(self, image_id):
        """

        :param image_id:
        :return:
        """
        info = self.image_info[image_id]

        mask_dir = os.path.dirname(info['path'])
        mask_path = os.path.join(mask_dir, 'instances_ids.png')

        ids_mask = cv2.imread(mask_path, 0)
        instances_num = len(np.unique(ids_mask)) - 1

        mask = np.zeros((ids_mask.shape[0], ids_mask.shape[1], instances_num))
        for i in range(instances_num):
            # print(np.where(ids_mask == (i + 1)))
            slice = mask[..., i]
            slice[np.where(ids_mask == (i + 1))] = 1
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """

        :param image_id:
        :return:
        """
        info = self.image_info[image_id]
        if info["source"] == "cell":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


def calculate_steps_per_epoch(batch_size, dir):
    _, dirs, _ = next(os.walk(dir))
    steps_per_epoch = np.ceil((len(dirs) / batch_size))

    return steps_per_epoch


def check_steps(config, train_dir, val_dir):
    if config.STEPS_PER_EPOCH == 0:
        config.STEPS_PER_EPOCH = calculate_steps_per_epoch(config.BATCH_SIZE, train_dir)

    if config.VALIDATION_STEPS == 0:
        config.VALIDATION_STEPS = calculate_steps_per_epoch(config.BATCH_SIZE, val_dir)

    if config.STEPS_PER_EPOCH == 0 or config.VALIDATION_STEPS == 0:
        print(
            'Calculated steps per epoch resulted in 0, this may indicate that there has no training dataset been found.')
        exit(1)


def train(model, config, train_dir, val_dir):
    train_dataset = cellDataset()
    train_dataset.load_cell(train_dir)
    train_dataset.prepare()

    val_dataset = cellDataset()
    val_dataset.load_cell(val_dir)
    val_dataset.prepare()

    augmentation = iaa.SomeOf((0, 2), [iaa.Fliplr(0.5), iaa.Flipud(0.5),
                                       iaa.OneOf(
                                           [iaa.Affine(rotate=90), iaa.Affine(rotate=180), iaa.Affine(rotate=270)]),
                                       iaa.Multiply((0.8, 1.5)),
                                       iaa.GaussianBlur(sigma=(0.0, 5.0)),
                                       iaa.AdditiveGaussianNoise(10, 20),
                                       iaa.GammaContrast(gamma=0.5)
                                       ])

    print("Train network heads")
    print("Training with batch size {}".format(config.BATCH_SIZE))
    history_heads = model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE,
                                epochs=config.EPOCHS_HEAD,  # was 100
                                augmentation=augmentation, layers='heads', custom_callbacks=[tboard_callback])

    # Train all layers
    print("Train all layers")
    print("Training with batch size {}".format(config.BATCH_SIZE))
    history_all = model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE,
                              epochs=config.EPOCHS_ALL, # was 200
                              augmentation=augmentation, layers='all', custom_callbacks=[tboard_callback])

    """plot_heads = create_plot_graph(history_heads, config.EPOCHS_HEAD)
    plot_all = create_plot_graph(history_all, config.EPOCHS_ALL)

    plot_accuracy_heads = create_training_val_accuracy_graph(history_heads, config.EPOCHS_HEAD)
    plot_accuracy_all = create_training_val_accuracy_graph(history_heads, config.EPOCHS_HEAD)

    plot_heads.show()
    plot_all.show()
    plot_accuracy_heads.show()
    plot_accuracy_all.show()"""


def create_plot_graph(history, epochs):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    plot = plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plot = plot.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plot = plot.title('Training and validation loss of heads')
    plot = plot.xlabel('Epochs')
    plot = plot.ylabel('Loss')
    plot = plot.legend()

    return plot


def create_training_val_accuracy_graph(history, epochs):
    history_dict = history.history
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    plot = plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plot = plot.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plot = plot.title('Training and validation accuracy')
    plot = plot.xlabel('Epochs')
    plot = plot.ylabel('Accuracy')
    plot = plot.legend()

    return plot


def check_train_val_dir(crystal_dir):
    train_dir = os.path.join(crystal_dir, 'train')
    val_dir = os.path.join(crystal_dir, 'val')

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print('> Training on data: ', train_dir)
    else:
        print("Can't find val and train folder in path. Preparing dataset...")
        from maskrcnn.prepareDataSet import prepare_data_set
        prepare_data_set(crystal_dir)

        train_dir = os.path.join(crystal_dir, 'train')
        val_dir = os.path.join(crystal_dir, 'val')

        if not os.path.exists(train_dir) and not os.path.exists(val_dir):
            print('Could not find or create train & val folder.')
            exit(1)


def main():
    # args = docopt(__doc__)
    #
    # train_dir = args['--train-dir']
    # val_dir = args['--val-dir']
    # out_dir = args['--out-dir']
    # weights_path = args['--weights']
    # sys.stdout = Logger()

    crystal_dir = weights_path = ""

    if len(sys.argv) < 2:
        print('Expected at least 1 argument but received {}'.format(len(sys.argv) - 1))
        exit(1)
    else:
        crystal_dir = sys.argv[1]

    if len(sys.argv) == 2:
        weights_path = os.path.join(ROOT_DIR, 'trainingModels/model.h5')
    else:
        extension = os.path.splitext(sys.argv[2])[1]
        if extension == ".h5":
            weights_path = sys.argv[2]

    train_by_path(crystal_dir, weights_path, use_new_model=False)


def train_by_path(crystal_dir, weights_path, use_new_model=False):
    if not os.path.isdir(os.path.relpath(os.path.abspath(crystal_dir), './')):
        print("Crystal dir not found.")
        exit(1)

    # Check if the train and val directories are present. If not, the dataset will be prepared. If after that they
    # still can't be found, the process will be stopped.
    check_train_val_dir(crystal_dir)

    train_dir = os.path.join(crystal_dir, 'train')
    val_dir = os.path.join(crystal_dir, 'val')

    out_dir = os.path.join("./trainingModels/", os.path.basename(crystal_dir))

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print('> Saving results to: ', out_dir)

    config = CrystalConfig()

    # check the steps per epoch and calculate if needed
    check_steps(config, train_dir, val_dir)

    # initialise the maskrcnn model
    model_name = os.path.basename(weights_path)
    model_name = os.path.splitext(model_name)[0]

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=out_dir, model_name=model_name)

    # To find the optimal batch size
    # findBatchSize(model.keras_model)

    # load weights into the model
    if weights_path:
        weights_path = os.path.relpath(os.path.abspath(weights_path), './')
        print('> Loading weights from: ', weights_path)

        if not use_new_model:
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"
            ])

        try:
            # train the model
            import timeit
            start_time = timeit.default_timer()

            train(model, config, train_dir, val_dir)

            elapsed_time = timeit.default_timer()
            elapsed_time = elapsed_time - start_time
            hours = elapsed_time / (60*60)
            print(f"Elapsed time: {elapsed_time}. Hours: {hours}")
        except Exception as e:
            print(f'An error has occurred: {e}')
        finally:
            # If the training crashes we would still like for the progress we made to be saved to the model
            model.save_weights(weights_path)
    else:
        print('No weights found')
        exit(1)


if __name__ == '__main__':
    try:
        # 1/0  # run the except clause to test notifications
        folder = ''

        # if a crystal folder was provided as cmd arg:
        if len(sys.argv) >= 1:
            if len(sys.argv) > 1:
                print("crystal folder was provided")
                folder = sys.argv[1]
            else:
                folder = ROOT_DIR + "/trainingSet/combined/"
                print("crystal folder was not provided, using default {}".format(folder))

        if len(sys.argv) > 2:
            sys.argv = [sys.argv[0], folder, sys.argv[2]]
        else:
            sys.argv = [sys.argv[0], folder]

        main()
    except Exception as e:
        try:
            notify = Notify()
            notify.send("training failed")
        except Exception as e:
            # if it can't send a notification because no one is registered, just ignore it
            pass
        raise
