
import pathlib
import numpy
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# data_dir = tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', fname='flower_photos', untar=True)
data_dir = pathlib.Path('D:\DataSets\Raw Image\scrape-wtf-new')
image_count = len(list(data_dir.glob('*')))
print("image count: " + str(image_count))

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = numpy.ceil(image_count/BATCH_SIZE)
CLASS_NAMES = numpy.array([item.name for item in data_dir.glob('*')])

print(CLASS_NAMES)


image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(
                                                         IMG_HEIGHT, IMG_WIDTH),
                                                     classes=list(CLASS_NAMES))
