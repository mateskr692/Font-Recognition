import tensorflow as tf
import typing
from tensorflow import keras
from keras import layers
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory


# model.load_dataset()
# model.build()
# model.train()
# model.save()
# model.load()
# model.predict()
class model:

    IMAGE_WIDTH: int = 120
    IMAGE_HEIGHT: int = 120
    BATCH_SIZE: int = 50

    CLASS_COUNT: int = 0
    CLASS_NAMES: typing.List[str] = []
    EPOCHS: int = 10

    model: keras.Sequential
    train_ds: tf.data.Dataset
    test_ds: tf.data.Dataset

    def load_dataset(self, data_path: str = r"Dataset") -> None:

        self.CLASS_NAMES = os.listdir(os.path.join(data_path, "Test"))
        self.CLASS_COUNT = len(self.CLASS_NAMES)

        print("class count: " + str(self.CLASS_COUNT))
        print(self.CLASS_NAMES)

        self.train_ds = image_dataset_from_directory(
            directory=os.path.join(data_path, "Train"),
            labels='inferred',
            label_mode='categorical',
            batch_size=32,
            image_size=(120, 120)
        )

        self.test_ds = image_dataset_from_directory(
            directory=os.path.join(data_path, "Test"),
            labels='inferred',
            label_mode='categorical',
            batch_size=32,
            image_size=(120, 120)
        )

    def build(self) -> None:

        self.model = keras.Sequential()
        # Cu Layers
        self.model.add(layers.Conv2D(64, kernel_size=(
            48, 48), activation='relu', input_shape=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Conv2D(
            128, kernel_size=(24, 24), activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(layers.Conv2DTranspose(128, (24, 24), strides=(
            2, 2), activation='relu', padding='same', kernel_initializer='uniform'))
        self.model.add(layers.UpSampling2D(size=(2, 2)))
        self.model.add(layers.Conv2DTranspose(64, (12, 12), strides=(
            2, 2), activation='relu', padding='same', kernel_initializer='uniform'))
        self.model.add(layers.UpSampling2D(size=(2, 2)))

        # Cs Layers
        self.model.add(layers.Conv2D(
            256, kernel_size=(12, 12), activation='relu'))
        self.model.add(layers.Conv2D(
            256, kernel_size=(12, 12), activation='relu'))
        self.model.add(layers.Conv2D(
            256, kernel_size=(12, 12), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(4096, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(4096, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(2383, activation='relu'))
        self.model.add(layers.Dense(self.CLASS_COUNT, activation='softmax'))

        sgd = keras.optimizers.SGD(
            lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mean_squared_error',
                           optimizer=sgd, metrics=['accuracy'])

        return None

    # trains a model on given data

    def train(self) -> None:

        self.model.fit(self.train_ds, self.EPOCHS,
                       validation_data=self.test_ds)

        return None


m = model()
m.load_dataset()
m.build()
# m.train()
