import os
import tensorflow as tf
import typing
import numpy as np
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

    IMAGE_WIDTH: int = 105
    IMAGE_HEIGHT: int = 105
    BATCH_SIZE: int = 128

    CLASS_COUNT: int = 0
    CLASS_NAMES: typing.List[str] = []
    EPOCHS: int = 20

    model: tf.keras.Sequential
    train_ds: tf.data.Dataset
    test_ds: tf.data.Dataset

    def load_dataset(self, data_path: str = r"Dataset") -> None:

        self.CLASS_NAMES = os.listdir(os.path.join(data_path, "Test"))
        self.CLASS_COUNT = len(self.CLASS_NAMES)

        print("class count: " + str(self.CLASS_COUNT))
        print(self.CLASS_NAMES)

        f = open("NetworkInfo.txt", "w")
        f.write("Output layer node count: " + str(self.CLASS_COUNT) + "\n\n")
        f.writelines(self.CLASS_NAMES)
        f.close()

        self.train_ds = image_dataset_from_directory(
            directory=os.path.join(data_path, "Train"),
            labels='inferred',
            label_mode='categorical',
            batch_size=self.BATCH_SIZE,
            image_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            color_mode="grayscale"
        )

        self.test_ds = image_dataset_from_directory(
            directory=os.path.join(data_path, "Test"),
            labels='inferred',
            label_mode='categorical',
            batch_size=self.BATCH_SIZE,
            image_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            color_mode="grayscale"
        )

        self.train_ds.batch(self.BATCH_SIZE)
        self.test_ds.batch(self.BATCH_SIZE)

    def build(self) -> None:

        self.model = tf.keras.Sequential()

        # Cu Layers
        self.model.add(tf.keras.layers.Conv2D(64, kernel_size=(
            48, 48), activation='relu', input_shape=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 1)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(tf.keras.layers.Conv2D(
            128, kernel_size=(24, 24), activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(tf.keras.layers.Conv2DTranspose(128, (24, 24), strides=(
            2, 2), activation='relu', padding='same', kernel_initializer='uniform'))
        self.model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))

        self.model.add(tf.keras.layers.Conv2DTranspose(64, (12, 12), strides=(
            2, 2), activation='relu', padding='same', kernel_initializer='uniform'))
        self.model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))

        # Cs Layers
        self.model.add(tf.keras.layers.Conv2D(
            256, kernel_size=(12, 12), activation='relu'))
        self.model.add(tf.keras.layers.Conv2D(
            256, kernel_size=(12, 12), activation='relu'))
        self.model.add(tf.keras.layers.Conv2D(
            256, kernel_size=(12, 12), activation='relu'))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dense(
            self.CLASS_COUNT, activation='softmax'))

        sgd = tf.keras.optimizers.SGD(
            lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mean_squared_error',
                           optimizer=sgd, metrics=['accuracy'])

        return None

    # trains a model on given data

    def train(self) -> None:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="Checkpoints",
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         period=1)

        self.model.fit(self.train_ds, epochs=self.EPOCHS,
                       validation_data=self.test_ds, callbacks=[cp_callback])

        return None

    def save_model(self, dir: str = "MainModel.hdf5") -> None:
        self.model.save(dir)

    def load_model(self, dir: str = "MainModel.hdf5") -> None:
        self.model = tf.keras.models.load_model(dir)


m = model()
m.load_dataset()
m.build()
m.model.summary()
m.train()
m.save_model()
m.load_model()
