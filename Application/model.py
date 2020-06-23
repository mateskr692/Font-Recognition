import os
import tensorflow as tf
import typing
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image


# model.load_dataset()
# model.build()
# model.train()
# model.save()
# model.load()
# model.predict()
class networkModel:

    IMAGE_WIDTH: int = 105
    IMAGE_HEIGHT: int = 105
    BATCH_SIZE: int = 100

    CLASS_COUNT: int = 0
    CLASS_NAMES: typing.List[str] = []
    EPOCHS: int = 80

    train_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    model: tf.keras.Sequential

    def load_dataset(self, data_path: str = r"Dataset") -> None:

        # Class names and count
        self.CLASS_NAMES = os.listdir(os.path.join(data_path, "Test"))
        self.CLASS_COUNT = len(self.CLASS_NAMES)

        print("class count: " + str(self.CLASS_COUNT))
        print(self.CLASS_NAMES)

        f = open("NetworkInfo.txt", "w")
        f.write("Output layer node count: " + str(self.CLASS_COUNT) + "\n\n")
        f.writelines(self.CLASS_NAMES)
        f.close()


        # Load data and squeeze images to the bounding box
        self.train_ds = image_dataset_from_directory(
            directory=os.path.join(data_path, "Train"),
            labels='inferred',
            label_mode='categorical',
            batch_size=self.BATCH_SIZE,
            image_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            color_mode="grayscale",

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

        # Input layer
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.GaussianNoise(
            0.01, input_shape=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1)))

        # Cu layers
        self.model.add(tf.keras.layers.Conv2D(
            64, kernel_size=(58, 58), activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(tf.keras.layers.Conv2D(
            128, kernel_size=(24, 24), activation='relu', padding="same"))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # Cs layers
        self.model.add(tf.keras.layers.Conv2D(
            256, kernel_size=(12, 12), activation='relu', padding="same"))
        self.model.add(tf.keras.layers.Conv2D(
            256, kernel_size=(12, 12), activation='relu', padding="same"))
        self.model.add(tf.keras.layers.Conv2D(
            256, kernel_size=(12, 12), activation='relu', padding="same"))
        self.model.summary()

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(
            512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(
            256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(
            128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))

        # Output layer
        self.model.add(tf.keras.layers.Dense(
            self.CLASS_COUNT, activation='softmax'))
        self.model.summary()

        # Compile model
        opt = tf.keras.optimizers.SGD(
            learning_rate=0.0075, momentum=0.8, nesterov=True) #decay=0.0005,
        self.model.compile(optimizer=opt,
                           loss=tf.keras.losses.BinaryCrossentropy(
                               from_logits=True),
                           metrics=['accuracy', tf.keras.losses.BinaryCrossentropy(
                               from_logits=True, name='binary_crossentropy'), ]
                           )

        return None

    # trains a model on given data

    def train(self) -> None:
        cp_callback = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath="Checkpoints", save_weights_only=True, verbose=1, period=1),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_binary_crossentropy', patience=10),
            tf.keras.callbacks.TensorBoard("Logs"),
        ]

        self.model.fit(self.train_ds, epochs=self.EPOCHS,
                       validation_data=self.test_ds, callbacks=[cp_callback])

        return None

    def save_model(self, dir: str = "MainModel.hdf5") -> None:
        self.model.save(dir)

    def load_model(self, dir: str = "MainModel.hdf5") -> None:
        self.model = tf.keras.models.load_model(dir, compile=False)
        opt = tf.keras.optimizers.SGD(
            learning_rate=0.0075, momentum=0.8, nesterov=True) #decay=0.0005,
        self.model.compile(optimizer=opt,
                           loss=tf.keras.losses.BinaryCrossentropy(
                               from_logits=True),
                           metrics=['accuracy', tf.keras.losses.BinaryCrossentropy(
                               from_logits=True, name='binary_crossentropy'), ]
                           )

    def test(self) -> None:
        self.model.evaluate(self.test_ds)

    def predict(self, dir: str):
        img = Image.open(dir)
        img = img.convert('L')
        dim = min(img.width, img.height)
        img = img.crop((0, 0, dim, dim))
        img = img.resize((self.IMAGE_WIDTH, self.IMAGE_HEIGHT))

        #img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        prediction = (self.model.predict(img, batch_size=1))[0].tolist()
        print(prediction)

        m = max(prediction)
        idx = prediction.index(m)
        result = self.CLASS_NAMES[idx] + "\n" + str( (m/sum(prediction)) * 100.0) + "%"

        return result

# m = networkModel()
# m.load_model()
# m.load_dataset()
# m.model.summary()
# m.test()
# m.build()
# m.model.summary()
# m.train()
# m.save_model()

