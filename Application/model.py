import tensorflow as tf
import typing
from tensorflow import keras

# prepares a proper model layer structure


def prepare_model() -> keras.Sequential:
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10)
        ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model

# trains a model on given data


def train_model(model: keras.Sequential) -> None:

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    model.fit(train_images,
              train_labels,
              epochs=10)

    return None


# # Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

# # Train the model with the new callback
# model.fit(train_images,
#           train_labels,
#           epochs=10,
#           validation_data=(test_images,test_labels),
#           callbacks=[cp_callback])  # Pass callback to training


# # Save the weights
# model.save_weights('./checkpoints/my_checkpoint')
# # Restore the weights
# model.load_weights('./checkpoints/my_checkpoint')


# model.save('saved_model/my_model')
# new_model = tf.keras.models.load_model('saved_model/my_model')
