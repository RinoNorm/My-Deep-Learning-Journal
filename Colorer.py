import tensorflow_datasets as tfds
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt

# Train a neural network to colorize grayscale images using the Imagenette training and validation sets.
train_ds, val_ds = tfds.load("imagenette/320px-v2", split=["train", "validation"], as_supervised=False)

# Define a preprocessing function.
def preprocessing(example):
    color = tf.image.convert_image_dtype(example["image"], tf.float32)
    color = tf.image.resize(color, (128, 128), antialias=True)
    grayscale = tf.image.rgb_to_grayscale(color) # Convert the color image to grayscale.
    return grayscale, color

# Use 7000 images from each split for the initial experiments.
prepared_train_dataset = (train_ds.take(7000).shuffle(7000).map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE).batch(16).prefetch(tf.data.AUTOTUNE))
# .prefetch(tf.data.AUTOTUNE) prepares upcoming batches while the model processes the current batch to reduce data loading delays.
prepared_val_dataset = (val_ds.take(7000).map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE).batch(16).prefetch(tf.data.AUTOTUNE))

# Build a convolutional autoencoder:
# the encoder extracts features, and the decoder reconstructs a color image.
model = keras.Sequential([
    keras.Input(shape=(128, 128, 1)),

    # Encoder: extract features and reduce spatial dimensions.
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPool2D(2),  # 64 × 64

    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPool2D(2),  # 32 × 32

    layers.Conv2D(128, 3, activation="relu", padding="same"),

    # Decoder: restore the original spatial dimensions.
    layers.UpSampling2D(2, interpolation="bilinear"),  # 64 × 64
    layers.Conv2D(64, 3, activation="relu", padding="same"),

    layers.UpSampling2D(2, interpolation="bilinear"),  # 128 × 128
    layers.Conv2D(32, 3, activation="relu", padding="same"),

    # Output three RGB channels with values between 0 and 1.
    layers.Conv2D(3, 1, activation="sigmoid", padding="same"),
])

model.compile(optimizer="adam", loss="mse")
colorer = model.fit(prepared_train_dataset, validation_data=prepared_val_dataset, epochs=100)
model.save("colorer.keras") # To load the model, use: model = keras.models.load_model("colorer.keras")

gray_batch, color_batch = next(iter(prepared_val_dataset))
gray_batch, color_batch = next(iter(prepared_val_dataset))

gray_batch_last5 = gray_batch[-5:]
color_batch_last5 = color_batch[-5:]

predictions = model.predict(gray_batch_last5, verbose=0)

fig, axes = plt.subplots(5, 3, figsize=(9, 12))

for i in range(5):
    axes[i, 0].imshow(gray_batch_last5[i, ..., 0], cmap="gray", vmin=0, vmax=1)
    axes[i, 1].imshow(predictions[i])
    axes[i, 2].imshow(color_batch_last5[i])

    for ax in axes[i]:
        ax.axis("off")

for ax, title in zip(axes[0], ["Grayscale", "Prediction", "Original"]):
    ax.set_title(title)

plt.savefig("colorized_image.png", dpi=150, bbox_inches="tight")