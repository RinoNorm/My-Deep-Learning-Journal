import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras.layers import Conv2D

# The task was to implement image stylization. Two neural networks are used for this:
# a ready-made pretrained VGG19 and a stylization network. First, the image datasets are loaded, then normalized.
raw_dataset = tfds.load("imagenette/320px", split="train", as_supervised=True) # The first parameter loads the dataset, the second loads its training split, and the third specifies the format,
# namely, image + class.
prepared_images = []
for image, label in raw_dataset.take(1000): # The dataset contains both images and their classes, so we take both the image itself and its class.
    # The class does not take part in preparing the image. Since it is just a class label, we do not use it, but unpack it in the loop to avoid an error.
    # .take(1000) takes only 1000 images.
    image = tf.image.convert_image_dtype(image, tf.float32) # In Imagenette, pixel color values are represented by integers from 0 to 255.
    # This line converts them to float32 floating-point numbers and scales them to the range from 0 to 1.
    image = tf.image.resize(image, (128, 128), antialias=True) # antialias enables smoothing when reducing the image size to reduce jagged edges and other distortions.
    prepared_images.append(image)
images = tf.stack(prepared_images) # Combine the list of images into a single tensor.
prepared_dataset = tf.data.Dataset.from_tensor_slices((images, images)) # Create a dataset of pairs (input image, original image for content loss).
content, _ = next(iter(prepared_dataset)) # iter lets us retrieve dataset elements one by one, and next takes the first element.
content = content[tf.newaxis, ...] # Add a batch dimension: (128, 128, 3) → (1, 128, 128, 3).

# Now load the style image and preprocess it.
style = keras.utils.load_img("style.png", color_mode="rgb", target_size=(128, 128))
style = keras.utils.img_to_array(style)

# The initial data is ready. Now we need to load VGG19 to extract the features used to calculate content and style losses.
# To do this, we need to write a script to evaluate stylization and content quality, then calculate the overall loss function.
# Let's start with the style loss function.
# Prepare the model for extracting feature maps.
vgg = keras.applications.VGG19(include_top=False, weights="imagenet")
# Pretrained VGG19 expects pixel values in a specific format. The function performs two transformations:
# 1. Changes the color channel order from RGB to BGR.
# 2. Subtracts the channel mean values used when preparing data for VGG19.
vgg.trainable = False # This line prevents VGG19's weights from changing during stylization, since the network has already been trained on other images.

# Now prepare the style image itself.
style = np.expand_dims(style, axis=0)
style_input = keras.applications.vgg19.preprocess_input(style)

# Now we need to pause here. Next, we need to prepare the data for selecting the VGG19 layers whose features we will use: five for evaluating style and one for evaluating content.
style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
content_layer = "block5_conv2"

outputs = []

for name in style_layers + [content_layer]:
    layer = vgg.get_layer(name)
    outputs.append(layer.output)

# To calculate the loss function, we need the mean of the squared differences between the features of the original and stylized images, divided by 4.
# Formula: content_loss = sum of squared differences / (4 * batch * height * width * channels).
def content_loss(content_features, generated_features):
    return tf.reduce_mean(tf.square(generated_features - content_features)) / 4.0

# Set the weighting hyperparameters for content and style. The weights control how much the content and style losses contribute to the total loss.
content_weight = 4.0
style_weight = 0.000000009

extractor = keras.Model(inputs=vgg.input, outputs=outputs) # This line creates a model for obtaining tensors from the selected VGG19 blocks.
style_features = extractor(style_input) # Here, an array of 6 feature tensors is created for the style image.

# Next, we need to write a function for creating Gram matrices. They are needed to evaluate the stylization criteria. A Gram matrix is a tensor represented as a square matrix.
# The tensor is represented as a matrix where the rows are spatial positions and the columns are feature channels. This matrix is multiplied by its transpose.
def gram_matrix(features):
    shape = tf.shape(features)
    batch, height, width, channels = shape[0], shape[1], shape[2], shape[3] # Get the number of images, height, width, and number of channels.

    matrix = tf.reshape(features, (batch, height * width, channels))
    gram = tf.matmul(matrix, matrix, transpose_a=True) # This is how matrices are multiplied in TensorFlow.

    return gram / tf.cast(height * width, features.dtype) # Next, normalize the Gram matrix by dividing its elements by the number of spatial positions in the feature map.

# Style Gram matrices.
style_grams = []

for features in style_features[:5]: # The first 5 tensors correspond to the layers used to evaluate style.
    style_grams.append(gram_matrix(features))

# We have all the pieces needed for the loss function, so now we can write it.
def total_loss(y_true, y_pred):
    # y_true is the original image, and y_pred is the stylized image.
    content_input = keras.applications.vgg19.preprocess_input(y_true * 255.0)
    generated_input = keras.applications.vgg19.preprocess_input(y_pred * 255.0)

    content_features = extractor(content_input)
    generated_features = extractor(generated_input)

    loss_content = content_loss(content_features[-1], generated_features[-1])

    loss_style = 0.0

    # Now create a Gram matrix for the stylized image to calculate the style loss.
    for i in range(len(style_layers)):
        generated_gram = gram_matrix(generated_features[i])
        difference = generated_gram - style_grams[i] # And this is where we calculate the difference.

        # Sum the squares for each image, then normalize the squared differences over the batch and both dimensions of the Gram matrices.
        layer_loss = tf.reduce_mean(tf.square(difference))
        loss_style += layer_loss

    loss_style /= len(style_layers)
    return content_weight * loss_content + style_weight * loss_style # And here is the final formula. According to the formula, we also need to multiply the losses by their weights.

# The first neural network is ready. Now let's create the stylization network.
stylizer = keras.Sequential([
    keras.Input(shape=(128, 128, 3)),

    # Extract features while keeping the same spatial size.
    Conv2D(32, 3, padding="same", activation="relu"),

    # Reduce the spatial size: 128 → 64 → 32.
    Conv2D(64, 3, strides=2, padding="same", activation="relu"),
    Conv2D(128, 3, strides=2, padding="same", activation="relu"),

    # Transform features at a spatial size of 32×32.
    Conv2D(128, 3, padding="same", activation="relu"),
    Conv2D(128, 3, padding="same", activation="relu"),

    # Increase the spatial size: 32 → 64.
    keras.layers.UpSampling2D(size=2, interpolation="bilinear"),
    Conv2D(64, 3, padding="same", activation="relu"),

    # Increase the spatial size: 64 → 128.
    keras.layers.UpSampling2D(size=2, interpolation="bilinear"),
    Conv2D(32, 3, padding="same", activation="relu"),

    # An RGB image with values from 0 to 1.
    Conv2D(3, 3, padding="same", activation="sigmoid")
])

stylizer.compile(optimizer="adam", loss=total_loss)

# For the first experiment, take 1000 images from Imagenette.
train_dataset = (prepared_dataset.take(1000).shuffle(1000).batch(4).prefetch(tf.data.AUTOTUNE))
stylizer.fit(train_dataset, epochs=40)
stylizer.save_weights("stylizer.weights.h5")

generated = stylizer(content, training=False)

# Save the result.
plt.imshow(generated[0].numpy())
plt.axis("off")
plt.savefig("comparison.png", dpi=150, bbox_inches="tight")