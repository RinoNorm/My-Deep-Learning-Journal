import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Flatten, Dense, Dropout
from keras.datasets import mnist
from keras import Sequential

import tkinter as tk
from PIL import Image, ImageDraw

# A neural network that recognizes handwritten digits
# The goal is to independently build a neural network for recognizing handwritten digits using Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data() # Load training and test images of handwritten digits

plt.figure(figsize=(10, 5)) # Create a 10 by 5 window

for i in range(10): # Display the first 10 digits
    plt.subplot(2, 5, i + 1) # Select a cell in a grid with two rows and five columns
    plt.imshow(x_train[i], cmap="gray") # Display the digits themselves
plt.show()

# Now the task is to build the neural network. To do this, create an input layer consisting of 784 pixels
# Use one hidden layer, since two or more would be excessive and make overfitting harder to prevent in this implementation
# The hidden layer will contain 200 neurons
# This should be enough for reasonably accurate training without overloading the computer or overfitting, provided the parameters are chosen correctly
# Use the ReLU activation function in the hidden layers. It is well suited for training because it can handle any numbers, replacing negative values with zeros
# Add Dropout because the network overfits without it. After several test runs, a rate of 0.5 proved to be the best fit for this network
# The output layer will contain 10 neurons. Each neuron will correspond to one digit. For example, if the network identifies the digit 0, the first neuron will contain 1 and all the others will contain zeros
# Use softmax in the output layer, as is typical for object classification with a neural network
model = Sequential([Flatten(input_shape=(28, 28)), Dense(200, activation="relu"), Dropout(0.5), Dense(10, activation="softmax")])

# Here we train the neural network. Use the Adam optimizer, which is sufficient for this task. Cross-entropy is a standard loss function for softmax
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Now normalize pixel brightness values to the 0–1 range. To do this, divide the training and test sets by the maximum value of an unsigned 8-bit integer
x_train = x_train / 255
x_test = x_test / 255
# Train the model. Using 1600 batches per epoch produced a very good test loss of 0.0751 without overfitting. I decided to keep this configuration
# Use 10 epochs because increasing the number of epochs does not improve the result
model.fit(x_train, y_train, batch_size=30, epochs=10, validation_split=0.2)

model.evaluate(x_test, y_test) # Evaluate the trained neural network on the MNIST test set

# Create a window where we can draw digits for recognition. Since I had never worked with Tkinter before, GPT astra helped me write the code below
window = tk.Tk() # Call a class from the Tkinter library
window.title("Number recognizer") # Window title
window.resizable(False, False) # Prevent the user from resizing the window

canvas = tk.Canvas(window, width=280, height=280, bg="black", highlightthickness=0) # Set the parameters of the canvas inside the window
canvas.pack(padx=10, pady=10)

image = Image.new("L", (280, 280), color=0) # Create a black image in memory
image_draw = ImageDraw.Draw(image) # A drawing tool created using ImageDraw
last_point = None # Store the previous mouse coordinates. Drawing has not started yet, so there are no coordinates — hence None

# Now define functions that work with coordinates in the window
def start_draw(event):
    global last_point
    last_point = (event.x, event.y)

    r = 9
    canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="white", outline="white")
    image_draw.ellipse((event.x - r, event.y - r, event.x + r, event.y + r), fill=255)

def draw(event):
    global last_point

    if last_point is not None:

        image_draw.line((last_point[0], last_point[1], event.x, event.y), fill=255, width=18)
        image_draw.ellipse((event.x - 9, event.y - 9, event.x + 9, event.y + 9), fill=255)
        canvas.create_line(last_point[0], last_point[1], event.x, event.y, fill="white", width=18, capstyle=tk.ROUND, smooth=True)

    last_point = (event.x, event.y)

def stop_draw(event):
    global last_point
    last_point = None

def clear_canvas():
    global last_point
    image.paste(0, (0, 0, 280, 280))
    canvas.delete("all")
    last_point = None

canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_draw)

clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.pack(pady=(0, 10))

def recognize():
    bounds = image.getbbox()

    if bounds is None:
        result_label.config(text="Draw a number")
        return

    digit = image.crop(bounds)

    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    prepared = Image.new("L", (28, 28), color=0)
    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2
    prepared.paste(digit, (x, y))

    pixels = np.asarray(prepared, dtype=np.float32) / 255.0
    pixels = np.expand_dims(pixels, axis=0)

    probabilities = model.predict(pixels, verbose=0)
    number = np.argmax(probabilities[0])

    result_label.config(text=f"Recognized: {number}")

recognize_button = tk.Button(window, text="Recognize", command=recognize)
recognize_button.pack(pady=5)

result_label = tk.Label(window, text="Draw a number")
result_label.pack(pady=(0, 10))

window.mainloop()