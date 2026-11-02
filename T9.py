import numpy as np
import re
from keras.layers import Dense, SimpleRNN, Input, TextVectorization
from keras.models import Sequential

# Let's start exploring recurrent neural networks. We'll build a network that predicts the next character.
# We'll use a many-to-one setup: six characters go in, and one prediction comes out.
# TextVectorization gives each character an integer ID. We'll turn those IDs into one-hot vectors
# and group them into six-character windows to build our 3D training array.
with open("train_data_true.txt", encoding="utf-8-sig") as f:
    text = f.read().lower().replace("’", "'")
    # Keep English letters and apostrophes. Replace each run of other characters with a space.
    text = re.sub(r"[^a-z']+", " ", text).strip()

alphabet = "qwertyuiopasdfghjklzxcvbnm '" # The order determines the index of each character.
num_characters = len(alphabet) # 28 characters: 26 letters, a space, and an apostrophe.
inp_chars = 6 # Use the last six characters to predict the next one.

tokenizer = TextVectorization(standardize=None, split="character", output_mode="int", vocabulary=list(alphabet))

# Keras reserves ID 0 for padding and ID 1 for an unknown character.
# The 28 characters in our alphabet get IDs 2 through 29, in the order listed above.
# Each row of this 28-by-28 identity matrix is a one-hot vector for one character.
one_hot = np.eye(num_characters, dtype=np.float32)


def encode_chars(s):
    ids = tokenizer([s]).numpy()[0] # Get the sequence of character IDs for this string.
    if np.any(ids < 2):
        raise ValueError("The string contains a character outside the alphabet")
    # Subtract 2 to get row indices from 0 to 27, then look up the one-hot vectors.
    return one_hot[ids - 2]


if len(text) <= inp_chars:
    raise ValueError("Training requires at least 7 characters of text")

data = encode_chars(text)
n = len(data) - inp_chars

# Slide the window forward one character at a time.
# For each window in X, the matching row in Y is the character that comes right after it.
X = np.stack([data[i:i + inp_chars] for i in range(n)])
Y = data[inp_chars:]

print(X.shape, Y.shape) # (n, 6, 28) and (n, 28)

# Now let's build the network.
model = Sequential([Input(shape=(inp_chars, num_characters)), SimpleRNN(128, activation="tanh"), Dense(num_characters, activation="softmax")])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# history = model.fit(X, Y, batch_size=32, epochs=300)
# model.save_weights("T9.weights.h5")
model.load_weights("T9.weights.h5")

rng = np.random.default_rng()


def buildPhrase(inp_str, str_len=50):
    inp_str = inp_str.lower()
    if len(inp_str) < inp_chars:
        raise ValueError(f"Start with at least {inp_chars} characters")

    for _ in range(str_len):
        # Take the last six characters and encode them. The shape is (6, 28).
        window = encode_chars(inp_str[-inp_chars:])

        # None adds a batch dimension: (6, 28) becomes (1, 6, 28).
        # We're passing in one example. The two colons keep all six positions and all 28 features.
        # predict returns shape (1, 28). [0] gives us the 28 probabilities for our example.
        probs = model.predict(window[None, :, :], verbose=0)[0]

        # Normalize the probabilities, then use them to sample the next character.
        p = np.asarray(probs, dtype=np.float64)
        p /= p.sum()
        index = rng.choice(len(alphabet), p=p)
        inp_str += alphabet[index]

    return inp_str


print(buildPhrase("journali"))