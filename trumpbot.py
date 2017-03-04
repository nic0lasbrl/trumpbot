import math
from random import randint
from warnings import warn

import numpy as np
import pandas as pd
from keras.layers import Dense, Activation, LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop

# data import
df = pd.read_csv("./data/trump_tweets.csv", encoding="latin-1")
sequences = df["Tweet_Text"]

chars = sorted(list(set(sequences.str.cat())))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
max_len = max(sequences.str.len())

# padding all tweets to the same length
padd_char = "^"
if padd_char in char_indices:
    warn(padd_char + " is already in the corpus. Try to pick another one for better results.")
else:
    char_indices[padd_char] = len(chars)
    indices_char[len(chars)] = padd_char
    chars += [padd_char]
padd = max_len - sequences.str.len()
padd = padd.apply(lambda i: padd_char * i)
sequences += padd

# building sequences based on the twwets
seq_len = 51
step = 2
n_seq_by_tweet = math.floor((max_len - seq_len) / step)
sequences = sequences.apply(lambda s: [s[i * step: i * step + seq_len] for i in range(n_seq_by_tweet + 1)])
sequences = [seq for list_ in sequences for seq in list_]
next_char = [seq[-1] for seq in sequences]
sequences = [seq[:-1] for seq in sequences]
seq_len = len(sequences[0])

# vectorization
X = np.zeros((len(sequences), seq_len, len(chars)), dtype=np.bool)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
for i, seq in enumerate(sequences):
    for t, char in enumerate(seq):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_char[i]]] = 1

# keras model building
model = Sequential()
model.add(LSTM(128, input_shape=(None, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# get the character and the index of the softmax predictions
# the chosen character is based on the probs given by the softmax
def random_next_char(probas, dic):
    """
        Predicts the next character thanks to the probas given by the softmax function
        :param probas: probas given by the softmax function
        :param dic: dictionary which gives the character corresponding to a position
        :return: the next character and the one hot position
    """
    probs = probas.reshape(probas.shape[1])
    probs = np.round(probs, 5)
    probs = np.maximum(probs, 0)
    probs[-1] = 1.0 - np.sum(probs[:-1])
    try:
        drawn = np.random.multinomial(1, probs, 1)
    except ValueError:
        print("Sum of softmax probas > 1 (Python rounding error), picking the char with the top proba ...")
        drawn = probs
    return dic[np.argmax(drawn)], np.argmax(drawn)


# training + model saving + printing of one example at each epoch
for epoch in range(60):
    print()
    print("-" * 10 + " Training #" + str(epoch) + " " + "-" * 10)
    hist = model.fit(X, y, batch_size=128, nb_epoch=1)
    model.save("data/models/model-" + str(epoch))
    print("Random sentence :")
    print()
    x = np.zeros((1, len(chars)))
    x[-1, randint(0, len(chars) - 1)] = 1
    sentence = ''
    for i in range(max_len):
        probs = model.predict(x.reshape((1, x.shape[0], len(chars))))
        next_char, pos = random_next_char(probs, indices_char)
        sentence += next_char
        x = np.concatenate((x[-seq_len + 1:, :], np.zeros((1, len(chars)))), axis=0)
        x[-1, pos] = 1
    print(sentence)
