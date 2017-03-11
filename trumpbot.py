import math
import string
from random import randint, choice

import numpy as np
import pandas as pd
from keras.layers import Dense, Activation, Dropout, GRU
from keras.models import Sequential

from utils import *

# data import
df = pd.read_csv("./data/trump_tweets.csv", encoding="mbcs")
sequences = df["Tweet_Text"]

chars = sorted(list(set(sequences.str.cat())))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# building sequences based on the tweets + padding
seq_len = 50
step = 2

padd_char_end = "^"
add_char_or_warn(padd_char_end, chars, char_indices, indices_char)
padd = max(sequences.str.len()) - sequences.str.len()
padd = padd.apply(lambda i: padd_char_end * i)
sequences += padd

padd_char_st = "µ"
add_char_or_warn(padd_char_st, chars, char_indices, indices_char)
sequences = padd_char_st * (seq_len - 1) + sequences

max_len = len(sequences[0])
n_seq_by_tweet = math.floor((max_len - seq_len) / step) + 1
sequences = sequences.apply(lambda s: [s[i * step: i * step + seq_len + 1] for i in range(n_seq_by_tweet)])
sequences = [seq for list_ in sequences for seq in list_]
next_char = [seq[-1] for seq in sequences]
sequences = [seq[:-1] for seq in sequences]

# vectorization
X = np.zeros((len(sequences), seq_len, len(chars)), dtype=np.bool)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
for i, seq in enumerate(sequences):
    for t, char in enumerate(seq):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_char[i]]] = 1

# keras model building
model = Sequential()
model.add(GRU(256, input_shape=(seq_len, len(chars)), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(256))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# training + model saving + printing of one example at each epoch
for epoch in range(60):
    print()
    print("-" * 10 + " Training #" + str(epoch) + " " + "-" * 10)
    hist = model.fit(X, y, batch_size=128, nb_epoch=1)
    model.save("data/models/model-" + str(epoch) + ".h5")
    print("Random sentence :")
    print()
    x = np.zeros((seq_len, len(chars)))
    x[:-1, char_indices[padd_char_st]] = 1
    sentence = choice(string.ascii_uppercase)
    x[-1, char_indices[sentence]] = 1
    for i in range(max_len-seq_len-1):
        probs = model.predict(x.reshape((1, x.shape[0], len(chars))))
        next_char, pos = random_next_char(probs, indices_char)
        sentence += next_char
        x = np.concatenate((x[-seq_len + 1:, :], np.zeros((1, len(chars)))), axis=0)
        x[-1, pos] = 1
    sentence = sentence.replace(padd_char_end, "")
    print(sentence)
