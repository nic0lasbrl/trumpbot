import numpy as np
from warnings import warn

def add_char_or_warn(char, chars, char_indices, indices_char):
    """
        Checks if char is already in the chars character corpus,
        if yes, warns the user
        if not, adds the new char
    """
    if char in char_indices:
        warn(char + " is already in the corpus. Try to pick another one for better results.")
    else:
        char_indices[char] = len(chars)
        indices_char[len(chars)] = char
        chars += [char]


def random_next_char(probas, dic, verbose=True):
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
        if verbose:
            print("Sum of softmax probas > 1 (Python rounding error), picking the char with the top proba ...")
        drawn = probs
    return dic[np.argmax(drawn)], np.argmax(drawn)
