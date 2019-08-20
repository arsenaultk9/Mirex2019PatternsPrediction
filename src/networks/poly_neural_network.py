import sys
import numpy as np
import random

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras import metrics

import src.constants as constants
import src.note_parser as note_parser

maxlen = 31


def one_hot_encoding_to_music_sequence(segment):
    text = ''

    for row in segment:
        text += '('
        for index in range(len(row)):
            if(row[index] == 1):
                text += note_parser.parse_number_to_note(index) + ' '

        text += ')'

    return text


def sample(preds):
    predict_from_threshold = np.zeros(preds.shape)

    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0

    # in case nothing is to predict, predict silence.
    if(len(preds[preds >= 0.5]) == 0):
        preds[0] = 1

    return preds


class NeuralNetwork:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=False, input_shape=(
            X.shape[1], X.shape[2])))
        # self.model.add(Dropout(0.2))
        # self.model.add(LSTM(128))
        self.model.add(Dense(constants.MIDI_NOTE_AND_SILENCE_COUNT,
                             activation='sigmoid', name='ouput'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss={"ouput": "binary_crossentropy"},
                           optimizer=sgd,
                           metrics={"ouput": [metrics.binary_accuracy, metrics.binary_crossentropy]})

    def on_epoch_end(self, epoch, _):
        if(epoch < 45):
            return

        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(self.X) - maxlen - 1)
        generated = ''
        segment = self.X[start_index]
        generated += one_hot_encoding_to_music_sequence(segment)
        sys.stdout.write(generated + " | ")

        for i in range(64):
            x_pred = np.array([segment])

            preds = self.model.predict(x_pred, verbose=0)[0]
            preds_normalized = sample(preds)

            generated += ', ' + \
                one_hot_encoding_to_music_sequence([preds_normalized])

            segment = np.vstack(
                (segment[1:segment.shape[0]], preds_normalized))

            sys.stdout.write(
                one_hot_encoding_to_music_sequence([preds_normalized]) + ', ')
            sys.stdout.flush()
        print()

    def on_epoch_end_stats(self, epoch, _):
        print()
        print('----- Generating stats after Epoch: %d' % epoch)

        start_index = random.randint(0, len(self.X) - maxlen - 1)
        segment = self.X[start_index]

        x_pred = np.array([segment])
        preds = self.model.predict(x_pred, verbose=0)[0]
        min_pred = preds.min()
        max_pred = preds.max()
        mean_pred = preds.mean()
        sum_pred = preds.sum()

        print('min: %f, max: %f, mean: %f, total: %f' %
              (min_pred, max_pred, mean_pred, sum_pred))

    def train(self):
        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end_stats)

        self.model.fit(self.X, self.Y,
                       batch_size=128,
                       epochs=128,
                       shuffle=False,
                       callbacks=[print_callback])

    def generate_continuation(self, last_window_slide,
                              quarter_beats_to_generate):

        total_slides_to_generate = quarter_beats_to_generate * constants.SEGEMENTS_PER_BEAT
        contuation = []

        sequence = last_window_slide

        for index in range(total_slides_to_generate):
            prediction = self.model.predict(np.array([sequence]))[0]
            preds_normalized = sample(prediction)

            contuation.append(preds_normalized)

            sequence = np.vstack(
                (sequence[1:sequence.shape[0]], preds_normalized))

        return np.array(contuation)
