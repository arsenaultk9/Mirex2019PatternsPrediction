import sys
import numpy as np
import random

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras import metrics

import src.constants as constants
import src.note_parser as note_parser

maxlen = 31


def one_hot_encoding_to_music_sequence(segment):
    text = ''
    for row in segment:
        for index in range(len(row)):
            if(row[index] == 1):
                text += note_parser.parse_number_to_note(index) + ' '

    return text


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


class NeuralNetwork:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=False, input_shape=(
            X.shape[1], X.shape[2])))
        # self.model.add(LSTM(128))
        self.model.add(Dense(constants.MIDI_NOTE_AND_SILENCE_COUNT,
                             activation='softmax', name='ouput'))

        optimizer = RMSprop(lr=0.001)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=[metrics.mae, metrics.categorical_accuracy])

    def on_epoch_end(self, epoch, _):
        if(epoch < 72):
            return

        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(self.X) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            segment = self.X[start_index]
            generated += one_hot_encoding_to_music_sequence(segment)
            sys.stdout.write(generated + " | ")

            for i in range(64):
                x_pred = np.array([segment])

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)

                # generated += ', ' + str(next_index)

                next_note_arr = np.zeros(
                    (constants.MIDI_NOTE_AND_SILENCE_COUNT))
                next_note_arr[next_index] = 1

                segment = np.vstack(
                    (segment[1:segment.shape[0]], next_note_arr))

                sys.stdout.write(
                    note_parser.parse_number_to_note(next_index) + ' ')
                sys.stdout.flush()
            print()

    def train(self):
        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)

        self.model.fit(self.X, self.Y,
                       batch_size=256,
                       epochs=61,  # 240
                       shuffle=False,
                       callbacks=[print_callback])

    def generate_continuation(self, last_window_slide,
                              quarter_beats_to_generate):

        total_slides_to_generate = quarter_beats_to_generate * constants.SEGEMENTS_PER_BEAT
        contuation = []

        sequence = last_window_slide

        for index in range(total_slides_to_generate):
            prediction = self.model.predict(np.array([sequence]))[0]
            next_index = sample(prediction, 0.2)
            next_note_arr = np.zeros(
                (constants.MIDI_NOTE_AND_SILENCE_COUNT))
            next_note_arr[next_index] = 1
            contuation.append(next_note_arr)

            sequence = np.vstack(
                (sequence[1:sequence.shape[0]], next_note_arr))

        return np.array(contuation)
