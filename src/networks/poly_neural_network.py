import sys
import numpy as np
import random

from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import metrics

import src.constants as constants
import src.note_parser as note_parser
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed

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
    all_preds = np.asarray(preds).astype('float64')

    for preds in all_preds:
        preds_withoutsegments = preds[0:constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED]
        segments = preds[constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED:
                         constants.ALL_NOTE_INPUT_VERTOR_SIZE]

        # If no segment activated take maximum one and activate if
        if(len(preds_withoutsegments[preds_withoutsegments >= 0.5]) == 0):
            preds_withoutsegments[preds_withoutsegments ==
                                  np.max(preds_withoutsegments)] = 0.5

        preds_withoutsegments[preds_withoutsegments < 0.5] = 0

        # Limit maximum note on to 7.
        topPredidctions = []
        for pred in preds_withoutsegments:
            if pred >= 0.5:
                topPredidctions.append(pred)

        topPredidctions.sort(reverse=True)
        topPredidctions = topPredidctions[0:6]

        for topPred in topPredidctions:
            preds_withoutsegments[preds_withoutsegments == topPred] = 1

        preds_withoutsegments[preds_withoutsegments != 1] = 0

        # in case nothing is to predict, predict silence.
        if(len(preds_withoutsegments[preds_withoutsegments >= 0.5]) == 0):
            preds[0] = 1

        # region notes and segments
        top_segment = 0
        for segment in segments:
            top_segment = top_segment if top_segment >= segment else segment

        segments[segments == top_segment] = 1
        segments[segments != 1] = 0

        preds[0:constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED] = preds_withoutsegments
        preds[constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED:
              constants.ALL_NOTE_INPUT_VERTOR_SIZE] = segments

    return all_preds


class NeuralNetwork:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self.model = Sequential()

        # LSTM Encoder for dimensionality reduction of input space and simplification/generalisation of data
        self.model.add(LSTM(512, return_sequences=False, input_shape=(
            X.shape[1], X.shape[2])))

        # Second lstm to decode encoded/dimensionality reduced layer
        self.model.add(RepeatVector(constants.PREDICTION_SIZE))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(LSTM(128, return_sequences=True))

        # Repeat outputs of lstm so each output can pass by a softmax layer to predict on inputs at time step.
        self.model.add(TimeDistributed(Dense(constants.ALL_NOTE_INPUT_VERTOR_SIZE,
                                             activation='sigmoid', name='ouput')))

        optimizer = RMSprop(lr=0.001)

        self.model.compile(loss="binary_crossentropy",
                           optimizer=optimizer,
                           metrics=[metrics.binary_accuracy, metrics.binary_crossentropy])

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

        total_on_per_position = []

        for preds_at_pos in preds:
            preds_at_pos_count = np.size(preds_at_pos[preds_at_pos >= 0.5])
            total_on_per_position.append(preds_at_pos_count)

        total_on_per_position_text = ', '.join(map(str, total_on_per_position))
        print('total on by position: {}'.format(total_on_per_position_text))

    def train(self):
        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end_stats)

        self.model.fit(self.X, self.Y,
                       batch_size=16,
                       epochs=254,
                       shuffle=True,
                       callbacks=[print_callback])

    def generate_continuation(self, last_window_slide,
                              quarter_beats_to_generate):

        total_slides_to_generate = quarter_beats_to_generate * constants.SEGEMENTS_PER_BEAT
        contuation = []

        sequence = last_window_slide

        for index in range(total_slides_to_generate):
            prediction = self.model.predict(np.array([sequence]))[0]
            preds_normalized = sample(prediction)

            # Append following predictions while removing first item for each prediction.
            for cur_preds_normalized in preds_normalized:
                contuation.append(cur_preds_normalized)
                sequence = np.vstack(
                    (sequence[1:sequence.shape[0]], cur_preds_normalized))

        return np.array(contuation)
