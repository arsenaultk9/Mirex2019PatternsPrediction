import numpy as np
import random
from datetime import datetime

from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from keras.callbacks import TensorBoard

import src.constants as constants

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)

maxlen = 31


# helper function to sample an index from a probability array
def sample(nn_preds):
    preds_adjusted = np.zeros((constants.PREDICTION_SIZE,
              constants.ALL_NOTE_INPUT_VECTOR_SIZE))

    notes_preds = np.asarray(nn_preds[0]).astype('float64')

    for index, preds in enumerate(notes_preds[0]):
        preds_withoutsegments = preds

        # If no segment activated take maximum one and activate if
        if(len(preds_withoutsegments[preds_withoutsegments >= 0.5]) == 0):
            preds_withoutsegments[preds_withoutsegments ==
                                  np.max(preds_withoutsegments)] = 0.5

        preds_withoutsegments[preds_withoutsegments < 0.5] = 0

        # Limit maximum note on to 7.
        topPredictions = []
        for pred in preds_withoutsegments:
            if pred >= 0.5:
                topPredictions.append(pred)

        topPredictions.sort(reverse=True)
        topPredictions = topPredictions[0:6]

        for topPred in topPredictions:
            preds_withoutsegments[preds_withoutsegments == topPred] = 1

        preds_withoutsegments[preds_withoutsegments != 1] = 0

        # in case nothing is to predict, predict silence.
        if(len(preds_withoutsegments[preds_withoutsegments >= 0.5]) == 0):
            preds[0] = 1

        # region notes and segments

        preds_adjusted[index, 0:constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED] = preds_withoutsegments
        

    segment_preds = np.asarray(nn_preds[1]).astype('float64')
    for index, preds in enumerate(segment_preds[0]):
        segments = preds
        top_segment = 0

        for segment in segments:
            top_segment = top_segment if top_segment >= segment else segment

        segments[segments == top_segment] = 1
        segments[segments != 1] = 0

        preds_adjusted[index, constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED:
              constants.ALL_NOTE_INPUT_VECTOR_SIZE] = segments

    return preds_adjusted


class NeuralNetwork:
    def __init__(self, X, Y_NOTES, Y_LENGTHS):
        self.X = X
        self.Y_NOTES = Y_NOTES
        self.Y_LENGTHS = Y_LENGTHS

        input_layer = Input((X.shape[1], X.shape[2]))

        # LSTM Encoder for dimensionality reduction of input space and simplification/generalization of data
        start_encoder_layer = LSTM(512, return_sequences=False)(input_layer)

        # Second lstm to decode encoded/dimensionality reduced layer
        repeat_decoder_layer = RepeatVector(constants.PREDICTION_SIZE)(start_encoder_layer)
        second_lstm_layer = LSTM(128, return_sequences=True)(repeat_decoder_layer)

        # Repeat outputs of lstm so each output can pass by a softmax layer to predict on inputs at time step.
        notes_output_layer = TimeDistributed(Dense(constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED,
                                             activation='sigmoid', name='notes_output'))(second_lstm_layer)

        lengths_output_layer = TimeDistributed(Dense(constants.SEGMENTS_PER_BEAT,
                                             activation='sigmoid', name='length_output'))(second_lstm_layer)

        self.model = Model(input_layer, [notes_output_layer, lengths_output_layer])
        

        optimizer = RMSprop(lr=0.001)

        self.model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
                           optimizer=optimizer,
                           metrics={ 'time_distributed' : [metrics.binary_accuracy, metrics.binary_crossentropy], 'time_distributed_1': metrics.categorical_accuracy })

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

        self.model.fit(self.X, [self.Y_NOTES, self.Y_LENGTHS],
                       batch_size=64,
                       epochs=256,
                       shuffle=True,
                       validation_split= 0.2,
                       callbacks=[print_callback, tensorboard_callback])

    def generate_continuation(self, last_window_slide,
                              quarter_beats_to_generate):

        total_slides_to_generate = quarter_beats_to_generate * constants.SEGMENTS_PER_BEAT
        continuation = []

        sequence = last_window_slide

        for index in range(total_slides_to_generate):
            prediction = self.model.predict(np.array([sequence]))
            preds_normalized = sample(prediction)

            # Append following predictions while removing first item for each prediction.
            for cur_preds_normalized in preds_normalized:
                continuation.append(cur_preds_normalized)
                sequence = np.vstack(
                    (sequence[1:sequence.shape[0]], cur_preds_normalized))

        return np.array(continuation)
