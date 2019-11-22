import numpy as np
from htm.bindings.sdr import SDR

import src.constants as constants


class HtmNetwork:
    def __init__(self, song_matrix):
        self.song_matrix = song_matrix
        self.song_slices = []

        for current_slice in self.song_matrix:
            slice_with_buffer = np.zeros(
                (constants.SDR_DIMENSION_LENGTH ** constants.SDR_DIMENSIONS))

            slice_with_buffer[0:constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED] = current_slice

            slice_as_good_dimension = np.reshape(
                slice_with_buffer, (constants.SDR_DIMENSION_LENGTH, constants.SDR_DIMENSION_LENGTH))
            slice_sdr = SDR(dimensions=(
                constants.SDR_DIMENSION_LENGTH, constants.SDR_DIMENSION_LENGTH))

            slice_sdr.dense = slice_as_good_dimension.tolist()
            self.song_slices.append((slice_sdr))

    def train(self):
        return

    def generate_continuation(self, last_window_slide,
                              quarter_beats_to_generate):
        return
