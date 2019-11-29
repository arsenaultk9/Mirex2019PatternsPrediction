import numpy as np
from htm.bindings.sdr import SDR

import src.constants as constants

bucket_size = 4


def encode_to_sdr(song_matrix_slice):

    slice_with_buffer = np.zeros(
        (constants.SDR_DIMENSION_LENGTH ** constants.SDR_DIMENSIONS))

    current_bucket_position = 0
    for slice_note_value in song_matrix_slice:
        for bucket_index in range(bucket_size):
            slice_with_buffer[current_bucket_position] = slice_note_value
            current_bucket_position += 1

    # slice_with_buffer[0:constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED] = song_matrix_slice

    slice_as_good_dimension = np.reshape(
        slice_with_buffer, (constants.SDR_DIMENSION_LENGTH, constants.SDR_DIMENSION_LENGTH))
    slice_sdr = SDR(dimensions=(constants.SDR_DIMENSION_LENGTH,
                                constants.SDR_DIMENSION_LENGTH))

    slice_sdr.dense = slice_as_good_dimension.tolist()

    return slice_sdr


def decode_to_note_number(sdr_sparse_data):
    activated_notes = {}

    for index_of_range in range(constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED):
        lower_range = index_of_range * bucket_size
        upper_range = (index_of_range + 1) * bucket_size

        for sparse_data_instance in sdr_sparse_data:
            if(sparse_data_instance >= lower_range and sparse_data_instance < upper_range):
                activated_notes[index_of_range] = 1

    return list(activated_notes.keys())
