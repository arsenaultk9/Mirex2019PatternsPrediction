import src.constants as constants
import numpy as np


def generate_window_slide(song_matrix_data):
    X = []
    Y = []

    for current_start_slide in range(constants.WINDOW_SLIDE_SIZE):
        x_window_slide = np.zeros(
            (constants.WINDOW_SLIDE_SIZE, constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED))

        for current_x_pos in range(constants.WINDOW_SLIDE_SIZE - current_start_slide):
            x_window_slide[current_x_pos, constants.EMPTY_SONG_DATA_INDEX] = 1

        start_data = constants.WINDOW_SLIDE_SIZE - current_start_slide - 1
        x_window_slide[start_data: constants.WINDOW_SLIDE_SIZE] = song_matrix_data[start_data: constants.WINDOW_SLIDE_SIZE]

        X.append(x_window_slide)
        Y.append(song_matrix_data[current_start_slide + 1])

    remaining_slides = song_matrix_data.shape[0] - constants.WINDOW_SLIDE_SIZE

    for current_slide in range(remaining_slides - 1):
        X.append(
            song_matrix_data[current_slide:current_slide+constants.WINDOW_SLIDE_SIZE])

        Y.append(song_matrix_data[current_slide +
                                  constants.WINDOW_SLIDE_SIZE+1])

    return np.array(X), np.array(Y)
