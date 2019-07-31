import src.constants as constants
import numpy as np


def generate_window_slide(song_matrix_data):
    total_slides = song_matrix_data.shape[1] - constants.WINDOW_SLIDE_SIZE

    X = []
    Y = []

    for current_slide in range(total_slides - 1):
        X.append(
            song_matrix_data[:, current_slide:current_slide+constants.WINDOW_SLIDE_SIZE])

        Y.append(song_matrix_data[:, current_slide +
                                  constants.WINDOW_SLIDE_SIZE+1])

    return np.array(X), np.array(Y)
