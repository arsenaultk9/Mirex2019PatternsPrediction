import numpy as np
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM

import src.constants as constants
import src.note_parser as note_parser


def formatToNotes(sdr):
    s = ''

    for activated in sdr.sparse:
        s += note_parser.parse_number_to_note(activated) + ' '

    return s


def printStateTM(tm):
    # Useful for tracing internal states
    print("Active cells     " + formatToNotes(tm.getActiveCells()))
    print("Winner cells     " + formatToNotes(tm.getWinnerCells()))
    tm.activateDendrites(True)
    print("Predictive cells " + formatToNotes(tm.getPredictiveCells()))
    print("Anomaly", tm.anomaly * 100, "%")
    print("")


class HtmNetwork:
    def __init__(self, song_matrix):
        self.song_matrix = song_matrix[0:80]
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

        self.tm = TM(columnDimensions=(constants.SDR_DIMENSION_LENGTH, constants.SDR_DIMENSION_LENGTH),
                     cellsPerColumn=1, initialPermanence=0.5, connectedPermanence=0.5,
                     minThreshold=8, maxNewSynapseCount=20, permanenceIncrement=0.1, permanenceDecrement=0.0, activationThreshold=8)

        self.tm.printParameters()

    def train(self):
        for training_iteration in range(2):
            print('===================== training iteration {} =====================',
                  training_iteration)
            for song_slice in self.song_slices:
                self.tm.compute(song_slice, learn=True)
                printStateTM(self.tm)

        return

    def generate_continuation(self, last_window_slide,
                              quarter_beats_to_generate):
        return
