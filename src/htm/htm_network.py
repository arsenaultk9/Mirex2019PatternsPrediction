import numpy as np
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM

import src.constants as constants
import src.note_parser as note_parser
import src.htm.simple_scalar_encoder as sse


def formatToNotes(sdr):
    s = ''

    for activated in sse.decode_to_note_number(sdr.sparse):
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
        self.song_matrix = song_matrix
        self.song_slices = []

        # TODO: better sdr generation to not include duplicates of window slides.
        for song_matrix_slice in song_matrix:
            slice_sdr = sse.encode_to_sdr(song_matrix_slice)
            self.song_slices.append((slice_sdr))

        self.tm = TM(columnDimensions=(constants.SDR_DIMENSION_LENGTH, constants.SDR_DIMENSION_LENGTH),
                     cellsPerColumn=1, initialPermanence=0.5, connectedPermanence=0.5,
                     minThreshold=8, maxNewSynapseCount=20, permanenceIncrement=0.1, permanenceDecrement=0.0, activationThreshold=8)

        self.tm.printParameters()

    def train(self):
        for training_iteration in range(3):
            print('===================== training iteration {} =====================',
                  training_iteration)
            for song_slice in self.song_slices[0:20]:
                self.tm.compute(song_slice, learn=True)
                printStateTM(self.tm)

            self.tm.reset()
        return

    def generate_continuation(self, last_window_slide,
                              quarter_beats_to_generate):
        return
