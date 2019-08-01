import csv
import numpy as np

import src.constants as constants

# Data columns:
# Starting beat(Quarter notes),
# Pitch (0, 127),
# relative Pitch,
# Length(Quarter notes),
# Midi Channel


def generate_song_csv(file_name, song_matrix):
    file_path = "generated/%s.csv" % file_name
    notes = []

    for note_index in range(song_matrix.shape[0]):
        current_pos_in_quadrant = 0
        started = False
        started_pos = 0

        for current_pos_in_quadrant in range(song_matrix.shape[1]):
            if song_matrix[note_index, current_pos_in_quadrant] == 0 and \
                    not started:
                continue

            if song_matrix[note_index, current_pos_in_quadrant] == 0 and \
                    started:

                ended_pos = current_pos_in_quadrant / constants.SEGEMENTS_PER_BEAT
                note = np.array([started_pos,
                                 int(note_index),
                                 int(note_index),
                                 ended_pos - started_pos,
                                 0])
                notes.append(note)
                started = False

                continue

            if song_matrix[note_index, current_pos_in_quadrant] == 1 and \
                    not started:
                started_pos = current_pos_in_quadrant / constants.SEGEMENTS_PER_BEAT
                started = True

    notes = np.array(notes)
    notes = notes[notes[:, 0].argsort()]
    np.savetxt(file_path, notes, delimiter=',', fmt='%1.3f')
