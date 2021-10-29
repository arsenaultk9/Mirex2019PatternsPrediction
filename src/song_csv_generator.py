import csv
import numpy as np

import src.constants as constants

# Data columns:
# Starting beat(Quarter notes),
# Pitch (0, 127),
# relative Pitch,
# Length(Quarter notes),
# Midi Channel


def generate_song_csv(file_name, song_matrix, start_point):
    file_path = "generated/%s.csv" % file_name
    notes = []

    for note_index in range(song_matrix.shape[1]):
        # Skip silence note
        if note_index == 0:
            continue

        current_pos_in_quadrant = 0
        started = False
        started_pos = 0

        for current_pos_in_quadrant in range(song_matrix.shape[0]):
            if song_matrix[current_pos_in_quadrant, note_index] == 0 and \
                    not started:
                continue

            if song_matrix[current_pos_in_quadrant, note_index] == 0 and \
                    started:

                ended_pos = current_pos_in_quadrant / constants.SEGMENTS_PER_BEAT + start_point
                note = np.array([started_pos,
                                 int(note_index - 1),
                                 int(note_index - 1),
                                 ended_pos - started_pos,
                                 0])
                notes.append(note)
                started = False

                continue

            if song_matrix[current_pos_in_quadrant, note_index] == 1 and \
                    not started:
                started_pos = current_pos_in_quadrant / \
                    constants.SEGMENTS_PER_BEAT + start_point
                started = True

    if(len(notes) > 0):
        notes = np.array(notes)
        notes = notes[notes[:, 0].argsort()]

    np.savetxt(file_path, notes, delimiter=',', fmt='%1.3f')
