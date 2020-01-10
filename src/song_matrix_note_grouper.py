import numpy as np
import src.constants as constants


def group_note_clusters(song_matrix):
    song_note_clusters = []

    current_cluster_size = 0

    # In order to not segment first slice from rest
    last_song_slice = song_matrix[0]

    current_cluster = np.zeros(constants.ALL_NOTE_INPUT_VERTOR_SIZE)
    for song_slice in song_matrix:
        if(np.array_equal(last_song_slice, song_slice) and
                current_cluster_size + 1 < constants.SEGEMENTS_PER_BEAT):

            current_cluster_size += 1
            last_song_slice = song_slice
            continue

        current_cluster[0:constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED] = last_song_slice
        current_cluster[constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED +
                        current_cluster_size] = 1

        song_note_clusters.append(current_cluster)
        current_cluster = np.zeros(constants.ALL_NOTE_INPUT_VERTOR_SIZE)

        last_song_slice = song_slice

    return np.array(song_note_clusters)
