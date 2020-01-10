import numpy as np
import src.constants as constants


def get_length_of_cluster(clustered_song_note):
    cluster_size = 1

    for cluster_pointer in clustered_song_note[constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED:
                                               constants.ALL_NOTE_INPUT_VERTOR_SIZE]:
        if cluster_pointer == 1:
            return cluster_size

        cluster_size += 1

    return 12


def uncluster_song_notes(clustered_song_notes):
    unclustered_song_notes = []

    for clustered_song_note in clustered_song_notes:
        cluster_length = get_length_of_cluster(clustered_song_note)

        for cluster_index in range(cluster_length):
            unclustered_song_notes.append(
                clustered_song_note[0:constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED])

    return np.array(unclustered_song_notes)
