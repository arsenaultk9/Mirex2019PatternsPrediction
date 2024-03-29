import numpy as np
from os import listdir

import src.image_generator as ig
import src.song_matrix_generator as smg
import src.song_matrix_note_grouper as smng
import src.song_window_slide_generator as swsg
import src.song_matrix_note_ungrouper as smnu

import src.song_csv_generator as scg
import src.constants as constants

from src.models.note_info import NoteInfo
from src.networks.poly_neural_network import NeuralNetwork

directory_mono = 'data/PPDD-Sep2018_sym_mono_small/prime_csv/'
directory_poly = 'data/PPDD-Sep2018_sym_poly_medium/prime_csv/'
directory = directory_poly
file_name_a = 'data/PPDD-Sep2018_sym_mono_small/prime_csv/00b7561d-c09b-41f2-bf21-537603fbe758.csv'
file_name_b = 'data/PPDD-Sep2018_sym_mono_small/prime_csv/0b246118-2c95-4f4d-8e70-56e89f81fda2.csv'

file_names = listdir(directory)  # [file_name_a, file_name_b]
file_names = file_names[0:199]

file_index = 0
X = np.zeros((0, constants.WINDOW_SLIDE_SIZE,
              constants.ALL_NOTE_INPUT_VECTOR_SIZE))
Y_NOTES = np.zeros((0, constants.PREDICTION_SIZE,
              constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED))
Y_LENGTHS = np.zeros((0, constants.PREDICTION_SIZE,
              constants.SEGMENTS_PER_BEAT))              

song_matrix = None

print('===== Data setup start =====')
for index, file_name in enumerate(file_names):
    if(file_name.find(".csv") == -1):
        continue

    print("Data setup file number {} of {}".format(index, len(file_names)))

    file_data = np.loadtxt(directory + file_name, delimiter=",")

    note_infos = list(map(NoteInfo, file_data))
    min_beat_pos, max_beat_pos, song_matrix = smg.generate_song_matrix(
        note_infos)

    song_matrix_clusters = smng.group_note_clusters(song_matrix)
    if song_matrix.shape[0] < constants.WINDOW_SLIDE_SIZE:
        continue

    cur_X, cur_Y_NOTES, cur_Y_LENGTHS = swsg.generate_window_slide(song_matrix_clusters)

    X = np.vstack((X, cur_X))
    Y_NOTES = np.vstack((Y_NOTES, cur_Y_NOTES))
    Y_LENGTHS = np.vstack((Y_LENGTHS, cur_Y_LENGTHS))

    file_index += 1

print('===== Data setup end =====')


print('===== Neural network training start =====')

network = NeuralNetwork(X, Y_NOTES, Y_LENGTHS)
network.train()

print('===== Neural network training end =====')

print('===== Generation start =====')

for continuation_index in range(42):
    continuation = network.generate_continuation(
        cur_X[continuation_index], int(constants.WINDOW_SLIDE_SIZE / constants.PREDICTION_SIZE))

    continuation_unclustered = smnu.uncluster_song_notes(continuation)

    ig.sample_image('song_matrix_continuation_%d' %
                    continuation_index, continuation_unclustered)
    scg.generate_song_csv('test_continuation_%d' %
                          continuation_index, continuation_unclustered, max_beat_pos)

print('===== Generation end =====')
