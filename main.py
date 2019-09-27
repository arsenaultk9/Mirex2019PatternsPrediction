import numpy as np
from os import listdir

import src.image_generator as ig
import src.song_matrix_generator as smg
import src.song_window_slide_generator as swsg
import src.song_csv_generator as scg
import src.constants as constants

from src.models.note_info import NoteInfo
from src.networks.poly_neural_network import NeuralNetwork

directory_mono = 'data/PPDD-Sep2018_sym_mono_small/prime_csv/'
directory_poly = 'data/PPDD-Sep2018_sym_poly_small/prime_csv/'
directory = directory_poly
file_name_a = 'data/PPDD-Sep2018_sym_mono_small/prime_csv/00b7561d-c09b-41f2-bf21-537603fbe758.csv'
file_name_b = 'data/PPDD-Sep2018_sym_mono_small/prime_csv/0b246118-2c95-4f4d-8e70-56e89f81fda2.csv'

file_names = listdir(directory)  # [file_name_a, file_name_b]
file_names = file_names[0:27]

file_index = 0
X = np.zeros((0, constants.WINDOW_SLIDE_SIZE,
              constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED))
Y = np.zeros((0, constants.ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED))

print('===== Data setup start =====')
for file_name in file_names:
    if(file_name.find(".csv") == -1):
        continue

    file_data = np.loadtxt(directory + file_name, delimiter=",")

    note_infos = list(map(NoteInfo, file_data))
    min_beat_pos, max_beat_pos, song_matrix = smg.generate_song_matrix(
        note_infos)

    if song_matrix.shape[0] < constants.WINDOW_SLIDE_SIZE:
        continue

    cur_X, cur_Y = swsg.generate_window_slide(song_matrix)

    X = np.vstack((X, cur_X))
    Y = np.vstack((Y, cur_Y))

    ig.sample_image('song_matrix_%d' % file_index, song_matrix)
    scg.generate_song_csv('test_%d' % file_index, song_matrix, min_beat_pos)

    file_index += 1

print('===== Data setup end =====')
print('===== Neural network training start =====')

network = NeuralNetwork(X, Y)
network.train()

print('===== Neural network training end =====')

print('===== Generation start =====')

for continuation_index in range(18):
    continuation = network.generate_continuation(cur_X[continuation_index], 16)
    ig.sample_image('song_matrix_continuation_%d' %
                    continuation_index, continuation)
    scg.generate_song_csv('test_continuation_%d' %
                          continuation_index, continuation, max_beat_pos)

print('===== Generation end =====')
