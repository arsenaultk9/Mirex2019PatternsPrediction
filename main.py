import numpy as np
import src.image_generator as ig
import src.song_matrix_generator as smg
import src.song_window_slide_generator as swsg
import src.song_csv_generator as scg
import src.constants as constants
from src.models.note_info import NoteInfo
from src.neural_network import NeuralNetwork

file_name_a = 'data/PPDD-Sep2018_sym_mono_small/prime_csv/00b7561d-c09b-41f2-bf21-537603fbe758.csv'
file_name_b = 'data/PPDD-Sep2018_sym_mono_small/prime_csv/0b246118-2c95-4f4d-8e70-56e89f81fda2.csv'

file_names = [file_name_a, file_name_b]

file_index = 0
X = np.zeros((0, constants.WINDOW_SLIDE_SIZE,
              constants.MIDI_NOTE_AND_SILENCE_COUNT))
Y = np.zeros((0, constants.MIDI_NOTE_AND_SILENCE_COUNT))

for file_name in file_names:
    file_data = np.loadtxt(file_name, delimiter=",")

    note_infos = list(map(NoteInfo, file_data))
    min_beat_pos, song_matrix = smg.generate_song_matrix(note_infos)
    cur_X, cur_Y = swsg.generate_window_slide(song_matrix)

    X = np.vstack((X, cur_X))
    Y = np.vstack((Y, cur_Y))

    ig.sample_image('song_matrix_%d' % file_index, song_matrix)
    scg.generate_song_csv('test_%d' % file_index, song_matrix, min_beat_pos)

    file_index += 1

network = NeuralNetwork(X, Y)
network.train()
