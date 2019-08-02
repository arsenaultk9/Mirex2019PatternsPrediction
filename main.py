import numpy as np
import src.image_generator as ig
import src.song_matrix_generator as smg
import src.song_window_slide_generator as swsg
import src.song_csv_generator as scg
import src.constants as constants
from src.models.note_info import NoteInfo

file_name_a = 'data/PPDD-Sep2018_sym_mono_small/prime_csv/00b7561d-c09b-41f2-bf21-537603fbe758.csv'
file_name_b = 'data/PPDD-Sep2018_sym_poly_small/prime_csv/0a240995-eb9b-4070-b859-0c3eba04fa04.csv'

file_names = [file_name_a, file_name_b]

file_index = 0
X = np.zeros((0, constants.MIDI_NOTE_COUNT, constants.WINDOW_SLIDE_SIZE))
Y = np.zeros((0, constants.MIDI_NOTE_COUNT))

for file_name in file_names:
    file_data = np.loadtxt(file_name, delimiter=",")

    note_infos = list(map(NoteInfo, file_data))
    song_matrix = smg.generate_song_matrix(note_infos)
    cur_X, cur_Y = swsg.generate_window_slide(song_matrix)

    X = np.vstack((X, cur_X))
    Y = np.vstack((Y, cur_Y))

    ig.sample_image('song_matrix_%d' % file_index, X[0, 40:90])
    scg.generate_song_csv('test_%d' % file_index, song_matrix)
