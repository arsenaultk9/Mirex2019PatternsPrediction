import numpy as np
import src.image_generator as ig
import src.song_matrix_generator as smg
import src.song_window_slide_generator as swsg
import src.song_csv_generator as scg
from src.models.note_info import NoteInfo

file_name = 'data/PPDD-Sep2018_sym_mono_small/prime_csv/00b7561d-c09b-41f2-bf21-537603fbe758.csv'
file_data = np.loadtxt(file_name, delimiter=",")

note_infos = list(map(NoteInfo, file_data))
song_matrix = smg.generate_song_matrix(note_infos)
X, Y = swsg.generate_window_slide(song_matrix)

ig.sample_image('song_matrix_2', X[0, 40:90])
scg.generate_song_csv('test', song_matrix)
