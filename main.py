import numpy as np
import src.image_generator as ig
import src.song_matrix_generator as smg
import src.song_window_slide_generator as swsg
from src.models.note_info import NoteInfo

file_name = 'data/PPDD-Sep2018_sym_poly_small/prime_csv/cee62162-bf14-477c-81fe-064615fbec68.csv'
file_data = np.loadtxt(file_name, delimiter=",")

note_infos = list(map(NoteInfo, file_data))
song_matrix = smg.generate_song_matrix(note_infos)
X, Y = swsg.generate_window_slide(song_matrix)

ig.sample_image('song_matrix_2', X[0, 40:90])
