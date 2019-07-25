import numpy as np
import src.image_generator as ig
import src.song_matrix_generator as smg
from src.models.note_info import NoteInfo

file_name = 'data/PPDD-Sep2018_sym_poly_small/prime_csv/0a240995-eb9b-4070-b859-0c3eba04fa04.csv'
file_data = np.loadtxt(file_name, delimiter=",")

note_infos = list(map(NoteInfo, file_data))
song_matrix = smg.generate_song_matrix(note_infos)
ig.sample_image('song_matrix', song_matrix)
