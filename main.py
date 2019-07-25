import numpy as np
import src.image_generator as ig
import src.song_matrix_generator as smg
from src.models.note_info import NoteInfo

file_name = 'data/PPDD-Sep2018_sym_mono_small/prime_csv/1aea36b4-52ac-4890-9c63-1ddfa5709804.csv'
file_data = np.loadtxt(file_name, delimiter=",")


def noteInfoFactory(note_data):
    return NoteInfo(note_data)


note_infos = list(map(noteInfoFactory, file_data))
song_matrix = smg.generate_song_matrix(note_infos)
ig.sample_images(0)
