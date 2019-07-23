import numpy as np
import src.image_generator as ig
from src.models.note_info import NoteInfo

file_name = 'data/PPDD-Sep2018_sym_mono_small/prime_csv/1aea36b4-52ac-4890-9c63-1ddfa5709804.csv'

# Data columns: Starting beat(Quarter notes), Pitch (0, 127), relative Pitch, Length(Quarter notes), Midi Channel
file_data = np.loadtxt(file_name, delimiter=",")


def noteInfoFactory(note_data):
    return NoteInfo(note_data)


note_infos = list(map(noteInfoFactory, file_data))
ig.sample_images(0)
