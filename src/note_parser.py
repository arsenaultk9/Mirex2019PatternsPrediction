notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
notes_and_octave = dict({})

current_index = 0
for note in notes:
    for octave in range(11):
        notes_and_octave[current_index] = note + str(octave)
        current_index += 1


def parse_number_to_note(number):
    return notes_and_octave[number]
