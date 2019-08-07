notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
notes_and_octave = dict({})

notes_and_octave[0] = '(#)'
current_index = 1

for octave in range(11):
    for note in notes:
        notes_and_octave[current_index] = note + str(octave)
        current_index += 1


def parse_number_to_note(number):
    return notes_and_octave[number]
