notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
octave_symboles = ['-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
notes_and_octave = dict({})

notes_and_octave[0] = '---'
notes_and_octave[1] = '(#)'
current_index = 2

# TODO: manage window slide vs. song matrix. (window slide has an extra symbole).
for octave_symbole in octave_symboles:
    for note in notes:
        notes_and_octave[current_index] = note + octave_symbole
        current_index += 1


def parse_number_to_note(number):
    return notes_and_octave[number]
