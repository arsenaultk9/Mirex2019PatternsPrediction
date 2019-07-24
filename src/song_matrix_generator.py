import sys


# With 12 quadrants per beat we cover each situation
# For none triplet notes smallest denomination is 0.25 or 1/4
# For triplet notes smallest denomination is 0.083 (1/12) but mostly  0.1666 (1/6)
# So with 12 quadrant per beat we cover all cases of dataset (small).
def generate_song_matrix(note_infos):
    min_beat_pos = sys.maxsize
    max_beat_pos = 0

    for note_info in note_infos:
        if(note_info.starting_beat <= min_beat_pos):
            min_beat_pos = note_info.starting_beat

        note_end = note_info.starting_beat + note_info.length
        if(note_end >= max_beat_pos):
            max_beat_pos = note_end

    return ''
