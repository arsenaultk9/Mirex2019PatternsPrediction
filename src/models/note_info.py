import src.constants as constants

# Data columns:
# Starting beat(Quarter notes),
# Pitch (0, 127),
# relative Pitch,
# Length(Quarter notes),
# Midi Channel


class NoteInfo:
    def __init__(self, note_data):
        print(note_data[0])
        self.starting_beat = float(note_data[0])
        self.pitch = note_data[1]
        self.relative_pitch = note_data[2]
        self.length = float(note_data[3])
        self.midi_channel = note_data[4]

    def is_on_at_beat(self, beat):
        if(beat < self.starting_beat - constants.BEAT_SEGMENT_ACCEPTED_ROUNDING_ERROR):
            return False

        inclusive_end = self.starting_beat + \
            self.length - (1/constants.SEGEMENTS_PER_BEAT) + \
            constants.BEAT_SEGMENT_ACCEPTED_ROUNDING_ERROR

        if(beat > inclusive_end):
            return False

        return True

    @classmethod
    def create(cls, starting_beat, length=1, pitch=0):
        return cls([
            starting_beat,
            pitch,
            pitch,
            length,
            0
        ])
