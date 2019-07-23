class NoteInfo:
    # Data columns: Starting beat(Quarter notes), Pitch (0, 127), relative Pitch, Length(Quarter notes), Midi Channel

    def __init__(self, note_data):
        self.starting_beat = note_data[0]
        self.pitch = note_data[1]
        self.relative_pitch = note_data[2]
        self.length = note_data[3]
        self.midi_channel = note_data[4]
