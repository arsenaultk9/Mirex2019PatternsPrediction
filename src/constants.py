SEGEMENTS_PER_BEAT = 12
BEAT_SEGMENT_ACCEPTED_ROUNDING_ERROR = 1/50

MIDI_NOTE_COUNT = 128
MIDI_NOTE_AND_SILENCE_COUNT = MIDI_NOTE_COUNT + 1
ALL_POSSIBLE_INPUTS_COUNT = MIDI_NOTE_AND_SILENCE_COUNT + 1

BOTTOM_SKIPPING_INDEX = 0
TOP_SKIPPING_INDEX = 128
ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED = ALL_POSSIBLE_INPUTS_COUNT - \
    ((MIDI_NOTE_COUNT - TOP_SKIPPING_INDEX) + BOTTOM_SKIPPING_INDEX)

ALL_NOTE_INPUT_VERTOR_SIZE = ALL_POSSIBLE_INPUT_BOTTOM_TOP_CHOPPED + SEGEMENTS_PER_BEAT

NOTES_START_OFFSET = 2
SILENCE_INDEX = 1
EMPTY_SONG_DATA_INDEX = 0

# Do about 4 measures of 4/4 for window slide.
WINDOW_SLIDE_SIZE = 16

PREDICTION_SIZE = 2

# HTM
SDR_DIMENSIONS = 2
SDR_DIMENSION_LENGTH = 23
