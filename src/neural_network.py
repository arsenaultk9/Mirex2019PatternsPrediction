from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop


class NeuralNetwork:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(
            X.shape[1], X.shape[2]), name='input'))
        self.model.add(Dense(128, activation='softmax', name='ouput'))

        optimizer = RMSprop(lr=0.001)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer)

    def train(self):
        self.model.fit(self.X, self.Y,
                       batch_size=128,
                       epochs=240,
                       shuffle=False)
