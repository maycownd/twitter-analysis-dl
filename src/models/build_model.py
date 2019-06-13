from keras.layers import Embedding, Flatten, Dropout, Dense, Conv1D, MaxPooling1D
from keras.models import Sequential


def baseline_model(vocab_size=7267, input_length=16):
    model = Sequential()
    model.add(Embedding(vocab_size, 8, input_length=input_length, trainable=False))
    model.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    return model


