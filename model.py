import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing import sequence



# Load the IMDb movie review dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Preprocess the dataset by truncating or padding sequences to a fixed length
max_review_length = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

embedding_vector_length = 32
model = Sequential()
model.add(Embedding(10000, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dropout(0.5))
# model.add(LSTM(100))
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=64)


scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

