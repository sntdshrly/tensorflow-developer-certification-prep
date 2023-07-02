# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.91 and logs.get('val_accuracy') > 0.91:
            print("\n accuracy and validation accuracy > 91%")
            self.model.stop_training = True


def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    labels = bbc["category"].values.tolist()
    sentences = bbc["text"].values.tolist()

    # Split the data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        sentences, labels, train_size=training_portion
    )

    # Tokenize the text
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_data)

    # Convert text to sequences
    training_sequences = tokenizer.texts_to_sequences(train_data)
    training_padded_sequences = pad_sequences(
        training_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type
    )
    validation_sequences = tokenizer.texts_to_sequences(val_data)
    validation_padded_sequences = pad_sequences(
        validation_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type
    )

    # Tokenize the labels
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    training_label_sequences = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_label_sequences = np.array(label_tokenizer.texts_to_sequences(val_labels))

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    callback = MyCallback()
    model.fit(training_padded_sequences, training_label_sequences, epochs=100,
              validation_data=(validation_padded_sequences, validation_label_sequences), callbacks=[callback])
    return model


if __name__ == '__main__':
    model = solution_B4()
    model.save("model_B4.h5")
