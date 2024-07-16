import tensorflow as tf
import numpy as np
import pandas as pd
import json, pickle, string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder

# Load intents
with open('chatbot/intents.json', 'r') as file:
    data1 = json.load(file)

# Prepare data
tags = []
inputs = []
responses = {}

for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']
    for line in intent['patterns']:
        inputs.append(line)
        tags.append(intent['tag'])

data = pd.DataFrame({"inputs": inputs, "tags": tags})

# Preprocess data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])
x_train = pad_sequences(train)

le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]
vocabulary = len(tokenizer.word_index) + 1
output_length = le.classes_.shape[0]
embedding_dim = 100

# Build model
i = Input(shape=(input_shape,))
x = Embedding(vocabulary, embedding_dim)(i)
x = LSTM(64, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i, x)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=50)

# Save model
model.save('chatbot/chatbot_model.h5')

# Save tokenizer
with open('chatbot/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save label encoder
with open('chatbot/label_encoder.pickle', 'wb') as handle:
    pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save responses
with open('chatbot/responses.json', 'w') as file:
    json.dump(responses, file)
