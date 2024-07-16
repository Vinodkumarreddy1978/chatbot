# chatbot/views.py

from django.shortcuts import render
from django.http import JsonResponse
import json, pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and necessary files
model = tf.keras.models.load_model('chatbot/chatbot_model.h5')

with open('chatbot/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('chatbot/label_encoder.pickle', 'rb') as handle:
    le = pickle.load(handle)

with open('chatbot/responses.json', 'r') as file:
    responses = json.load(file)

def predict(request):
    if request.method == 'POST':
        user_input = json.loads(request.body).get('message')
        if not user_input:
            return JsonResponse({"error": "No input provided"}, status=400)

        # Preprocess user input
        seq = tokenizer.texts_to_sequences([user_input])
        padded_seq = pad_sequences(seq, maxlen=model.input_shape[1])
        
        # Predict intent
        pred = model.predict(padded_seq)
        intent = le.inverse_transform([np.argmax(pred)])

        # Get response
        response = np.random.choice(responses[intent[0]])
        return JsonResponse({"response": response}, status=200)

    return JsonResponse({"error": "Invalid request method"}, status=405)

def chat(request):
    return render(request, 'chatbot/chat.html')
