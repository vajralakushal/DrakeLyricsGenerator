from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask
from flask import request
from flask_cors import CORS

import numpy as np
import random
import io

app = Flask(__name__)
CORS(app)

model_1 = keras.models.load_model('/Users/ayaanmahimwala/Coding/DrakeLyricsGenerator/DrakeGenerator/model1')

path = '/Users/ayaanmahimwala/Coding/DrakeLyricsGenerator/DrakeGenerator/drake_dataset/drake_lyrics.txt'
with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")  # We remove newlines chars for nicer display

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
diversity = 0.5

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def predict(seed):
    generated = ""
    seed = seed.lower()
    maxlen = len(seed)
    for i in range(400):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(seed):
            x_pred[0, t, char_indices[char]] = 1.0
        preds = model_1.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        seed = seed[1:] + next_char
        generated += next_char
    return generated
    


@app.route("/", methods=['POST'])
def helloWorld():
    req_data = request.get_json()
    seed = req_data['seed']
    predicted = predict(seed)
    return {
        "response" : predicted
    }
        