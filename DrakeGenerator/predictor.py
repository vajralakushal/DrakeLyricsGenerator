from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import random
import io

model_1 = keras.models.load_model('/Users/kushalvajrala/ProgrammingProjects/DrakeLyricsGenerator/DrakeGenerator/model1')

path = '/Users/kushalvajrala/ProgrammingProjects/DrakeLyricsGenerator/DrakeGenerator/drake_dataset/drake_lyrics.txt'
with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")  # We remove newlines chars for nicer display

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print('Enter your favorite Drake line:')
sentence = input()#text[start_index : start_index + maxlen]
sentence = sentence.lower()
maxlen = len(sentence)

diversity = 0.5
start_index = random.randint(0, len(text) - maxlen - 1)
generated = ""

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


print('...Generating with seed: "' + sentence + '"')
for i in range(400):
  x_pred = np.zeros((1, maxlen, len(chars)))
  for t, char in enumerate(sentence):
      x_pred[0, t, char_indices[char]] = 1.0
  preds = model_1.predict(x_pred, verbose=0)[0]
  next_index = sample(preds, diversity)
  next_char = indices_char[next_index]
  sentence = sentence[1:] + next_char
  generated += next_char

def predict(seed):
    for i in range(400):
      x_pred = np.zeros((1, maxlen, len(chars)))
      for t, char in enumerate(sentence):
          x_pred[0, t, char_indices[char]] = 1.0
      preds = model_1.predict(x_pred, verbose=0)[0]
      next_index = sample(preds, diversity)
      next_char = indices_char[next_index]
      sentence = sentence[1:] + next_char
      generated += next_char
    
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def helloWorld(seed):
  return {
          "response": predict(seed)
          }
       