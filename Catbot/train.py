import json
import numpy as np
from preprocessing import tokenize, stem_indo, bag_of_word


with open('intents.json', 'r') as f:
    intents = json.load(f)

all_word = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent.tag
    tags.append(tag)
    for pattern in intent['patterns']:
        word = tokenize(pattern)
        all_word.extend(word)
        xy.append((word, tag))