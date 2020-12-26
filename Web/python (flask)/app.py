# -*- coding: utf-8 -*- 
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

import torch

from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer

import gluonnlp as nlp

from model.encoder import *
from model.classifier import *
from model.bertsum import *

from utils import *
from utils.data import *

def tokenizing(tokenizer, article, max_sentence_num=128, max_word_num=128, pad=True, pair=False):

    transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=max_word_num, pad=pad, pair=pair)

    numbers_tokenized = np.array([[]])

    for sentence in article:
        # tokenizing
        tokenized_to_num = transform([sentence])
        tokenized_to_num = np.expand_dims(tokenized_to_num[0], axis=0)
        if numbers_tokenized.shape == (1, 0):
            numbers_tokenized = np.concatenate((tokenized_to_num, ), axis = 0)
        else:
            numbers_tokenized = np.concatenate((numbers_tokenized, tokenized_to_num), axis = 0)

    # padding
    if len(numbers_tokenized) < max_sentence_num:
        remaining_adds = max_sentence_num - len(numbers_tokenized)
        padding_sentence = np.ones((remaining_adds, max_word_num),dtype=np.int32)
        numbers_tokenized = np.concatenate((numbers_tokenized, padding_sentence),axis=0)
    else:
        numbers_tokenized = numbers_tokenized[:max_sentence_num]
    
    numbers_tokenized = numbers_tokenized.astype(np.int32)

    return numbers_tokenized

device = 'cpu'

# bert call
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

app = Flask(__name__)

@app.route('/')
def index():
    return "Flask server"

@app.route('/summary', methods=['POST'])

def recommend():
    args = request.get_json(force=True)
    user_input = args.get('user_input', [])

    max_sentence_num = 64
    max_word_num = 64
    num_workers = 5
    batch_size = 64
    learning_rate = 1e-5

    config = Config({
        "n_enc_vocab": len(vocab),
        "n_dec_vocab": len(vocab),
        "n_enc_seq":128,
        "n_layer":6,
        "d_hidn":128,
        "i_pad":1,
        "d_ff":1024,
        "n_head":4,
        "d_head":64,
        "dropout":0.1,
        "layer_norm_epsilon":1e-12
    })

    # model
    encoder = BertEncoder(bertmodel)
    reducer = DimensionReducer(768, 128)
    second = SecondEncoder(config=config, n_layer=1)
    classifier = BinaryClassifier(128)
    model = BERTSummarizer(config, reducer, second, classifier, device)

    # load model params
    save_path = './checkpoints/checkpoint_1 final.pt'
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    article = user_input.split('. ')
    tokenized = tokenizing(tok, article, max_sentence_num=128, max_word_num=128, pad=True, pair=False)
    tokenized = torch.from_numpy(tokenized).unsqueeze(0)

    new_batch = embedding(encoder, tokenized, torch.tensor([len(article)]))
    output = model(new_batch, torch.tensor([len(article)])).squeeze()[:len(article)]

    sum_ext = []
    for k in range(3):
        max_v = max(output)
        for idx in range(len(output)):
            if output[idx] == max_v:
                output[idx] = -1
                sum_ext.append(idx)
                break

    summary = [article[i] for i in sorted(sum_ext) if i < len(article)]
    return jsonify(
        summary = summary,
        userInput = user_input
    )

if __name__ == "__main__":
    app.run(host='localhost', port=5000)