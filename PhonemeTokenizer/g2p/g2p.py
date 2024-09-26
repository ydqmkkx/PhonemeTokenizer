# -*- coding: utf-8 -*-
# /usr/bin/python
'''
By kyubyong park(kbpark.linguist@gmail.com) and Jongseok Kim(https://github.com/ozmig77)
https://www.github.com/kyubyong/g2p


yd log:
1. Changed for punctuation: retain only the first of consecutive punctuations
2. Changed for 'the': two pronunciations

'''
from nltk import pos_tag
from nltk.corpus import cmudict
import nltk
from nltk.tokenize import TweetTokenizer
word_tokenize = TweetTokenizer().tokenize
import numpy as np
import codecs
import re
import os
import unicodedata
from builtins import str as unicode
from .expand import normalize_numbers, expand_abbr

import string
_punctuation = list(string.punctuation)+['..']+['...']

dirname = os.path.dirname(__file__)
nltk.data.load(os.path.join(dirname,'averaged_perceptron_tagger'), format='pickle')
nltk.data.load(os.path.join(dirname,'cmudict'))

def construct_homograph_dictionary():
    f = os.path.join(dirname,'homographs.en')
    homograph2features = dict()
    for line in codecs.open(f, 'r', 'utf8').read().splitlines():
        if line.startswith("#"): continue
        headword, pron1, pron2, pos1 = line.strip().split("|")
        homograph2features[headword.lower()] = (pron1.split(), pron2.split(), pos1)
    return homograph2features


class G2p(object):
    def __init__(self):
        super().__init__()
        self.graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
        self.phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                                             'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
                                                             'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                                             'EY2', 'F', 'G', 'HH',
                                                             'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
                                                             'M', 'N', 'NG', 'OW0', 'OW1',
                                                             'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
                                                             'UH0', 'UH1', 'UH2', 'UW',
                                                             'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self.idx2g = {idx: g for idx, g in enumerate(self.graphemes)}

        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}

        self.cmu = cmudict.dict()
        self.load_variables()
        self.homograph2features = construct_homograph_dictionary()

    def load_variables(self):
        self.variables = np.load(os.path.join(dirname,'checkpoint20.npz'))
        self.enc_emb = self.variables["enc_emb"]  # (29, 64). (len(graphemes), emb)
        self.enc_w_ih = self.variables["enc_w_ih"]  # (3*128, 64)
        self.enc_w_hh = self.variables["enc_w_hh"]  # (3*128, 128)
        self.enc_b_ih = self.variables["enc_b_ih"]  # (3*128,)
        self.enc_b_hh = self.variables["enc_b_hh"]  # (3*128,)

        self.dec_emb = self.variables["dec_emb"]  # (74, 64). (len(phonemes), emb)
        self.dec_w_ih = self.variables["dec_w_ih"]  # (3*128, 64)
        self.dec_w_hh = self.variables["dec_w_hh"]  # (3*128, 128)
        self.dec_b_ih = self.variables["dec_b_ih"]  # (3*128,)
        self.dec_b_hh = self.variables["dec_b_hh"]  # (3*128,)
        self.fc_w = self.variables["fc_w"]  # (74, 128)
        self.fc_b = self.variables["fc_b"]  # (74,)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def grucell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        rzn_ih = np.matmul(x, w_ih.T) + b_ih
        rzn_hh = np.matmul(h, w_hh.T) + b_hh

        rz_ih, n_ih = rzn_ih[:, :rzn_ih.shape[-1] * 2 // 3], rzn_ih[:, rzn_ih.shape[-1] * 2 // 3:]
        rz_hh, n_hh = rzn_hh[:, :rzn_hh.shape[-1] * 2 // 3], rzn_hh[:, rzn_hh.shape[-1] * 2 // 3:]

        rz = self.sigmoid(rz_ih + rz_hh)
        r, z = np.split(rz, 2, -1)

        n = np.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h

        return h

    def gru(self, x, steps, w_ih, w_hh, b_ih, b_hh, h0=None):
        if h0 is None:
            h0 = np.zeros((x.shape[0], w_hh.shape[1]), np.float32)
        h = h0  # initial hidden state
        outputs = np.zeros((x.shape[0], steps, w_hh.shape[1]), np.float32)
        for t in range(steps):
            h = self.grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)  # (b, h)
            outputs[:, t, ::] = h
        return outputs

    def encode(self, word):
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        x = np.take(self.enc_emb, np.expand_dims(x, 0), axis=0)

        return x

    def predict(self, word):
        # encoder
        enc = self.encode(word)
        enc = self.gru(enc, len(word) + 1, self.enc_w_ih, self.enc_w_hh,
                       self.enc_b_ih, self.enc_b_hh, h0=np.zeros((1, self.enc_w_hh.shape[-1]), np.float32))
        last_hidden = enc[:, -1, :]

        # decoder
        dec = np.take(self.dec_emb, [2], axis=0)  # 2: <s>
        h = last_hidden

        preds = []
        for i in range(20):
            h = self.grucell(dec, h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh)  # (b, h)
            logits = np.matmul(h, self.fc_w.T) + self.fc_b
            pred = logits.argmax()
            if pred == 3: break  # 3: </s>
            preds.append(pred)
            dec = np.take(self.dec_emb, [pred], axis=0)

        preds = [self.idx2p.get(idx, "<unk>") for idx in preds]
        return preds

    def __call__(self, text, stress=False, prefix=True):
        # preprocessing
        text = unicode(text)
        text = re.sub(r'\d{17,}', lambda m: ' '.join(list(m.group())), text) #When the digits exceeds 17, adding spaces between them
        text = normalize_numbers(text)
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        text = text.lower()
        #text = re.sub("[^ a-z'.,?!\-]", "", text)
        #text = re.sub("[^ a-z'`.,?!:;()\[\]{}<>\-\"#]", "", text)
        text = expand_abbr(text)

        # tokenization
        words = word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)

        # steps
        prons = []
        words = []
        the_flag = False
        punc_flag = False
        for word, pos in tokens:
            if word in _punctuation and punc_flag:
                continue
            punc_flag = word in _punctuation

            if re.search("[a-z]", word) is None:
                pron = [word]
            elif word == 'the':
                pron = ['DH', 'AH0']
                the_flag = True
            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmu:  # lookup CMU dict
                pron = self.cmu[word][0]
            else: # predict for oov
                pron = self.predict(word)
            
            if the_flag and word!='the':
                the_flag = False
                if pron[0][0] in ['A', 'E', 'I', 'O', 'U']:
                    prons[-2] = 'IY0'

            prons.extend(pron)
            prons.extend([" "])
            words.append(word)

        prons = prons[:-1]
        prons = "|".join(prons)
        if not stress:
            prons = re.sub(r'\d+', '', prons)
        if prefix:
            prons = re.sub(r'(?<=\S)\|(?=\S)', '|##', prons)

        prons = prons.replace('|', ' ').split("   ")
        return words, prons