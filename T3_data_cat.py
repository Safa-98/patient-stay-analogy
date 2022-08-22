#!/usr/bin/env python
# coding: utf-8

# In[3]:


import itertools
import pandas as pd
from  itertools import chain
import numpy as np
import torch, torch.nn as nn
import random

import json
import os



def load_data(filename):
    '''Load the data from the  files in the form of a list of triples (STAY A, STAY B, PATIENT_ID).'''
    with open(filename, "r", encoding="utf-8") as f:
        #print(f.readlines())
        #print([line.split()for line in f])
        return ([line.strip().split(',') for line in f ])


def enrich(data):
    """Apply the example generation process from 'Solving Word Analogies: a Machine Learning Perspective'to generate valid analogies."""
    for a, b, c, d in data:
        yield a, b, c, d
        yield c, d, a, b
        yield a, b, a, b


def generate_negative(positive_data):
    """Apply the negative example generation process from 'Solving Word Analogies: a Machine Learning Perspective'."""
    for a, b, c, d in positive_data:
        yield d, a, b, c
        yield a, c, b, d
        yield b, a, d, c
        yield b, a, c, d
        yield a, b, d, c
        yield c, d, b, a
        yield a, a, b, b
        yield d, c, a, b 



class Task1Dataset(torch.utils.data.Dataset):

    def __init__(self, filename = "T3_ID+DSEQ_CAT.txt"): #T2_ID+SEQ_ICD_1800.txt #T3_ID+DSEQ_ICD_1800.txt
        super(Task1Dataset).__init__()
        #self.mode = mode
        self.raw_data = load_data(filename = filename)
        #self.prepare_data()
        self.set_analogy_classes()
        self.demo_dict = json.load(open(os.path.join('processed_cat_T3/files/demo_dict.json'), 'r')) #processed_icd_SEQ
        self.vector_dict = json.load(open('processed_cat_T3/files/vector_dict.json', 'r'))

    # def prepare_data(self):
    #     """Generate embeddings for the 4 elements."""
    #     voc = set()
    #     for stay_a, stay_b, patient_a in self.raw_data:
    #         voc.update(stay_a)
    #         voc.update(stay_b)
    #     self.word_voc = list(voc)
    #     self.word_voc.sort()
    #     self.word_voc_id = {character: i for i, character in enumerate(self.word_voc)}

    def set_analogy_classes(self):
        self.analogies = []
        self.all_words = set()
        for i, (stay_a_i, stay_b_i, patient_a_i, icd_code_a_i) in enumerate(self.raw_data):
            self.all_words.add(stay_a_i)
            self.all_words.add(stay_b_i)
            for j, (stay_a_j, stay_b_j, patient_a_j, icd_code_a_j) in enumerate(self.raw_data[i:]):
                if icd_code_a_i == icd_code_a_j:
                    self.analogies.append((i, i+j))

    def __len__(self):
        return len(self.analogies)

    def __getitem__(self, index):
        ab_index, cd_index = self.analogies[index]
        a, b, patient_a, icd_code_a = self.raw_data[ab_index]
        c, d, patient_c, icd_code_c = self.raw_data[cd_index]

        #obtain demographics tensors
        demo_tensor_a = torch.from_numpy(np.array(self.demo_dict.get(a, 0), dtype=np.int64))
        demo_tensor_b = torch.from_numpy(np.array(self.demo_dict.get(b, 0), dtype=np.int64))
        demo_tensor_c = torch.from_numpy(np.array(self.demo_dict.get(c, 0), dtype=np.int64))
        demo_tensor_d = torch.from_numpy(np.array(self.demo_dict.get(d, 0), dtype=np.int64))

        #padding doc2vec
        content = self.vector_dict[a]
        while len(content) < 16:
            content.append([0] * 200)
        content = content[:16]
        content_tensor_a = torch.from_numpy(np.array(content, dtype=np.float32))
        
        content = self.vector_dict[b]
        while len(content) < 16:
            content.append([0] * 200)
        content = content[:16]
        content_tensor_b = torch.from_numpy(np.array(content, dtype=np.float32))

        content = self.vector_dict[c]
        while len(content) < 16:
            content.append([0] * 200)
        content = content[:12]
        content_tensor_c = torch.from_numpy(np.array(content, dtype=np.float32))
        
        content = self.vector_dict[d]
        while len(content) < 16:
            content.append([0] * 200)
        content = content[:16]
        content_tensor_d = torch.from_numpy(np.array(content, dtype=np.float32))
        
        a = (a, demo_tensor_a, content_tensor_a, patient_a, icd_code_a)
        b = (b, demo_tensor_b, content_tensor_b)
        c = (c, demo_tensor_c, content_tensor_c, patient_c, icd_code_c)
        d = (d, demo_tensor_d, content_tensor_d)
        return (a, b, c, d) 
    

    

if __name__ == "__main__":
    print(len(Task1Dataset().analogies))
    #print(len(Task1Dataset().all_words))
    #print(Task1Dataset().all_words)
    print(Task1Dataset()[20])




