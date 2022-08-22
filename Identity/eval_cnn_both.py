#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import numpy as np
import random as rd
import json
from tqdm import tqdm


import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from copy import copy
from statistics import mean
from utils import elapsed_timer, get_accuracy_classification
from data import Task1Dataset, enrich, generate_negative, generate_pos_only
from analogy_classif_both import Classification
import cnn_both
import sys
sys.path.append('tools')
import parse



args = parse.args
args.embed_size = 200
args.workers = 8
args.batch_size = 64
if torch.cuda.is_available():
    args.gpu = 1
else:
    args.gpu = 0

args.use_unstructure = 1
args.n_ehr = len(json.load(open(os.path.join('processed_id/files/demo_index_dict.json'), 'r'))) + 10

def evaluate_classifier(filename, nb_analogies, rd_seed):
    '''Produces the accuracy for valid analogies, invalid analogies for a given model.
    Arguments:
    language -- The language of the model.
    nb_analogies -- The maximum number of analogies (before augmentation) we evaluate the model on. If the number is greater than the number of analogies in the dataset, then all the analogies will be used.
    epochs -- The number of epochs the models were trained on (we use this parameter to use the right files).'''

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rd.seed(rd_seed)

    # --- Prepare data ---
    train_dataset = Task1Dataset(filename)
   
    train_analogies, test_analogies = train_test_split(train_dataset.analogies, test_size=0.3, random_state = 42)
    test_dataset = copy(train_dataset)

    test_dataset.analogies = test_analogies
    train_dataset.analogies = train_analogies

    if len(test_dataset) > nb_analogies:
        test_indices = list(range(len(test_dataset)))
        test_sub_indices = rd.sample(test_indices, nb_analogies)
        test_subset = Subset(test_dataset, test_sub_indices)
    else:
        test_subset = test_dataset
    

    test_dataloader = DataLoader(test_subset, batch_size=1, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    #add the path to the model
    path_models = f"mimic_id/classif_cnn/classification_1_both_40e.pth"
    saved_models = torch.load(path_models)


    embedding_model = cnn_both.CNN(args)
    embedding_model.load_state_dict(saved_models['state_dict_embeddings'])
    embedding_model.eval()

    classification_model = Classification(args.embed_size) 
    classification_model.load_state_dict(saved_models['state_dict_classification'])
    classification_model.eval()

    embedding_model.to(device)
    classification_model.to(device)

    accuracy_true = []
    accuracy_false = []

    for a, b, c, d in test_dataloader:        
        analogy_element_a, dd_a, content_a, patient_a = a
        analogy_element_b, dd_b, content_b = b
        analogy_element_c, dd_c, content_c, patient_c = c
        analogy_element_d, dd_d, content_d = d
        #print(a)

        # compute the embeddings
        a = embedding_model(dd_a.to(device), content_a.to(device)) 
        b = embedding_model(dd_b.to(device), content_b.to(device))
        c = embedding_model(dd_c.to(device), content_c.to(device))
        d = embedding_model(dd_d.to(device), content_d.to(device))

        data = torch.stack([a, b, c, d], dim = 1)

        if patient_a != patient_c:
            for a, b, c, d in enrich(data):

                # positive example, target is 1
                a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
                b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
                c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
                d = torch.unsqueeze(torch.unsqueeze(d, 0), 0)

                is_analogy = torch.squeeze(classification_model(a, b, c, d))

                expected = torch.ones(is_analogy.size(), device=is_analogy.device)

                accuracy_true.append(get_accuracy_classification(expected, is_analogy))


            for a, b, c, d in generate_negative(data):

                # negative examples, target is 0
                a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
                b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
                c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
                d = torch.unsqueeze(torch.unsqueeze(d, 0), 0)

                is_analogy = torch.squeeze(classification_model(a, b, c, d))

                expected = torch.zeros(is_analogy.size(), device=is_analogy.device)

                accuracy_false.append(get_accuracy_classification(expected, is_analogy))
        else:

            for a, b, c, d in generate_pos_only(data):

                # positive example, target is 1
                a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
                b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
                c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
                d = torch.unsqueeze(torch.unsqueeze(d, 0), 0)
                
                is_analogy = torch.squeeze(classification_model(a, b, c, d))

                expected = torch.ones(is_analogy.size(), device=is_analogy.device)

                accuracy_true.append(get_accuracy_classification(expected, is_analogy))

            for a, b, c, d in generate_negative(data):

                # positive examples, target is 1
                a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
                b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
                c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
                d = torch.unsqueeze(torch.unsqueeze(d, 0), 0)

                is_analogy = torch.squeeze(classification_model(a, b, c, d))

                expected = torch.ones(is_analogy.size(), device=is_analogy.device)

                accuracy_true.append(get_accuracy_classification(expected, is_analogy))


    print(f'Accuracy for valid analogies: {mean(accuracy_true)}\nAccuracy for invalid analogies: {mean(accuracy_false)}')

if __name__ == '__main__':
    evaluate_classifier(filename = "T1_IDENTITY_200.txt", nb_analogies = 50000, rd_seed = 1)

