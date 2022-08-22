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
from utils import elapsed_timer, get_accuracy_classification, tpr_tnr_balacc_harmacc_f1
from T3_data_blk import Task1Dataset, enrich, generate_negative
from analogy_classif import Classification
import cnn_both
import sys
sys.path.append('tools')
import parse, py_op



args = parse.args
args.embed_size = 200
args.workers = 8
args.batch_size = 64
if torch.cuda.is_available():
    args.gpu = 1
else:
    args.gpu = 0

args.use_unstructure = 1
args.n_ehr = len(json.load(open(os.path.join('processed_blk_T3/files/demo_index_dict.json'), 'r'))) + 10

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

    path_models = f"mimic_blk_T3/classif_cnn/classification_1_both_40e.pth"
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
    
    pos, neg = [], []

    for a, b, c, d in test_dataloader:        
        analogy_element_a, dd_a, content_a, patient_a, icd_code_a = a
        analogy_element_b, dd_b, content_b = b
        analogy_element_c, dd_c, content_c, patient_c, icd_code_c = c
        analogy_element_d, dd_d, content_d = d
        print("a", analogy_element_a, patient_a, icd_code_a, "b", analogy_element_b, "c", analogy_element_c, patient_c, icd_code_c, "d", analogy_element_d)

        # compute the embeddings
        a = embedding_model(dd_a.to(device), content_a.to(device)) 
        b = embedding_model(dd_b.to(device), content_b.to(device))
        c = embedding_model(dd_c.to(device), content_c.to(device))
        d = embedding_model(dd_d.to(device), content_d.to(device))

        data = torch.stack([a, b, c, d], dim = 1)

        i = 0
        for a, b, c, d in enrich(data):
            # positive example, target is 1
            a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
            b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
            c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
            d = torch.unsqueeze(torch.unsqueeze(d, 0), 0)
            is_analogy = torch.squeeze(classification_model(a, b, c, d))
            print("predicted value", is_analogy)

            expected = torch.ones(is_analogy.size(), device=is_analogy.device)
            print("expected value", expected)

            accuracy_true.append(get_accuracy_classification(expected, is_analogy))
            print("valid analogy form", i)
            
            pos.append(is_analogy >= 0.5)

        j = 0
        for a, b, c, d in generate_negative(data):

            # negative examples, target is 0
            a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
            b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
            c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
            d = torch.unsqueeze(torch.unsqueeze(d, 0), 0)
            is_analogy = torch.squeeze(classification_model(a, b, c, d))
            print("predicted value", is_analogy)

            expected = torch.zeros(is_analogy.size(), device=is_analogy.device)
            print("expected value", expected)

            accuracy_false.append(get_accuracy_classification(expected, is_analogy))
            print("invalid analogy form", j)

            neg.append(is_analogy < 0.5)

    pos = torch.stack(pos).float()#.view(-1).float()
    neg = torch.stack(neg).float()#.view(-1).float()


    # actual data to comute stats at the end
    tp, tn, fn, fp = pos.sum(), neg.sum(), (1-pos).sum(), (1-neg).sum()


    tpr, tnr, balacc, harmacc, f1 = tpr_tnr_balacc_harmacc_f1(tp, tn, fp, fn)

    print(f'Accuracy for valid analogies: {mean(accuracy_true)}\nAccuracy for invalid analogies: {mean(accuracy_false)}')
    torch.save({"valid": mean(accuracy_true), "invalid": mean(accuracy_false)}, f"mimic_blk_T3/eval_cnn/evaluation_correct_{rd_seed}_both_40e.pth")
    print(f"Scores: tp {tp} tp, tn {tn}, fn {fn}, fp {fp}, f1{f1}")

if __name__ == '__main__':
    evaluate_classifier(filename = "T3_ID+DSEQ_BLK.txt", nb_analogies = 50000, rd_seed = 1)

