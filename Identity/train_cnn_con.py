#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import random as rd
from functools import partial
from statistics import mean
from sklearn.model_selection import train_test_split
from copy import copy
from utils import elapsed_timer
from data import Task1Dataset, enrich, generate_negative, generate_pos_only
from analogy_classif_con import Classification
import cnn_con
import sys
import json
import os
from tqdm import tqdm
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

def train_classifier(filename, nb_analogies, epochs, rd_seed):
    '''Trains a classifier and a word embedding model for a given language.
    Arguments:
    nb_analogies -- The number of analogies to use (before augmentation) for the training.
    epochs -- The number of epochs we train the model for.
    rd_seed -- The seed for the random module.'''

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rd.seed(rd_seed)

    # --- Prepare data ---

    ## Train and test dataset
    
    train_dataset = Task1Dataset(filename = filename)
   
    train_analogies, test_analogies = train_test_split(train_dataset.analogies, test_size=0.3, random_state = 42)

    train_dataset.analogies = train_analogies


    # Get subsets

    if len(train_dataset) > nb_analogies:
        train_indices = list(range(len(train_dataset)))
        train_sub_indices = rd.sample(train_indices, nb_analogies)
        train_subset = Subset(train_dataset, train_sub_indices)
    else:
        train_subset = train_dataset



    # Load data
    train_dataloader = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=args.workers, pin_memory=True)


    # --- Training models ---


    classification_model = Classification(args.embed_size) # 16 because 16 filters of each size, 5 because 5 sizes
    embedding_model = cnn_con.CNN(args)


    # --- Training Loop ---
    classification_model.to(device)
    embedding_model.to(device)

    optimizer = torch.optim.Adam(list(classification_model.parameters()) + list(embedding_model.parameters()))
    criterion = nn.BCELoss()

    losses_list = []
    times_list = []
    
    for epoch in range(epochs):

        losses = []
        with elapsed_timer() as elapsed:
            for a, b, c, d in train_dataloader:

                optimizer.zero_grad()
                analogy_element_a, dd_a, content_a, patient_a = a
                analogy_element_b, dd_b, content_b = b
                analogy_element_c, dd_c, content_c, patient_c = c
                analogy_element_d, dd_d, content_d = d
                #print("a", analogy_element_a, patient_a, "b", analogy_element_b, "c", analogy_element_c, patient_c, "d", analogy_element_d)

                # compute the embeddings
                a = embedding_model(content_a.to(device)) 
                b = embedding_model(content_b.to(device))
                c = embedding_model(content_c.to(device))
                d = embedding_model(content_d.to(device))

                # to be able to add other losses, which are tensors, we initialize the loss as a 0 tensor
                loss = torch.tensor(0).to(device).float()

                data = torch.stack([a, b, c, d], dim = 1)
       

                if patient_a != patient_c:
                    for a, b, c, d in enrich(data):
                        # positive example, target is 1
                        a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
                        b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
                        c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
                        d = torch.unsqueeze(torch.unsqueeze(d, 0), 0)

                        is_analogy = classification_model(a, b, c, d)
                        expected = torch.ones(is_analogy.size(), device=is_analogy.device)

                        loss += criterion(is_analogy, expected)
 
                    for a, b, c, d in generate_negative(data):
                        # negative examples, target is 0
                        a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
                        b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
                        c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
                        d = torch.unsqueeze(torch.unsqueeze(d, 0), 0)

                        is_analogy = classification_model(a, b, c, d)

                        expected = torch.zeros(is_analogy.size(), device=is_analogy.device)

                        loss += criterion(is_analogy, expected)

                    loss.backward()
                    optimizer.step()
                    losses.append(loss.cpu().item())
    
                else:
                    for a, b, c, d in generate_pos_only(data):

                        # positive example, target is 1
                        a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
                        b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
                        c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
                        d = torch.unsqueeze(torch.unsqueeze(d, 0), 0)

                        is_analogy = classification_model(a, b, c, d)

                        expected = torch.ones(is_analogy.size(), device=is_analogy.device)

                        loss += criterion(is_analogy, expected)

                    for a, b, c, d in generate_negative(data):

                        # positive examples, target is 1
                        a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
                        b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
                        c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
                        d = torch.unsqueeze(torch.unsqueeze(d, 0), 0)

                        is_analogy = classification_model(a, b, c, d)

                        expected = torch.ones(is_analogy.size(), device=is_analogy.device)

                        loss += criterion(is_analogy, expected)

                    # once we have all the losses for one set of embeddings, we can backpropagate
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.cpu().item())

        losses_list.append(mean(losses))
        times_list.append(elapsed())
        print(f"Epoch: {epoch}, Run time: {times_list[-1]:4.5}s, Loss: {losses_list[-1]}")

    torch.save({"state_dict_classification": classification_model.cpu().state_dict(), "state_dict_embeddings": embedding_model.cpu().state_dict(), "losses": losses_list, "times": times_list}, f"mimic_id/classif_cnn/classification_{rd_seed}_con_{epochs}e.pth")

if __name__ == '__main__':
    train_classifier(filename = "T1_IDENTITY_200.txt", nb_analogies = 50000, epochs = 20, rd_seed = 1)

