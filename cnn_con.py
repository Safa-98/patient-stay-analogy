#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *

import numpy as np

import sys
sys.path.append('tools')
import parse, py_op


def conv3(in_channels, out_channels, stride=1, kernel_size=3):
    return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=stride, padding=1, bias=False)

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        self.dd_embedding = nn.Embedding (args.n_ehr, args.embed_size ) #demographics
        self.dd_mapping = nn.Sequential(
                nn.Linear ( args.embed_size, args.embed_size),
                nn.ReLU ( ),
                nn.Dropout(0.1),
                nn.Linear ( args.embed_size, args.embed_size),
                nn.ReLU ( ),
                nn.Dropout(0.1),
                )    
        self.pooling = nn.AdaptiveMaxPool1d(1)


        # unstructureL clinical notes
        if args.use_unstructure:
            self.vocab_layer = nn.Sequential(
                    nn.Dropout(0.2),
                    conv3(args.embed_size, args.embed_size, 2, 2),
                    nn.BatchNorm1d(args.embed_size),
                    nn.Dropout(0.2),
                    nn.ReLU(), 
                    )


    def forward(self, content, dd = None):  
        if dd is not None and content is not None:
            # demo embedding
            dsize = list(dd.size()) + [-1]
            d = self.dd_embedding(dd.view(-1)).view(dsize)
            d = self.dd_mapping(d)
            d = torch.transpose(d, 1,2).contiguous()                
            d = self.pooling(d)
            d = d.view((d.size(0), -1))
            #print("dd", d.shape)

            content = self.vocab_layer(content.transpose(1,2))
            content = self.pooling(content) # (64, 200, 1)
            content = content.view((content.size(0), -1))
            output = torch.cat((d, content), 1)
            #print("content",  output.shape)
            return output

        else:
            content = self.vocab_layer(content.transpose(1,2))
            content = self.pooling(content)
            content = content.view((content.size(0), -1)) #(64, 200)
            return content





