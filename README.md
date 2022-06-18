# An analogy based framework for patient-stay identification in healthcare
This repository contains source code for paper _An analogy based framework for patient-stay identification in healthcare_ submitted to the ICCBR-ATA 2022 Workshop.

## Requirements

### Dataset
MIMIC-III database analyzed in the study is available on PhysioNet repository. Here are some steps to prepare for the dataset:

To request access to MIMIC-III, please follow https://mimic.physionet.org/gettingstarted/access/. Make sure to place .csv files under data/mimic.
With access to MIMIC-III, to build the MIMIC-III dataset locally using Postgres, follow the instructions at https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic/postgres.
Run SQL queries to generate necessary views, please follow https://github.com/onlyzdd/clinical-fusion/tree/master/query.


### Installing the Dependencies
Install Anaconda (or miniconda to save storage space).

Then, create a conda environement (for example stay-analogy) and install the dependencies, using the following commands:

```bash
$ conda create --name stay-analogy python=3.9
$ conda activate stay-analogy
$ conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -c=conda-forge
$ conda install -y numpy scipy pandas scikit-learn
$ conda install -y tqdm gensim nltk
```

### Usage

```bash
$ python3 dataset.py # builds the dataset of triples for the _Identity setting_
$ python3 preprocess01.py # define patient cohort, collect labels, extract temporal signals, and extract clinical notes
$ python3 preprocess02.py # run full preprocessing to obtain dictionaries
$ python3 doc2vec.py --phase train # train doc2vec model
$ python3 doc2vec.py --phase infer # infer doc2vec vectors
```

To train and evaluate the classification and the corresponding embedding model on structured and unstructured data
```bash
$ python3 train_cnn_both.py # train classifier model together with the embedding model 
$ python3 evaluate_cnn_both.py # evaluate a classifier with the corresponding embedding model
```

To train and evaluate the classification and the corresponding embedding model on only unstructured data
```bash
$ python3 train_cnn_text.py # train classifier model together with the embedding model 
$ python3 evaluate_cnn_text.py # evaluate a classifier with the corresponding embedding model
```

## Files and Folders

- `data.py`: tools to load the dataset, contains the main dataset class `Task1Dataset` and the data augmentation functions
- `analogy_classif_both.py`: neural network to classify analogies for structured and unstructured data
- `analogy_classif_con.py`: neural network to classify analogies for unstructured data
- `cnn_con.py`: neural network to embed clinical notes
- `cnn_both.py`: neural network to embed static information and clinical notes
- `train_cnn_both.py`: file to train the classifier model together with the embedding model for structured and unstructured data
- `evaluate_cnn_both.py`: file to evaluate a classifier with the corresponding embedding model and type of data
- `train_cnn_con.py`: file to train the classifier model together with the embedding model for unstructured data
- `evaluate_cnn_con.py`: file to evaluate a classifier with the corresponding embedding model and type of data
- `utils.py`: tools for the different codes
