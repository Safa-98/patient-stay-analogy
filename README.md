# Exploring Analogy based Applications in Healthcare
This repository contains source code for paper _An analogy based framework for patient-stay identification in healthcare_ submitted to the ICCBR-ATA 2022 Workshop and all the work carried out in the M2 thesis titled _Exploring Analogy based Applications in Healthcare_. In this thesis, we focus on two tasks: (1) **patient-stay identification**, _i.e._, does a hospital stay belong to a patient or not?, using our first setting, and (2) **disease prognosis**,  _i.e._, will a certain disease develop in the same way in two distinct patients?, using our second and third settings. We propose a prototypical architecture that combines patient-stay representation learning and the analogical reasoning framework. We train a neural model to detect patient-stay analogies. Our models are implemented using PyTorch.

## Requirements

### Dataset
MIMIC-III database analyzed in the study is available on PhysioNet repository. Here are some steps to prepare for the dataset:

* To request access to MIMIC-III, please follow https://mimic.physionet.org/gettingstarted/access/. Make sure to place ```.csv files``` under ``` data/mimic```.
* With access to MIMIC-III, to build the MIMIC-III dataset locally using Postgres, follow the instructions at https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic/postgres.
* Run SQL queries to generate necessary views, please follow https://github.com/onlyzdd/clinical-fusion/tree/master/query.



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

**For the _Identity setting_ (T1), run the following codes:**
```bash
	$ python3 T1_dataset.py # build the dataset of triples for the _Identity setting_
	$ python3 preprocess01.py # define patient cohort, collect labels, extract temporal signals, and extract clinical notes
	$ python3 preprocess02.py # run full preprocessing to obtain dictionaries
	$ python3 doc2vec.py --phase train # train doc2vec model
	$ python3 doc2vec.py --phase infer # infer doc2vec vectors
```

To train and evaluate the classification and the corresponding embedding model on **structured and unstructured** data, run:
```bash
	$ python3 Identity/train_cnn_both.py # train classifier model together with the embedding model 
	$ python3 Identity/evaluate_cnn_both.py # evaluate a classifier with the corresponding embedding model
```

To train and evaluate the classification and the corresponding embedding model on only **unstructured** data, run:
```bash
	$ python3 Identity/train_cnn_con.py # train classifier model together with the embedding model 
	$ python3 Identity/evaluate_cnn_con.py # evaluate a classifier with the corresponding embedding model
```
To train and evaluate the classification and the corresponding embedding model on only **structured** data
```bash
	$ python3 Identity/train_cnn_demo.py # train classifier model together with the embedding model 
	$ python3 Identity/evaluate_cnn_demo.py # evaluate a classifier with the corresponding embedding model
```
  
**For the _Identity + Sequent setting_ (T2), run the following codes:**

```bash
	$ python3 T2&T3_dataset.py # build the dataset for the _Identity + Sequent setting_
	# Before using the next four commands, make sure to change the file path in the script depending on the diagnosis level you are exploring, _e.g._, processed_icd_T2, processed_cat_T2, or processed_blk_T2. Do the same for the mimic file path, _e.g._, mimic_icd_T2, mimic_cat_T2, or mimic_blk_T2.
	$ python3 preprocess01.py 
	$ python3 preprocess02.py # run full preprocessing to obtain dictionaries
	$ python3 doc2vec.py --phase train # train doc2vec model
	$ python3 doc2vec.py --phase infer # infer doc2vec vectors
```

To train and evaluate the classification and the corresponding embedding model on **structured and unstructured** data
**For _level 4_ code, run the following:**
```bash
	$ python3 Identity+Sequent/T2_train_both_icd.py # train classifier model together with the embedding model 
	$ python3 Identity+Sequent/T2_evaluate_both_icd.py # evaluate a classifier with the corresponding embedding model
```

**For _level 3_ code, run the following:**
```bash
	$ python3 Identity+Sequent/T2_train_both_cat.py # train classifier model together with the embedding model 
	$ python3 Identity+Sequent/T2_evaluate_both_cat.py # evaluate a classifier with the corresponding embedding model
```

**For _level 2_ code, run the following:**
```bash
	$ python3 Identity+Sequent/T2_train_both_blk.py # train classifier model together with the embedding model 
	$ python3 Identity+Sequent/T2_evaluate_both_blk.py # evaluate a classifier with the corresponding embedding model
```


To train and evaluate the classification and the corresponding embedding model on only **unstructured** data
**For _level 4_ code, run the following:**
```bash
	$ python3 Identity+Sequent/T2_train_con_icd.py # train classifier model together with the embedding model 
	$ python3 Identity+Sequent/T2_evaluate_con_icd.py # evaluate a classifier with the corresponding embedding model
```

**For _level 3_ code, run the following:**
```bash
	$ python3 Identity+Sequent/T2_train_con_cat.py # train classifier model together with the embedding model 
	$ python3 Identity+Sequent/T2_evaluate_con_cat.py # evaluate a classifier with the corresponding embedding model
```

**For _level 2_ code, run the following:**
```bash
	$ python3 Identity+Sequent/T2_train_con_blk.py # train classifier model together with the embedding model 
	$ python3 Identity+Sequent/T2_evaluate_con_blk.py # evaluate a classifier with the corresponding embedding model
```


**For the _Identity + Directly Sequent_ setting (T3), run the following codes:**
```bash
	$ python3 T2&T3_dataset.py # build the dataset for the _Identity + Directly Sequent setting_
	# Before using the next four commands, make sure to change the file path in the script depending on the diagnosis level you are exploring, _e.g._, processed_icd_T3, processed_cat_T3, or processed_blk_T3. Do the same for the mimic file path, _e.g._, mimic_icd_T3, mimic_cat_T3, or mimic_blk_T3.
	$ python3 preprocess01.py 
	$ python3 preprocess02.py # run full preprocessing to obtain dictionaries
	$ python3 doc2vec.py --phase train # train doc2vec model
	$ python3 doc2vec.py --phase infer # infer doc2vec vectors
```

To train and evaluate the classification and the corresponding embedding model on **structured and unstructured** data
**For _level 4_ code, run the following:**
```bash
	$ python3 Identity+DSequent/T3_train_both_icd.py # train classifier model together with the embedding model 
	$ python3 Identity+DSequent/T3_evaluate_both_icd.py # evaluate a classifier with the corresponding embedding model
```

**For _level 3_ code, run the following:**
```bash
	$ python3 Identity+DSequent/T3_train_both_cat.py # train classifier model together with the embedding model 
	$ python3 Identity+DSequent/T3_evaluate_both_cat.py # evaluate a classifier with the corresponding embedding model
```
**For _level 2_ code, run the following:**
```bash
	$ python3 Identity+DSequent/T3_train_both_blk.py # train classifier model together with the embedding model 
	$ python3 Identity+DSequent/T3_evaluate_both_blk.py # evaluate a classifier with the corresponding embedding model
```


To train and evaluate the classification and the corresponding embedding model on only **unstructured** data
**For _level 4_ code, run the following:**
```bash
	$ python3 Identity+DSequent/T3_train_con_icd.py # train classifier model together with the embedding model 
	$ python3 Identity+DSequent/T3_evaluate_con_icd.py # evaluate a classifier with the corresponding embedding model
```

**For _level 3_ code, run the following:**
```bash
	$ python3 Identity+DSequent/T3_train_con_cat.py # train classifier model together with the embedding model 
	$ python3 Identity+DSequent/T3_evaluate_con_cat.py # evaluate a classifier with the corresponding embedding model
```

**For _level 2_ code, run the following:**
```bash
	$ python3 Identity+DSequent/T3_train_con_blk.py # train classifier model together with the embedding model 
	$ python3 Identity+DSequent/T3_evaluate_con_blk.py # evaluate a classifier with the corresponding embedding model
```

  
 
  
  
## Files and Folders

- `data.py`: tools to load the dataset, contains the main dataset class `Task1Dataset` and the data augmentation functions for the _Identity_ setting
- `T2_data_blk.py`: tools to load the dataset, contains the main dataset class `Task1Dataset` and the data augmentation functions for the _Identity + Sequent_ setting in the _level 2_ code
- `T2_data_cat.py`: tools to load the dataset, contains the main dataset class `Task1Dataset` and the data augmentation functions for the _Identity + Sequent_ setting in the _level 3_ code
- `T2_data_icd.py`: tools to load the dataset, contains the main dataset class `Task1Dataset` and the data augmentation functions for the _Identity + Sequent_ setting in the _level 4_ code
- `T3_data_blk.py`: tools to load the dataset, contains the main dataset class `Task1Dataset` and the data augmentation functions for the _Identity + Directly Sequent_ setting in the _level 2_ code
- `T3_data_cat.py`: tools to load the dataset, contains the main dataset class `Task1Dataset` and the data augmentation functions for the _Identity + Directly Sequent_ setting in the _level 3_ code
- `T3_data_icd.py`: tools to load the dataset, contains the main dataset class `Task1Dataset` and the data augmentation functions for the _Identity + Directly Sequent_ setting in the _level 4_ code
- `analogy_classif_both.py`: neural network to classify analogies for structured and unstructured data
- `analogy_classif_con.py`: neural network to classify analogies for unstructured data
- `analogy_classif_demo.py`: neural network to classify analogies for structured data
- `cnn_con.py`: neural network to embed clinical notes
- `cnn_dem.py`: neural network to embed static information
- `cnn_both.py`: neural network to embed static information and clinical notes
- `utils.py`: tools for the different codes
- **`Identity` folder**: contains all training and evaluating codes for _Identity_ setting
- **`Identity+Sequent` folder**: contains all training and evaluating codes for _Identity + Sequent_ setting
- **`Identity+DSequent` folder**: contains all training and evaluating codes for _Identity + Directly Sequent_ setting
