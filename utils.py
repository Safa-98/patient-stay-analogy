import pandas as pd
import numpy as np

import torch

import nltk
from nltk.corpus import stopwords
from sklearn import metrics

import re
import os
import json
import random
from timeit import default_timer
import torch.nn.functional as F
from contextlib import contextmanager


stops = set(stopwords.words("english"))
regex_punctuation = re.compile('[\',\.\-/\n]')
regex_alphanum = re.compile('[^a-zA-Z0-9 ]')
regex_num = re.compile('\d[\d ]+')
regex_spaces = re.compile('\s+')

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start
    
def bin_age(age):
    if age < 25:
        return '18-25'
    elif age < 45:
        return '25-45'
    elif age < 65:
        return '45-65'
    elif age < 89:
        return '65-89'
    else:
        return '89+'


def clean_text(text):
    text = text.lower().strip()

    # remove phi tags
    tags = re.findall('\[\*\*.*?\*\*\]', text)
    for tag in set(tags):
        text = text.replace(tag, ' ')

    text = re.sub(regex_punctuation, ' ', text)
    text = re.sub(regex_alphanum, '', text)
    text = re.sub(regex_num, ' 0 ', text)
    text = re.sub(regex_spaces, ' ', text)
    return text.strip()

def text2words(text):
    words = text.split()
    words = [w for w in words if not w in stops]
    return words


def convert_icd_group(icd):
    icd = str(icd)
    if icd.startswith('V'):
        return 19
    if icd.startswith('E'):
        return 20
    icd = int(icd[:3])
    if icd <= 139:
        return 1
    elif icd <= 239:
        return 2
    elif icd <= 279:
        return 3
    elif icd <= 289:
        return 4
    elif icd <= 319:
        return 5
    elif icd <= 389:
        return 6
    elif icd <= 459:
        return 7
    elif icd <= 519:
        return 8
    elif icd <= 579:
        return 9
    elif icd < 629:
        return 10
    elif icd <= 679:
        return 11
    elif icd <= 709:
        return 12
    elif icd <= 739:
        return 13
    elif icd <= 759:
        return 14
    elif icd <= 779:
        return np.nan
    elif icd <= 789:
        return 15
    elif icd <= 796:
        return 16
    elif icd <= 799:
        return 17
    else:
        return 18


def cal_metric(y_true, probs):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, probs)
    optimal_idx = np.argmax(np.sqrt(tpr * (1-fpr)))
    optimal_threshold = thresholds[optimal_idx]
    preds = (probs > optimal_threshold).astype(int)
    auc = metrics.roc_auc_score(y_true, probs)
    auprc = metrics.average_precision_score(y_true, probs)
    f1 = metrics.f1_score(y_true, preds)
    return f1, auc, auprc


def save_model(all_dict, name='best_model.pth'):
    model_dir = all_dict['args'].model_dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = os.path.join(model_dir, name)
    torch.save(all_dict, model_path)


def load_model(model_dict, name='best_model.pth'):
    model = model_dict['model']
    model_dir = model_dict['args'].model_dir
    model_path = os.path.join(model_dir, name)
    if os.path.exists(model_path):
        all_dict = torch.load(model_path)
        model.load_state_dict(all_dict['state_dict'])
        return model, all_dict['best_metric'], all_dict['epoch']
    else:
        return model, 0, 1
    

def get_ids(split_json):
    splits = list(range(10))
    adm_ids = json.load(open(split_json))
    train_ids = np.hstack([adm_ids[t] for t in splits[:7]])
    val_ids = np.hstack([adm_ids[t] for t in splits[7:8]])
    test_ids = np.hstack([adm_ids[t] for t in splits[8:]])
    train_ids = [adm_id[-10:-4] for adm_id in train_ids]
    val_ids = [adm_id[-10:-4] for adm_id in val_ids]
    test_ids = [adm_id[-10:-4] for adm_id in test_ids]
    return train_ids, val_ids, test_ids


def get_ids2(split_json, seed):
    splits = list(range(10))
    random.Random(seed).shuffle(splits)
    adm_ids = json.load(open(split_json))
    train_ids = np.hstack([adm_ids[t] for t in splits[:7]])
    val_ids = np.hstack([adm_ids[t] for t in splits[7:8]])
    test_ids = np.hstack([adm_ids[t] for t in splits[8:]])
    train_ids = [adm_id[-10:-4] for adm_id in train_ids]
    val_ids = [adm_id[-10:-4] for adm_id in val_ids]
    test_ids = [adm_id[-10:-4] for adm_id in test_ids]
    return train_ids, val_ids, test_ids


def balance_samples(df, times, task):
    df_pos = df[df[task] == 1]
    df_neg = df[df[task] == 0]
    df_neg = df_neg.sample(n=times * len(df_pos), random_state=42)
    df = pd.concat([df_pos, df_neg]).sort_values('hadm_id')
    return df


def mkdir(d):
    path = d.split('/')
    for i in range(len(path)):
        d = '/'.join(path[:i+1])
        if not os.path.exists(d):
            os.mkdir(d)


def csv_split(line, sc=','):
    res = []
    inside = 0
    s = ''
    for c in line:
        if inside == 0 and c == sc:
            res.append(s)
            s = ''
        else:
            if c == '"':
                inside = 1 - inside
            s = s + c
    res.append(s)
    return res

def get_accuracy_classification(y_true, y_pred):
    '''Computes the accuracy for a batch of data of the classification task.

    Arguments:
    y_true -- The tensor of expected values.
    y_pred -- The tensor of predicted values.'''
    assert y_true.size() == y_pred.size()
    y_pred = y_pred > 0.5
    if y_pred.ndim > 1:
        return (y_true == y_pred).sum().item() / y_true.size(0)
    else:
        return (y_true == y_pred).sum().item()


def tpr_tnr_balacc_harmacc_f1(tp,tn,fp,fn):
    """Compute usefull classification statistics:
        - true positive rate (TPR), i.e., accuracy on positive samples
        - true negative rate (TNR), i.e., accuracy on negative samples
        - ballanced accuracy (balacc), i.e., mean of TPR and TNR
        - harmonic mean accuracy (harmacc), i.e., harmonic mean of TPR and TNR
        - F1-score
    :param tp: Number of true positives, i.e., number of correctly preddicted positive samples.
    :param tn: Number of true negatives, i.e., number of correctly preddicted negative samples.
    :param fp: Number of false positives, i.e., number of negative samples predicted as positive.
    :param fn: Number of false negtives, i.e., number of positive samples predicted as negtive.
    :return: TPR, TNR, balacc, harmacc, F1
    """
    tpr = tp / (tp + fn) # true positive rate i.e. accuracy on positive examples
    tnr = tn / (tn + fp) # true negative rate i.e. accuracy on negative examples
    balacc = (tpr + tnr) / 2 # mean of TPR & TNR
    harmacc = (2 * tpr * tnr) / (tpr + tnr) # harmonic mean of TPR & TNR
    f1 = (2 * tp) / (2 * tp + fp + fn) # actual F1 score

    return tpr, tnr, balacc, harmacc, f1

def clean_files(): 
    ADMISSIONS_CSV = pd.read_csv('mimic_csv/ADMISSIONS.csv')
    PATIENTS_CSV = pd.read_csv('mimic_csv/PATIENTS.csv')
    ICD = pd.read_csv('DIAGNOSES_ICD.csv')
    ICD.columns = map(str.lower, ICD.columns)
    df = pd.merge(PATIENTS_CSV, ADMISSIONS_CSV, on = 'SUBJECT_ID' )
    df.columns = map(str.lower, df.columns)

    #keep only adults
    df['dob'] = pd.to_datetime(df['dob']).dt.date
    df['admittime'] = pd.to_datetime(df['admittime']).dt.date
    df['age'] = df.apply(lambda e: (e['admittime'] - e['dob']).days/365.242, axis=1)
    df = df[df['age'] >= 18]  # keep adults
    print('Adult admissions:', len(df))

    #keep patients with chartevents
    df = df[df['has_chartevents_data'] == 1]
    print('admissions w\ chart:', len(df))

    # caculate how many times each patient has been admitted
    # keep the patient with at least 2 admissions
    df['admit_times'] = df.groupby(['subject_id'])['subject_id'].transform('size')
    df["admit_times"].astype(int)
    df = df[df['admit_times'] >= 2]
    print('patient with more than one adm', len(df))
    df = df.drop(['row_id_x', 'dod_hosp', 'dod_ssn', 'expire_flag', 'row_id_y', 'deathtime', 'admission_location', 'discharge_location', 'language', 'religion', 'edregtime', 'edouttime', 'diagnosis', 'age', 'admit_times'] , axis=1)

    #merge the ICD codes associated with single hospital admission
    diag_df = df.merge(ICD, on="hadm_id")
    diag_df = diag_df[diag_df.seq_num == 1.0] #kept only first diagnosis
    diag_df = diag_df.drop(['row_id','subject_id_y'], axis = 1)
    diag_df.rename(columns = {'subject_id_x':'subject_id'}, inplace = True)
    diag_df.to_csv("clean_diagnosis.csv")


def convert_icd_group(icd):
    icd = str(icd)
    if str(icd).startswith('V'):
        icd = icd.lstrip('V')
        icd = str(icd)
        icd = int(icd[:2])
        if icd <= int('09'):
            return 'V09'
        elif icd <= 19:
            return 'V19'
        elif icd <= 29:
            return 'V29'
        elif icd <= 39:
            return 'V39'
        elif icd <= 49:
            return 'V49'
        elif icd <= 59:
            return 'V59'
        elif icd <= 69:
            return 'V69' 
        elif icd <= 82:
            return 'V82' 
        elif icd == 85:
            return 'V85'  
        elif icd == 86:
            return 'V86'
        elif icd == 87:
            return 'V87'
        elif icd == 88:
            return 'V88'
        elif icd == 89:
            return 'V89'
        elif icd == 90:
            return 'V90'
        elif icd <= 91:
            return 'V91'     

    if str(icd).startswith('E'):
        icd = icd.lstrip('E')
        icd = str(icd)
        icd = int(icd[:2])
        if icd <= int('000'):
            return 'E000'
        elif icd <= int('030'):
            return 'E030'
        elif icd <= 807:
            return 'E807'
        elif icd <= 819:
            return 'E819'
        elif icd <= 825:
            return 'E825'
        elif icd <= 829:
            return 'E829' 
        elif icd <= 838:
            return 'E838' 
        elif icd <= 845:
            return 'E845' 
        elif icd <= 849:
            return 'E849'
        elif icd <= 858:
            return 'E858' 
        elif icd <= 869:
            return 'E869' 
        elif icd <= 876:
            return 'E876' 
        elif icd <= 879:
            return 'E879' 
        elif icd <= 888:
            return 'E888' 
        elif icd <= 899:
            return 'E899' 
        elif icd <= 909:
            return 'E909' 
        elif icd <= 915:
            return 'E915' 
        elif icd <= 928:
            return 'E928' 
        elif icd <= 949:
            return 'E949' 
        elif icd == 929:
            return 'E929'
        elif icd <= 959:
            return 'E959' 
        elif icd <= 969:
            return 'E969' 
        elif icd <= 978:
            return 'E978' 
        elif icd <= 989:
            return 'E989' 
        elif icd <= 999:
            return 'E999'
        
    if str(icd).isdigit():  
        icd = str(icd)
        icd = int(icd[:3])
        if icd <= int('009'):
            return int('009')
        elif icd <= int('018'):
            return int('018')
        elif icd <= int('027'):
            return int('027')
        elif icd <= int('041'):
            return int('041')
        elif icd == int('042'):
            return int('042')
        elif icd <= int('049'):
            return int('049')
        elif icd <= int('059'):
            return int('059')
        elif icd <= int('066'):
            return int('066')
        elif icd <= int('079'):
            return int('079')
        elif icd < int('088'):
            return int('088')
        elif icd <= int('099'):
            return int('099')
        elif icd <= 104:
            return 104
        elif icd <= 118:
            return 118
        elif icd <= 129:
            return 129
        elif icd <= 136:
            return 136
        elif icd <= 139:
            return 139
        elif icd <= 149:
            return 149
        elif icd <= 159:
            return 159
        elif icd <= 165:
            return 165
        elif icd <= 175:
            return 175
        elif icd == 176:
            return 176
        elif icd <= 189:
            return 189
        elif icd <= 199:
            return 199
        elif icd <= 208:
            return 208
        elif icd == 209:
            return 209
        elif icd <= 229:
            return 229
        elif icd <= 234:
            return 234
        elif icd <= 238:
            return 238
        elif icd == 239:
            return 239
        elif icd <= 246:
            return 246
        elif icd <= 259:
            return 259
        elif icd <= 269:
            return 269
        elif icd <= 279:
            return 279
        elif icd <= 289:
            return 289
        elif icd <= 294:
            return 294
        elif icd <= 299:
            return 299
        elif icd <= 316:
            return 316
        elif icd <= 319:
            return 319
        elif icd <= 326:
            return 326
        elif icd <= 337:
            return 337
        elif icd == 338:
            return 338
        elif icd == 339:
            return 339
        elif icd <= 349:
            return 349
        elif icd <= 359:
            return 359
        elif icd <= 379:
            return 379
        elif icd <= 389:
            return 389
        elif icd <= 392:
            return 392
        elif icd <= 398:
            return 398
        elif icd <= 405:
            return 405
        elif icd <= 414:
            return 414
        elif icd <= 417:
            return 417
        elif icd <= 429:
            return 429
        elif icd <= 438:
            return 438
        elif icd <= 448:
            return 448
        elif icd <= 459:
            return 459
        elif icd <= 466:
            return 466
        elif icd <= 478:
            return 478
        elif icd <= 488:
            return 488
        elif icd <= 496:
            return 496
        elif icd <= 508:
            return 508
        elif icd <= 519:
            return 519
        elif icd <= 529:
            return 529
        elif icd <= 539:
            return 539
        elif icd <= 543:
            return 543
        elif icd <= 553:
            return 553
        elif icd <= 558:
            return 558
        elif icd <= 569:
            return 569
        elif icd <= 579:
            return 579
        elif icd <= 589:
            return 589
        elif icd <= 599:
            return 599
        elif icd <= 608:
            return 608
        elif icd <= 611:
            return 611
        elif icd <= 616:
            return 616
        elif icd <= 629:
            return 629
        elif icd <= 639:
            return 639
        elif icd <= 649:
            return 649
        elif icd <= 659:
            return 659
        elif icd <= 669:
            return 669
        elif icd <= 677:
            return 677
        elif icd <= 679:
            return 679
        elif icd <= 686:
            return 686
        elif icd <= 698:
            return 698
        elif icd <= 709:
            return 709
        elif icd <= 719:
            return 719
        elif icd <= 724:
            return 724
        elif icd <= 729:
            return 729
        elif icd <= 739:
            return 739
        elif icd <= 759:
            return 759
        elif icd <= 763:
            return 763
        elif icd <= 779:
            return 779
        elif icd <= 789:
            return 789
        elif icd <= 796:
            return 796
        elif icd <= 799:
            return 799
        elif icd <= 804:
            return 804
        elif icd <= 809:
            return 809
        elif icd <= 819:
            return 819
        elif icd <= 829:
            return 829
        elif icd <= 839:
            return 839
        elif icd <= 848:
            return 848
        elif icd <= 854:
            return 854
        elif icd <= 869:
            return 869
        elif icd <= 879:
            return 879
        elif icd <= 887:
            return 887
        elif icd <= 897:
            return 897
        elif icd <= 904:
            return 904
        elif icd <= 909:
            return 909
        elif icd <= 919:
            return 919
        elif icd <= 924:
            return 924
        elif icd <= 879:
            return 879
        elif icd <= 929:
            return 929
        elif icd <= 939:
            return 939
        elif icd <= 949:
            return 949
        elif icd <= 957:
            return 957
        elif icd <= 959:
            return 959
        elif icd <= 979:
            return 979
        elif icd <= 989:
            return 989
        elif icd <= 995:
            return 995
        elif icd <= 999:
            return 999