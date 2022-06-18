#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils import bin_age, convert_icd_group, clean_text

if __name__ == '__main__':
    ADMISSIONS_CSV = pd.read_csv('mimic_csv/ADMISSIONS.csv')
    ADMISSIONS_CSV.columns = map(str.lower, ADMISSIONS_CSV.columns)
    PATIENTS_CSV = pd.read_csv('mimic_csv/PATIENTS.csv')
    PATIENTS_CSV.columns = map(str.lower, PATIENTS_CSV.columns)

    df_adm = PATIENTS_CSV.merge(ADMISSIONS_CSV, on = "subject_id")

    df_adm = df_adm[[ 'subject_id', 'gender', 'dob', 'dod', 'hadm_id', 'admittime', 'dischtime',
                     'admission_type', 'insurance', 'marital_status', 'ethnicity', 'hospital_expire_flag',
                     'has_chartevents_data']]

    df_adm.to_csv("mimic_id/adm_details.csv")


    # DEFINE COHORTS

    df_adm = pd.read_csv('mimic_id/adm_details.csv', parse_dates=['dob', 'dod', 'admittime', 'dischtime'])
    print('Total admissions:', len(df_adm))

    #keep patients with chartevents data
    df_adm = df_adm[df_adm['has_chartevents_data'] == 1]
    #keep only adult patients
    df_adm['dob'] = pd.to_datetime(df_adm['dob']).dt.date
    df_adm['admittime'] = pd.to_datetime(df_adm['admittime']).dt.date
    df_adm['age'] = df_adm.apply(lambda e: (e['admittime'] - e['dob']).days/365.242, axis=1)
    df_adm = df_adm[df_adm['age'] >= 18] 
    #split into age groups
    df_adm['age'] = df_adm['age'].apply(bin_age)
    print('After removing non-adults:', len(df_adm), "unique patients:", df_adm['subject_id'].nunique())

    # caculate the admission time for each patient
    # keep the patient with multiple admissions
    df_adm['admit_times'] = df_adm.groupby(['subject_id'])['subject_id'].transform('size')
    df_adm["admit_times"].astype(int)

    #patient with at least two admissions
    df_adm = df_adm[df_adm['admit_times'] >= 2]
    print('patients with at least two adm', df_adm['subject_id'].nunique())


    #process patient demographics
    print('Processing patients demographics...')
    df_adm['marital_status'] = df_adm['marital_status'].fillna('Unknown')
    df_static = df_adm[['hadm_id', 'age', 'gender', 'admission_type', 'insurance',
                'marital_status', 'ethnicity']]
    print(len(df_static))
    df_static.to_csv('processed_id/demo.csv', index=None)


    #extract first 24 hours signals
    def get_signals(start_hr, end_hr):
        df_adm = pd.read_csv('mimic_id/adm_details.csv',parse_dates=['admittime'])
        adm_ids = df_adm.hadm_id.tolist()
        for signal in ['vital', 'lab']:
            df = pd.read_csv('mimic_csv/pivoted_{}.csv'.format(signal), parse_dates=['charttime'])
            df = df.merge(df_adm[['hadm_id', 'admittime']], on='hadm_id')
            df = df[df.hadm_id.isin(adm_ids)]
            df['hr'] = (df.charttime - df.admittime) / np.timedelta64(1, 'h')
            df = df[(df.hr <= end_hr) & (df.hr >= start_hr)]
            df = df.set_index('hadm_id').groupby('hadm_id').resample(
                'H', on='charttime').mean().reset_index()
            df.to_csv('mimic_id/{}.csv'.format(signal), index=None)
        
        df = pd.read_csv('mimic_id/vital.csv', parse_dates=['charttime'])[
            ['hadm_id', 'charttime', 'heartrate', 'sysbp', 'diasbp', 'meanbp', 'resprate', 'tempc', 'spo2']]
        df_lab = pd.read_csv('mimic_id/lab.csv',
                             parse_dates=['charttime'])
        df = df.merge(df_lab, on=['hadm_id', 'charttime'], how='outer')
        df = df.merge(df_adm[['hadm_id', 'admittime']], on='hadm_id')
        df['charttime'] = ((df.charttime - df.admittime) / np.timedelta64(1, 'h'))
        df['charttime'] = df['charttime'].apply(np.ceil) + 1     
        df = df[(df.charttime <= end_hr) & (df.charttime >= start_hr)]
        df = df.sort_values(['hadm_id', 'charttime'])
        df['charttime'] = df['charttime'].map(lambda x: int(x))
        df = df.drop(['admittime', 'hr'], axis=1)
        na_thres = 3
        df = df.dropna(thresh=na_thres) 
        df.to_csv('processed_id/features.csv', index=None)

    get_signals(1, 24)

    # process clinical notes
    def extract_notes(df_notes): 
        '''Extract clinical notes grouped by hospital admission ID'''
        df_early = df_notes[df_notes['category'].isin(categories)]
        df_early['text'] = df_early['text'].apply(clean_text)
        df_early[['hadm_id', 'category', 'text']].to_csv('processed_id/notes.csv', index=None)

    print('Reading data...')
    categories = ['Nursing', 'Nursing/other', 'Physician ', 'Radiology']
    df_notes = pd.read_csv('mimic_csv/NOTEEVENTS.csv', parse_dates=['CHARTTIME'])
    df_notes.columns = map(str.lower, df_notes.columns)
    #exclude notes that have errors, lack an HADM_ID, or lack chartime info
    df_notes = df_notes[df_notes['iserror'].isnull()]
    df_notes = df_notes[~df_notes['hadm_id'].isnull()]
    df_notes = df_notes[~df_notes['charttime'].isnull()]

    df_adm = pd.read_csv('mimic_id/adm_details.csv', parse_dates=['admittime'])
    df_notes = df_notes.merge(df_adm, on='hadm_id', how='left')

    extract_notes(df_notes) 

    # Merge_IDS
    #keep only HADM IDS that have demographics, temporal signals, and notes

    df_static = pd.read_csv('processed_id/demo.csv')
    print(len(df_static))
    df_features = pd.read_csv('processed_id/features.csv')
    print(len(df_features))
    df_notes = pd.read_csv('processed_id/notes.csv')
    print(len(df_notes))
    df_notes = df_notes[~df_notes['text'].isnull()]
    adm_ids = df_static['hadm_id'].tolist()

    adm_ids = np.intersect1d(adm_ids, df_features['hadm_id'].unique().tolist())
    adm_ids = np.intersect1d(adm_ids, df_notes['hadm_id'].unique().tolist())


    df_static[df_static['hadm_id'].isin(adm_ids)].to_csv('processed_id/demo.csv', index=None)
    print(len(df_static))
    df_features[df_features['hadm_id'].isin(adm_ids)].to_csv('processed_id/features.csv', index=None)
    print(len(df_features))
    df_notes[df_notes['hadm_id'].isin(adm_ids)].to_csv('processed_id/notes.csv', index=None)
    print(len(df_notes))


    # 17286 352471 1560325 17286 352471 1560292

    # Based on the dataset, keep only HADM_IDs in both demo.csv, temporal signals, and notes.csv
    #you can use a subset or the entire file
    # Here we are using a subset of the first 200 patients
    df_static = pd.read_csv('processed_id/demo.csv')
    df_dataset = pd.read_csv('T1_IDENTITY_200.csv')
    print(len(df_dataset))
    adm_ids = df_static['hadm_id'].tolist()


    df_dataset = df_dataset[df_dataset['STAY A'].isin(adm_ids)]
    df_dataset = df_dataset[df_dataset['STAY B'].isin(adm_ids)]
    df_dataset = df_dataset.drop(["Unnamed: 0", "Unnamed: 0.1"], axis= 1)

    #Save as csv or txt file. Txt file is better for building analogies later.
    #df_dataset.to_csv('AN_IDENTITY_200.csv', index=None)
    df_dataset.to_csv('T1_IDENTITY_200.txt', header=None, index=None, sep=',', mode='a')




