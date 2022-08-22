#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import pandas as pd
from  itertools import chain
import numpy as np



if __name__ == '__main__':

    ADMISSIONS_CSV = pd.read_csv('mimic_csv/ADMISSIONS.csv')
    ADMISSIONS_CSV.columns = map(str.lower, ADMISSIONS_CSV.columns)
    PATIENTS_CSV = pd.read_csv('mimic_csv/PATIENTS.csv')
    PATIENTS_CSV.columns = map(str.lower, PATIENTS_CSV.columns)

    df_adm = PATIENTS_CSV.merge(ADMISSIONS_CSV, on = "subject_id")

    df_adm = df_adm[[ 'subject_id', 'gender', 'dob', 'dod', 'hadm_id', 'admittime', 'dischtime',
                     'admission_type', 'insurance', 'marital_status', 'ethnicity', 'hospital_expire_flag',
                     'has_chartevents_data']]

    df_adm = df_adm[df_adm['has_chartevents_data'] == 1]
    df_adm['dob'] = pd.to_datetime(df_adm['dob']).dt.date
    df_adm['admittime'] = pd.to_datetime(df_adm['admittime']).dt.date
    df_adm['age'] = df_adm.apply(lambda e: (e['admittime'] - e['dob']).days/365.242, axis=1)

    df_adm = df_adm[df_adm['age'] >= 18]  # keep adults
    print('After removing non-adults:', len(df_adm), "unique patients:", df_adm['subject_id'].nunique())


    # caculate the admission time for each patient
    # keep the patient with more than two admissions
    df_adm['admit_times'] = df_adm.groupby(['subject_id'])['subject_id'].transform('size')
    df_adm["admit_times"].astype(int)

    #keep patients with at least 2 admissions
    df_adm = df_adm[df_adm['admit_times'] >= 2]
    print('patients with at least two adm', df_adm['subject_id'].nunique())

    df_data = df_adm[["subject_id", "hadm_id"]]

    #obtain all the permutations for the hospital admissions 
    def get_perms(row):
        patient_id = row['subject_id']
        extra_sub_df = df_data[df_data.subject_id == patient_id]
        perms = itertools.permutations(extra_sub_df.hadm_id, 2)
        return [p for p in perms]
    df_data['permutations'] = df_data.apply(lambda row: get_perms(row), axis =1)

    #split the permutation column to have the two stays associated with a patient_ID
    df1 = pd.DataFrame(list(chain.from_iterable(df_data.permutations)), columns=['hadm_id', "STAY B"]).reset_index(drop=True)
    df1['subject_id'] = np.repeat(df_data.subject_id.values, df_data['permutations'].str.len())
    df1.rename(columns = {'hadm_id':'STAY A'}, inplace = True)
    #drop druplicates
    df1 = df1.drop_duplicates(subset=['STAY A', 'STAY B'], keep='first')


    print(len(df1)) #46986

    #Save as a csv file 
    df1.to_csv('T1_Identity.csv')

    #Uncomment the followinng lines for a subset with the first 200 patients
    #df_subset = pd.read_csv('T1_Identity.csv',nrows=2270)
    #print(df_subset['subject_id'].nunique())
    #df_subset.to_csv('T1_IDENTITY_200.csv')




