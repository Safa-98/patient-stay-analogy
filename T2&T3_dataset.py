#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from utils import convert_icd_group, clean_files
from  itertools import chain
import numpy as np


# In[40]:


ICD = pd.read_csv('DIAGNOSES_ICD.csv')
ICD.columns = map(str.lower, ICD.columns)
ICD = ICD[ICD.seq_num == 1.0] #kept only first diagnosis
ICD =  ICD.drop(["seq_num","subject_id", "row_id"],axis=1)

#if you want to discover the third diagnosis level (3 digit category), uncomment these lines
# ICD["Length"]= ICD["icd9_code"].str.len()
# ICD.loc[(ICD.Length == 4), 'icd9_code']=ICD['icd9_code'].astype(str).str[:-1]
# ICD.loc[(ICD.Length == 5), 'icd9_code']=ICD['icd9_code'].astype(str).str[:-2]
# ICD =ICD.drop(["Length"],axis=1)

#uncomment this line if you want to discover the second diagnosis level (block)
#ICD['icd9_code'] = ICD['icd9_code'].apply(convert_icd_group)


#run the function clean_files() to obtain clean csv file
clean_files()
diag_df = pd.read_csv("clean_diagnosis.csv")
diag_df = diag_df.drop(["Unnamed: 0", "icd9_code"], axis=1)


diag_df = diag_df.merge(ICD, on='hadm_id')



# we need to drop columns with disease happening only once to establish the relation
counts_col1 = diag_df.groupby("icd9_code")["icd9_code"].transform(len)
mask = (counts_col1 > 1)
diag_df =diag_df[mask]


#uncomment this command to see the value count for each ICD code
#diag_df['icd9_code'].value_counts() #925



# when we dropped icd_codes that appeared only once, the number of admission for that patient drops by one
# recaculate the admission time for each patient
# keep the patient with at least two admissions
diag_df['admit_times'] = diag_df.groupby(['subject_id'])['subject_id'].transform('size')
diag_df["admit_times"].astype(int)
diag_df = diag_df[diag_df['admit_times'] >= 2]
print('patient with more than one adm', diag_df['subject_id'].nunique())

#For the 2nd setting, we include a pairs of STAY 1 followed by stay 2 and  STAY 2 followed by STAY 3 and STAY 1 followed by STAY 3 (as stays can happen in between)
#For the 3rd setting, we include pairs of STAY 1 followed by stay 2 and  STAY 2 followed by STAY 3(no stay can happen in between)
#Here we are forming pairs of STAY 1 followed by STAY 2 (S1_S2)
df_data = diag_df[["subject_id", "hadm_id", "icd9_code", "admittime", "dischtime"]]

df_data['next_stay'] = df_data.groupby('subject_id')['hadm_id'].shift(-1)
df_data['next_stay'] = df_data['next_stay'].astype("Int32")
df_data = df_data[[ "hadm_id",  "next_stay","subject_id", "icd9_code"]]


df_data = df_data[df_data['next_stay'].notna()] #remove rows where next stay is NA
df_data.rename(columns = {'hadm_id':'STAY A'}, inplace = True)
df_data.rename(columns = {'next_stay':'STAY B'}, inplace = True)


# dropping again will influence the number of disease happening, therefore we drop those happening only once
counts_col1 = df_data.groupby("icd9_code")["icd9_code"].transform(len)
mask = (counts_col1 > 1)
df_data =df_data[mask]


#df_data['icd9_code'].value_counts() #727 diagnosis shared btwn at least two patients


df_data.to_csv("S1_S2.csv") 


# STAY 2 followed by STAY 3 (S2_S3)



df_data1 = diag_df[["subject_id", "hadm_id", "icd9_code", "admittime", "dischtime"]]

df_data1['next_stay'] = df_data1.groupby('subject_id')['hadm_id'].shift(-1)
df_data1['next_stay2'] = df_data1.groupby('subject_id')['hadm_id'].shift(-2)


df_data1['next_stay'] = df_data1['next_stay'].astype("Int32")
df_data1['next_stay2'] = df_data1['next_stay2'].astype("Int32")


df_data1 = df_data1[df_data1['next_stay'].notna()]
df_data1 = df_data1[df_data1['next_stay2'].notna()]

#print('patient', df_data1['subject_id'].nunique())


df_data1 = df_data1[[ "next_stay",  "next_stay2","subject_id"]]

df_data1.rename(columns = {'next_stay':'hadm_id'}, inplace = True)

df_data1 = df_data1.merge(ICD, on='hadm_id')


#df_data1['icd9_code'].value_counts() #608 diagnosis shared btwn at least two patients

# we need to drop columns with disease happening only once to establish the relation
counts_col1 = df_data1.groupby("icd9_code")["icd9_code"].transform(len)
mask = (counts_col1 > 1)
df_data1 =df_data1[mask]


df_data1.rename(columns = {'hadm_id':'STAY A'}, inplace = True)
df_data1.rename(columns = {'next_stay2':'STAY B'}, inplace = True)


df_data1.to_csv("S2_S3.csv")


# STAY 1 followed by STAY 3 (S1_S3)


df_data2 = diag_df[["subject_id", "hadm_id", "icd9_code", "admittime", "dischtime"]]

df_data2['next_stay2'] = df_data2.groupby('subject_id')['hadm_id'].shift(-2)
df_data2['next_stay2'] = df_data2['next_stay2'].astype("Int32")

df_data2 = df_data2[["hadm_id", "next_stay2", "subject_id", "icd9_code"]]

df_data2.rename(columns = {'hadm_id':'STAY A'}, inplace = True)
df_data2.rename(columns = {'next_stay2':'STAY B'}, inplace = True)

df_data2 = df_data2[df_data2['STAY B'].notna()]


#df_data2['icd9_code'].value_counts() #628 diagnosis shared btwn at least two patients


# we need to drop columns with disease happening only once to establish the relation
counts_col1 = df_data2.groupby("icd9_code")["icd9_code"].transform(len)
mask = (counts_col1 > 1)
df_data2 =df_data2[mask]


df_data2.to_csv("datasets/S1_S3.csv")


# To form the dataset for the third setting - T3
#merge 2 csv files
df1 = pd.read_csv("S1_S2.csv")
df3 = pd.read_csv("S2_S3.csv")


final_df = pd.concat([df1,df3])
final_df = final_df.drop(["Unnamed: 0"], axis =1)

print(len(final_df)) 

diag_once = final_df.drop_duplicates(subset=["STAY A", "STAY B", "subject_id", "icd9_code"], keep='first')
diag_once.drop(["Unnamed: 0"], axis=1)

#change the acronym ICD based on what level of the diagnosis constraint you are using
diag_once.to_csv("T3_ID+DSEQ_ICD.csv")

print(len(diag_once))



# To form the dataset for the second setting - T2
#merge all 3 csv files

df1 = pd.read_csv("S1_S2.csv")
df2 = pd.read_csv("S1_S3.csv")
df3 = pd.read_csv("S2_S3.csv")

final_df = pd.concat([df1,df2,df3]) 
final_df = final_df.drop(["Unnamed: 0"], axis =1)

print(len(final_df)) 


diag_once = final_df.drop_duplicates(subset=["STAY A", "STAY B", "subject_id", "icd9_code"], keep='first')
print(len(diag_once))

#change the acronym ICD based on what level of the diagnosis constraint you are using
diag_once.to_csv("T2_ID+SEQ_ICD.csv")
