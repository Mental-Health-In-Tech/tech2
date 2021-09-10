# imports
import pandas as pd
import requests
from typing import Mapping
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

################ Mother Functions ########################

def prep_the_strings(df):
    '''
    This function preps the mental health data through the use of a number of functions.
    '''
    df = convert_lower(df)
    df = to_datetime(df)
    df = drop_age_outliers(df)
    df = drop_columns(df)
    df = fill_work_nulls(df)
    df = clean_female(df)
    df = clean_male(df)
    df = clean_other(df)
    df = remove_countries(df)
    df = target_correction(df)
    df = employer_status(df)
    

    return df


def prep_encode(df):
    '''
    This function encodes the mental data into numeric values through a number of functions.
    '''
    
    df = encode_yes_no_dont_know(df)
    df = encode_yes_no_columns(df)
    df = encode_no_employee(df)
    df = encoding_no_yes_maybe(df)
    df = encode_supervisor(df)
    df = encode_coworkers(df)
    df = encode_care(df)
    df = encode_work(df)
    df = encode_leave(df)
    df = encode_gender(df)

    
    return df

########################## Child Functions ##############################

def to_datetime(df):
    '''
    This function takes in a pandas DataFrame and converts the target 'timestamp' column to datetime, and returns the DataFrame.
    '''
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def drop_age_outliers(df):
    '''
    This function takes in a pandas DataFrame, and removes all ages below 18 and over 85, and returns a pandas DataFrame with working-age employees only.
    '''
    # Drops those younger than 18
    df = df[df['age'] >= 18]
    # Drops those older than 85
    df = df[df['age'] <= 85]
    
    return df

def drop_columns(df):
    '''
    This function drops the 'comment' and 'state' columns as they are not needed.
    '''
    df = df.drop(columns=(['comments', 'state']))

    return df

def column_lower(df):
    '''
    This function takes the capitalized column names and converts them to lowercase.
    '''
    df = df.rename(columns = {'Timestamp':'timestamp',
                                'Age':'age',
                                'Gender':'gender',
                                'Country':'country'})
    return df

def fill_self_employed_nulls(df):
    '''
    This function takes the self_employed column and converts the nulls to 0.
    '''
    df.self_employed.fillna(value = 0, inplace = True)

    return df
    
def clean_male(df):
    '''
    This function takes the 'male' column and all its answers and converts them to 'male.
    '''
    df.gender.replace(to_replace = ['M','Male','male','m','Male-ish',
                                    'maile','something kinda male?','Mal',
                                    'Male (CIS)', 'Make','Guy (-ish) ^_^',
                                    'Male ','Man','msle','Mail','cis male',
                                    'Malr','Cis Man','Cis Male',
                                    'ostensibly male, unsure what that really means'],
                                    value = 'male', inplace = True)
    return df

def clean_female(df):
    '''
    This function takes the 'female' column and all its amswers and converts them to 'female'.
    '''
    df.gender.replace(to_replace = ['Female','female','Femake','Female ',
                                       'cis-female/femme','Woman','f','woman',
                                       'femail','Female (cis)','Cis Female','F'], value = 'female',
                                       inplace = True)

    return df

def clean_other(df):
    '''
    This function takes the 'other' column and all its answers and converts them to 'other'.
    '''
    df.gender.replace(to_replace = ['Trans-female','queer/she/they','non-binary',
                                       'Nah','All','Enby','fluid','Genderqueer',
                                       'Androgyne','Agender','male leaning androgynous',
                                       'Trans woman','Neuter','Female (trans)', 'queer',
                                       'A little about you', 'p'],  value = 'other',
                                       inplace = True)
    return df

def remove_countries(df):
    '''
    This function removes uneeded countries.
    '''
    countries = ['United States','Canada','Mexico','Switzerland',
                                   'Germany','Ireland','Poland','Austria','Italy',
                                   'Sweden','Spain','Norway','Czech Repulbic','Denmark',
                                   'Latvia','Moldova','Georgia','Romania','Finland','Bulgaria',
                                   'France','Slovenia','Russia','Bosnia and Herzegovina']
    df = df[df['country'].isin(countries)]

    return df

def fill_work_nulls(df):
    '''
    This function takes the work_interfere column and converts the nulls to 'not applicable'.
    '''
    df.work_interfere.fillna(value = 'Not applicable', inplace = True)

    return df

def encode_leave(df):
    '''
    This function takes the leave column and endcodes the values to '0', '1', '2', '3'or '4'.
    '''
    df['leave'] = df['leave'].map({"Very difficult":0,"Somewhat difficult":1,"Don't know":2,"Somewhat easy":3,"Very easy":4})
    return df

def encode_gender(df):
    '''
    This function takes the gender column and encoodes the values to '0', '1' or '2'.
    '''
    df['gender'] = df['gender'].map({'male': 0, 'female': 1, 'other': 2})

    return df

def encode_work(df):
    '''
    This function takes the work_interfere column and encodes the values to '0' or '1'.
    '''
    df['work_interfere'] = df['work_interfere'].map({'Never':0,'Not applicable':0, 'Rarely':1,'Sometimes':1,'Often':1})

    return df

def encode_care(df):
    '''
    This function takes the care_options column and encodes the values to '0' or '1'.
    '''
    df['care_options'] = df['care_options'].map({'No':0,'Yes':1,'Not sure':2})

    return df

def encode_coworkers(df):
    '''
    This function takes the coworkers column and encodes the values to '0', '1' or '2'.
    '''
    df['coworkers'] = df['coworkers'].map({'No':0, 'Yes':1, 'Some of them':2})

    return df

def encode_supervisor(df):
    '''
    This function takes the supervisor column and encodes the values to '0', '1' or '2'.
    '''
    df['supervisor'] = df['supervisor'].map({'No':0, 'Yes':1, 'Some of them':2})

    return df

def encoding_no_yes_maybe(df):
    '''
    This function takes the columns listed below in 'col_list' and encodes them to '0', '1' or '2'.
    '''
    col_list = ['mental_health_consequence', 'phys_health_consequence', 'mental_health_interview', 'phys_health_interview']   
    for col in col_list:
        df[col] = df[col].map({'No': 0, 'Yes': 1, 'Maybe': 2})

    return df

def encode_no_employee(df):
    '''
    This function takes in a pandas DataFrame, and encodes the 'no_employee' column, renames the column 'company_size', and returns the updated pandas DataFrame.
    '''
    # encodes the already binned 'no_employees' column
    df['no_employees'] = df['no_employees'].map({'1-5': 0, '6-25': 1, '26-100': 2, '100-500': 3, '500-1000': 4, 'More than 1000': 5})
    # renames the column to 'company_size'
    df = df.rename(columns=({'no_employees': 'company_size'}))
    
    return df

def encode_yes_no_columns(df):
    '''
    This function takes the columns defined in 'col_list' down below and encodes to '0' or '1'.
    '''
    #create a column list for for loop below:
    col_list= ['self_employed', 'family_history', 'treatment', 'remote_work', 'tech_company', 'obs_consequence']
    for col in col_list:
        df[col] = df[col].map({'No': 0, 'Yes': 1})
    return df

def encode_yes_no_dont_know(df):
    '''
    This function takes the columns defined in 'col_list' down below and encodes to '0', '1' or '2'.
    '''
    col_list= ['benefits', 'wellness_program', 'seek_help', 'anonymity', 'mental_vs_physical']
    for col in col_list:
        df[col] = df[col].map({'No': 0, 'Yes': 1, 'Don\'t know': 2})
    
    return df

####### Only Necessary if columns not already converted to lowercase #########

def convert_lower(df):
    '''
    This function takes in a pandas dataframe, converts all columns to lowercase, and returns the updated pandas DataFrame.
    '''
    # converts all column names to lowercase
    df.columns = df.columns.str.lower()
    
    return df

def target_correction(df):
    '''
    This function fills in the nulls for the work_interfere column with 'Not applicable'.
    '''
    df['work_interfere'] = df['work_interfere'].fillna(value= 'Not applicable')
    
    return df

def employer_status(df):
    '''
    This function fills in the nulls for the self_employed column with 'No'.
    '''
    df['self_employed'] = df['self_employed'].fillna(value = 'No')
    
    return df

