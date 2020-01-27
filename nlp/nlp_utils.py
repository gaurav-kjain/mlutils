"""
This file implement NLP related utils
Author: Gaurav Kumar Jain

"""

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="whitegrid")
from collections import Counter
import re

def get_words_count_from_col(df, col_name, split_rule=" "):
    new_col_name=col_name+'_word_count'
    df[new_col_name]=df[col_name].apply(lambda x:len(str(x).split(split_rule)))
    return df  

def print_text_df_statistical_features(df, text_col, target):
    #Average count of phrases per sentence in train/target
    #avg_count=df.groupby(target)[text_col].count().mean()
    #print("Average count of phrases per sentence in train/target is {0:.0f}".format(avg_count))
    print('Number of Total Data Samples: {}. Number of Target Classes: {}.'.format(df.shape[0], len(df[target].unique())))
    print('\nData Distribution vs Classes:',Counter(df[target]))
    print('\nAverage word length of phrases in training dataset is {0:.0f}.'.format(np.mean(df[text_col].apply(lambda x: len(x.split())))))


def get_regexp_frequencies(regexp="", text=None):
    return len(re.findall(regexp, text))
    
    
def generate_text_numerical_features(df, text_col, target, verbose=True):
    text_col_df=df[text_col]
    #create total characters features in text column
    df['total_chars_count']        = text_col_df.apply(len)
    
    #create total numer of words features in text column 
    df['num_words_count']          = text_col_df.apply(lambda x: len(x.split()))
    
    #create total numer of words features in text column 
    df['capital_chars_count']      = text_col_df.apply(lambda x: sum(1 for c in x if c.isupper()))
    
    #create total numer of exclmation marks in text column 
    df['exclamation_marks_count']  = text_col_df.apply(lambda x: x.count('!'))
    
    return df