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

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="whitegrid")
from collections import Counter
import re
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

smileys_array=[':-)', ':)', ';-)', ';)']

#print all the unique values in 
def print_col_unique_vals(df_col):
    tot=0
    for val in df_col.unique():
        print(val)
        tot=tot+1
    print(tot)
    
def print_all_cols(df):
    cols=df.columns.values
    for col in cols:
        print("****************"+col+"*******************")
        #print_col_unique_vals(df[col])
        print(df[col].value_counts())
        print("******************************************")
        
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

#Easy helper to search for all regular expressions in given text
def get_regexp_frequencies(regexp, text):
    return len(re.findall(regexp, text))

# Easy helper to get all the character counts
def get_char_count(df, text_col, search_char):
    return df[text_col].apply(lambda x: x.count(search_char))

def get_punctuation_count(df, text_col, search_chars=string.punctuation):
    return df[text_col].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

def get_stopwords_count(df, text_col, stopwords):
    return df[text_col].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))

#def get_hash
    
def generate_text_numerical_features(df, text_col, target,stopwords=STOP_WORDS, verbose=True):
    text_col_df=df[text_col]
    #create total characters features in text column per text row
    df['chars_count']        = text_col_df.apply(len)
    
    #create total numer of words features in text column per text row 
    df['words_count']          = text_col_df.apply(lambda x: len(x.split()))
    
    #create total numer of words features in text column per text row 
    df['CAPS_count']      = text_col_df.apply(lambda x: sum(1 for c in x if c.isupper()))
    
    #create total numer of exclmation marks in text column per text row 
    df['!_marks_count']  = get_char_count(df,text_col,'!')
    
    #create total numer of question marks in text column per text row 
    df['?_marks_count']  = get_char_count(df,text_col,'?')
    
    #create total number of hash marks in text column per text row 
    df['#_marks_count']  = text_col_df.apply(lambda x: sum([1 for word in x.split() if word[0] is '#']))
    #get_char_count(df,text_col,'#')
    
    #create total number of punctuations in text column per text row 
    df['punctuation_count'] = get_punctuation_count(df,text_col)
    
    df['stopwords_count'] = get_stopwords_count(df,text_col,stopwords)
    
    df['mean_word_len'] = text_col_df.apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    
    df['unique_words_count'] = text_col_df.apply(
    lambda comment: len(set(w for w in comment.split())))
    
    df['smilies_count'] = text_col_df.apply(lambda x: sum(x.count(w) for w in smileys_array))
    
    # Count number of \n
    df["\n_count"] = text_col_df.apply(lambda x: get_regexp_frequencies(r"\n", x))
    
    # Check for time stamp
    df["has_timestamp"] = text_col_df.apply(lambda x: get_regexp_frequencies(r"\d{2}|:\d{2}", x))
    
    # Check for http links
    df["has_http"] = text_col_df.apply(lambda x: get_regexp_frequencies(r"http[s]{0,1}://\S+", x))
    
    return df