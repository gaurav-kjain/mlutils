"""
This file implement dataframe related utils
Author: Gaurav Kumar Jain

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#print all the unique values of columns
def print_col_unique_vals(df_col):
    tot=0
    for val in df_col.unique():
        print(val)
        tot=tot+1
    print(tot)

#print dataframe column unique values 
def print_all_cols(df):
    cols=df.columns.values
    for col in cols:
        print("****************"+col+"*******************")
        #print_col_unique_vals(df[col])
        print(df[col].value_counts())
        print("******************************************")