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
from nltk import pos_tag
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

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
    if(target is not None):
        print('Number of Total Data Samples: {}. Number of Target Classes: {}.'.format(df.shape[0], len(df[target].unique())))
        print('\nData Distribution vs Classes:',Counter(df[target]))
    print('\nAverage word length of phrases in training dataset is {0:.0f}.'.format(np.mean(df[text_col].apply(lambda x: len(x.split())))))

#Easy helper to search for all regular expressions in given text
def get_regexp_frequencies(regexp, text):
    return len(re.findall(regexp, text))

#Easy helper to search and replace all regular expressions in given text
def replace_regexp(regexp, text, replacement=' '):
    return re.sub(regexp, replacement, text)

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
    df['#_tags_count']  = text_col_df.apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
    #get_char_count(df,text_col,'#')
    
    #create total number of punctuations in text column per text row 
    df['punctuation_count'] = text_col_df.apply(lambda x: get_regexp_frequencies("[^\w\s]", x))
    
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
    
    #Number of digits
    df['digit_count'] = df[text_col].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
    
    return df


def tag_part_of_speech(text):
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    pos_list = pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    adjective_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    verb_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    return[noun_count, adjective_count, verb_count]

def topNFrequentWords(df, text_col, N=10):
    return pd.Series(' '.join(df[text_col]).split()).value_counts()[:N]
    

def printTopNFrequentWords(df, text_col, N=10):
    print(topNFrequentWords(df, text_col, N=N))
    
def drawWordCloud(df, text_col, clean_stopwords=True):
    
    textstr=''.join(x for x in df[text_col])
    
    if(clean_stopwords==True):
        stopwords=set(STOPWORDS)
        wordcloud = WordCloud(background_color="black").generate(textstr)
    else:
        wordcloud = WordCloud(background_color="black").generate(textstr)
    
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
def clean_urls(df, text_col, new_text_col):
    url_regex='http[s]?://\S+'
    #url_regex=r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
    rows=[]
    for row in df[text_col]:
        rows.append(re.sub(url_regex, '', row))
    df[new_text_col]= rows
    return df

def clean_digits(df, text_col, new_text_col):
    df[new_text_col] = df[new_text_col].str.replace('\d+', '')
    #df[new_text_col]=df[new_text_col].apply((filter(lambda c: not c.isdigit(), word)) for word in text_splited )
    return df

def clean_stopwords(df, text_col, stopwords, new_text_col):
    df[new_text_col]=df[text_col].apply(lambda x: ' '.join(word for word in x.split() if word not in stopwords) )
    return df

#https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
def preprocess(ReviewText, extraPatsToRemove=[]):
    ReviewText = ReviewText.str.replace("(<br/>)", " ")
    ReviewText = ReviewText.str.replace('(<a).*(>).*(</a>)', ' ')
    ReviewText = ReviewText.str.replace('(&amp)', ' ')
    ReviewText = ReviewText.str.replace('(&gt)', ' ')
    ReviewText = ReviewText.str.replace('(&lt)', ' ')
    ReviewText = ReviewText.str.replace('(\xa0)', ' ')  
    ReviewText = ReviewText.str.replace(';', '')
    #for pattern in extraPatsToRemove:
     #   print(pattern)
      #  ReviewText = ReviewText.str.replace(str(pattern), ' ')
    return ReviewText
    
def process_dataframe_summarize_analysis(df, text_col, target_col, stopwords, extraPatsToRemove=[' '], N=10):
    new_text_col='clean_text'
    print("************START SUMMARIZATION\n\n")
    print_text_df_statistical_features(df=df, text_col=text_col, target=target_col)
    df=generate_text_numerical_features(df=df, text_col=text_col, target=target_col)
    print("\n\n")
    print("******Data head***********************")
    print(df.head(1).T)
    print("\n\n")
    print("************Top 10 Frequent words***********************")
    printTopNFrequentWords(df, text_col, N=10)
    print("\n\n")
    drawWordCloud(df, text_col)
    print("\n\n")
    print("************RAW WORD CLOUD as in DATASET*****************")
    df=clean_urls(df, text_col, new_text_col)
    print("\n\n")
    print("************WORD CLOUD after URLS Cleaning*****************")
    drawWordCloud(df, new_text_col)
    print("\n\n")
    print("************Cleaning Stopwords*****************")
    df=clean_stopwords(df, new_text_col, stopwords=stopwords,new_text_col=new_text_col)
    print("\n\n")
    print("************WORD CLOUD after STOPWORDS Cleaning*****************")
    drawWordCloud(df, new_text_col)
    print("\n\n")
    print("************Top 10 Frequent words***********************")
    printTopNFrequentWords(df, new_text_col, N=10)
    print("\n\n")
    print("************PRE PROCESSING*****************")
    df[new_text_col]=preprocess(df[new_text_col],extraPatsToRemove=extraPatsToRemove)
    print("************WORD CLOUD after PREPROCESSING*****************")
    drawWordCloud(df, new_text_col)
    print("\n\n")
    print("************Top 10 Frequent words***********************")
    printTopNFrequentWords(df, new_text_col, N=10)
    print("\n\n")
    
    print("************REMOVE DIGITS*****************")
    df=clean_digits(df, text_col, new_text_col)
    print("************WORD CLOUD after REMOVING DIGITS*****************")
    drawWordCloud(df, new_text_col)
    print("\n\n")
    print("************Top 10 Frequent words***********************")
    printTopNFrequentWords(df, new_text_col, N=10)
    print("\n\n")