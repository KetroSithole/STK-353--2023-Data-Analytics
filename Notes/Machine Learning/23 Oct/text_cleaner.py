import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd

def text_cleaner(txt):
    txt = pd.Series(txt)
# Step 1:  Convert the text to lowercase
    txt_lo = txt.str.lower()
# Step 2: Removing punctuation and numbers
    def rm_punctuation(row):
        return row.translate(row.maketrans('', '', string.punctuation + string.digits))  # string.digits involve the numbers as well
    txt_wo_punc = txt_lo.apply(lambda x: rm_punctuation(x))
# Step 3: Removing stopwords
    from nltk.corpus import stopwords
    x = stopwords.words('english')
    STOPWORDS = set(x)
    def rm_stopwords(text):
        return " ".join([word for word in text.split() if word not in STOPWORDS])
    txt_wo_stwd = txt_wo_punc.apply(lambda x: rm_stopwords(x))
# Step 4: Stemming or lemmatization
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    def stemfunc(row):
        token = word_tokenize(row)
        pst = PorterStemmer()
        return " ".join([pst.stem(x) for x in token])
    txt_cleaned = txt_wo_stwd.apply(lambda x: stemfunc(x))
    return txt_cleaned.values[0]