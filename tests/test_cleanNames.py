import pandas as pd
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob, Word

from src.modules.transform.preprocess import cleanNames

nltk.download("stopwords")
from nltk.tokenize import word_tokenize


def test_clean_names():
    """test_clean_names function
    
        Input name: "E. D. Abbott Ltd"
        
        Expected return: "abbott ltd"
    """
    
    df = pd.read_csv("resources/data.csv")
    
    df['cleaned_name'] = df['Name'].apply(lambda x: cleanNames(x))
    
    assert df["cleaned_name"][0] == "abbott ltd"