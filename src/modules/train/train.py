import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def trainNaiveBayes(X_train: 'pd.DataFrame', 
                    y_train: 'pd.DataFrame') -> "Callable":
    """Train Naïve bayers model

    Args:
        X_train (pd.DataFrame): training input features
        y_train (pd.DataFrame): training labels
        
    Process:
        1. Create a training pipeline with TF-IDF transformer
        2. Train the model
        
    Return:
        Model callable object
    """
    
    nb_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB(alpha=1)),])

    nb_clf.fit(X_train, y_train)
    
    return nb_clf

def saveModel(model) -> None:
    """Save the trained model

    Args:
        model (Model object): Model object
    """
    
    pickle.dump(model, open("resources/models/nb_model.pkl", "wb"))
