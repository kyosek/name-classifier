import argparse
import logging
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from modules.transform.preprocess import cleanNames
from modules.train.train import trainNaiveBayes, saveModel

logging.basicConfig(
    format="%(asctime)s --- %(levelname)s --- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main(train_required=True):
    """main function

    Args:
        train_required (bool, optional): 
            if True, it will start the training
            process.
            if False, it will load the trained model
            from "resources/models/bn_clf.pickle.
            Defaults to True.
            
    Process:
        1. Load the data
        2. Preprocess the text
        3. Split the data into train and test
        4. if train_required == 'False'
            Load the trained model
        4. if train_required == 'True'
            Train and save the model
        5. Make predictions
        6. Evaluate the model
        7. Complete the process
    """
    
    
    logging.info("Loading data")
    df = pd.read_csv("resources/data/data.csv")
    
    logging.info("Preprocess the names")
    df['cleaned_name'] = df['Name'].apply(lambda x: cleanNames(x))
    
    logging.info("Splitting the dataset")
    X_train, X_test, y_train, y_test = train_test_split(df["cleaned_name"], df["Class"], test_size=.3, random_state=42)
    
    if train_required == "False":
        logging.info("Loading the model")
        clf = pickle.load(open("resources/models/nb_model.pkl","rb"))
    
    elif train_required == "True":
        logging.info("Start training the model")
        clf = trainNaiveBayes(X_train, y_train)
        saveModel(clf)
        
    else:
        logging.info("Please enter the argument 'train_required' either True or False")
        exit()
        
    logging.info("Making predictions")
    predictions = clf.predict(X_test)
    
    logging.info("Evaluating the model performance")
    logging.info("Accuracy on the test set is " +
                 str(round(accuracy_score(y_test, predictions),4)))
    
    logging.info("Process completed successfully")
    
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser("StreetBees - Name classification")

    parser.add_argument("train_required", type=str, help="wether want to train")

    args = parser.parse_args()

    main(args.train_required)
