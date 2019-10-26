"""
     Python script that creates a machine learning model to categorize input
     messages into one of the 36 categories.
     The script takes as input the database file path and model file path, then
     creates and trains a pipeline classifier based on RandomForest, and stores
     the classifier into a pickle file to the specified model file path.

    Args:
        arg1 (str): (filepath) to the sqlite database where the input dataset are stored.
        arg2 (str): (filepath) to the output pickle file where the model will be stored.

"""

import sys
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from custom_scorer_module import custom_scoring_function
from sklearn.externals import joblib

def load_data(database_filepath):
    '''
        Function to load messages from database.
        
        Returns
        -------
        X: array with 1 column containing input messages
        Y: array with 36 columns specifying to which categories the messages  pertain
        category_names: list of 36 output categories names
        
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', con = engine)
    category_names = list(map(str, np.setdiff1d(df.columns.values, ['id', 'message', 'original', 'genre'])))

    return df["message"].values, df[category_names].values, category_names


def tokenize(text):
    '''
        Function to clean the input string by: normalizing, tokenizing, 
        removing stopwords, lemmatizing for nouns and for verbs, and stemming.
                
        Returns
        -------
        lemmed_stemmed: List of cleaned words.
        
        
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Lemmatize for nouns and for verbs, and stem
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    lemmed_stemmed = [PorterStemmer().stem(w) for w in lemmed]

    return lemmed_stemmed


def build_model():
    '''
        Function builds a machine learning pipeline for the input messages by applying
        vectorize, TF-IDF, and RandomForest classifier with multiple output categories.

        The evaluation of the clasifier is done using a custom scoring function.
        The scoring function is Fraction Correct or Accuracy ratio, which measures
        the fraction of all instances that are correctly categorized: (TP + TN)/
        Total Population.
        The alternative scoring function F-score (combining precision and recall),
        was not used as it does not take the true negative rate into account.
        In a real-life application different weights could be applied to the
        categorization results (TP + TN + FP + FN), leading to a case-specific
        scoring function.
    
        Returns
        -------
        GridSearchCV model

    '''

    # to save processing time, some parameters were optimized individually
    # (by running GridSearchCV for only one parameter). This does not guarantee
    # finding the best global parameters, but should be close.    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df = 0.75, max_features = 10000)),
        ('tfidf', TfidfTransformer(use_idf = True)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(max_depth = None, min_samples_split = 4), n_jobs = -1))
    ])

    parameters = {
#         'vect__max_df': (0.75, 1.0), -> mean_test_score': [ 0.94412797,  0.94390059]
#         'vect__max_features': (None, 10000, 15000), -> 'mean_test_score': [ 0.94408409,  0.94470906,  0.94429951]
#         'tfidf__use_idf': (True, False), -> 'mean_test_score': [ 0.94439923,  0.94435136]
         'clf__estimator__n_estimators': [10, 100] # -> 'mean_test_score': [ 0.9462728 ,  0.94922478]
#         'clf__estimator__max_depth': [None, 4], -> mean_test_score': [ 0.94413462,  0.92674326]
#         'clf__estimator__min_samples_split': [2, 4] -> mean_test_score': [ 0.94427557,  0.94563986]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=make_scorer(custom_scoring_function),\
                      n_jobs = -1, cv = 3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
        Function evaluates the model on the test set and prints the results.

        The evaluation of the clasifier is done using a custom scoring function.
        The scoring function is Fraction Correct or Accuracy ratio, which measures
        the fraction of all instances that are correctly categorized: (TP + TN)/
        Total Population.
        The alternative scoring function F-score (combining precision and recall),
        was not used as it does not take the true negative rate into account.

        Returns
        -------
        None.

    '''
    
    # print model results
    print("GridSearchCV results:\n")
    print(model.cv_results_)
    print("\n")
    
    # calculate accuracy per category and overall one
    Y_pred = model.predict(X_test)
    confusion_mat = multilabel_confusion_matrix(Y_test, Y_pred)
    accuracy_per_cat = pd.DataFrame(index = category_names)
    accuracy_per_cat["accuracy"] = 0
    for i, cat in enumerate(category_names):
        accuracy_per_cat.loc[cat,"accuracy"] = np.trace(confusion_mat[i,:,:]) / confusion_mat[i,:,:].sum()
    print_confusion_mat = confusion_mat.sum(axis = 0) / confusion_mat.sum()
    
    # print accuracy per category and overall one
    print("Accuracy per category:\n")
    print(accuracy_per_cat)
    np.set_printoptions(formatter={'all':lambda x: '{0:5.1f}%'.format(x*100)})
    print("Overall Confusion Matrix in %:\n(cols and rows = 0,1; C{i, j}: true i, predicted j)\n")
    print(print_confusion_mat)
    np.set_printoptions(formatter=None)
    
    pass


def save_model(model, model_filepath):
    '''
        Function saves the model into the specified filepath.
    '''
    
    joblib.dump(model, model_filepath)

    pass


def main():
    if len(sys.argv) == 3:
        
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()