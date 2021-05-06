import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy  as np

# Natrual language tool kit
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier   # Tried : Really slow in predict??

# f1 score
from sklearn.metrics import classification_report, precision_score, recall_score
# Model improvement
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """Load the database previously created
    returns
      X    : training values (text messages)
      Y    : categorical variable labels
      categ: Names of the categorical values
    """
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("frame1", engine)
    return df.message.values, df[df.columns[4:]], df.columns[4:]


def tokenize(text):
    """A functional composition of using
    - Regex removal of non character or numeric
    - tokenizing the series using nltk
    - Removing stop words
    - Tagging the words
    - Lemmatizing the words
    - Stemming the words
    """
    print("nltk config...")
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

    def clean(text):
        return re.sub(r"[^a-zA-Z0-9]", " ", text)

    def stopClean(textList):
        for word in textList:
            if word in stopwords.words("english"):
                textList.remove(word)
                return textList

    def whitespace_tokenize(text):
        return clean(text).lower().split()

    def sentence_tokinization(text):
        return sent_tokenize(text)

    def tokenize_base(text):
        return word_tokenize(text)

    def tokenize_full(text):
        return stopClean(tokenize_base(clean(text.lower())))

    def tag_and_tokenize(text):
        return pos_tag(tokenize_full(text))

    def stem_tag_tokenize(text):
        return pos_tag([PorterStemmer().stem(w) for w in tokenize_full(text)])

    def stem_lem_data(tokens):
        """Run stemming and lemmatization on the results returning
        a new list"""
        out = []
        for word, type_ in tokens:
            # First we lemmatize the work
            w = WordNetLemmatizer().lemmatize(word)
            # Then we stem the word
            w = PorterStemmer().stem(w)
            out.append((w, type_))
        return out

    return stem_lem_data(stopClean(tag_and_tokenize(clean(text))))


def build_model():
    """
    Build the ML model using the pipeline methodology
    """
    pipeline = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("class", MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Run evaluation metrics on the model
    """
    #TODO update this function
    pred = model.predict(X_test)
    for indx, column in enumerate(category_names):
        print(column,
              classification_report(Y_test[column].values,
                                    pred[:, indx]))


def save_model(model, model_filepath):
    """Save the model"""
    with open(model_filepath, "wb") as fout:
        pickle.dump(model, fout)


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
