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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # #  This section defines some helper functions # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def clean(text):
    return re.sub(r"[^a-zA-Z0-9]", " ", text)


def stopClean(textList):
    """Remove stopwords"""
    for word in textList:
        if word in stopwords.words("english"):
            textList.remove(word)
    return textList


def tokenize_full(text):
    return stopClean(word_tokenize(clean(text.lower())))


def tag_and_tokenize(text):
    return pos_tag(tokenize_full(text))


def stem_tag_tokenize(text):
    return pos_tag([PorterStemmer().stem(w) for w in tokenize_full(text)])


def lem_data(tokens):
    """Run lemmatization on the results returning
    a new list"""
    out = []
    for word, type_ in tokens:
        # First we lemmatize the work
        w = WordNetLemmatizer().lemmatize(word)
        out.append((w, type_))
    return out
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def tokenize(text):
    """A functional composition of using
    - Change urls to "urlplaceholder"
    - Sentence to lowercase
    - Regex removal of non character or numeric
    - tokenizing the series using nltk
    - Removing stop words
    - Tagging the words
    - Lemmatizing the words
    - Stemming the words
    """
    url_regex ='http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    return lem_data(stopClean(tag_and_tokenize(clean(text))))


def build_model():
    """
    Build the ML model using the pipeline methodology
    """
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("class", MultiOutputClassifier(KNeighborsClassifier(
            n_neighbors=36, algorithm="brute"
        )))
    ])

    # Select the optimized parameters
    parameters = {
        'class__estimator__n_neighbors':[30, 36, 42]
    }
    cs = GridSearchCV(pipeline, param_grid=parameters)

    return cs


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Run evaluation metrics on the model
    """
    def display_results(cv, y_test, y_pred):
        labels        = np.unique(y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
        accuracy      = (y_pred == y_test).mean()

        print("Labels:", labels)
        print("Confusion Matrix:\n", confusion_mat)
        print("Accuracy:", accuracy)
        print(classification_report(y_test, y_pred))
        print("\nBest Parameters:", cv.best_params_)

    y_pred = model.predict(X_test)
    for indx, label in enumerate(Y_test.columns):
        display_results(model, Y_test[label], y_pred[:, indx])


def save_model(model, model_filepath):
    """Save the model"""
    with open(model_filepath, "wb") as fout:
        pickle.dump(model, fout)


def ensure_nltk():
    """Ensure the packages used by the nltk are installed"""
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)


        print('Building model...')
        model = build_model()
        
        print('Training model...')
        ensure_nltk()
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
