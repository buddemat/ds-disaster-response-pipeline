"""
.. module:: train_classifier
   :platform: Unix, Windows
   :synopsis: A module to train a classifier to categorize message text and
              store it in a Pickle file.

.. moduleauthor:: Matthias Budde


"""
import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """Loads message and category data from database.

    :param str database_filepath: The path of the sqlite db file
    :return: loaded messages and categories data, category labels
    :rtype: pd.DataFrame
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df_categorized_messages = pd.read_sql_table('CategorizedMessages', engine)
    X = df_categorized_messages['message']
    Y = df_categorized_messages.iloc[:,4:]
    return X, Y, Y.columns


def tokenize(text):
    """Tokenizer function.

    :param text text to tokenize
    :return: cleaned tokens
    :rtype:
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Builds model pipeline.

    :return: trained model
    :rtype: sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
                RandomForestClassifier(random_state=42,
                                       n_estimators=200,
                                       min_samples_split=3
                )
            )
        )
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates model pipeline.

    :param model sklearn.pipeline.Pipeline trained model pipeline
    :param X_test pd.Dataframe test split part of training data
    :param Y_test pd.Dataframe test split part of labels
    :param category_names list list of category names
    :return: None
    """
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=Y_test.columns, index=Y_test.index)

    for col in category_names:
        rep = classification_report(y_pred=Y_pred[col], y_true=Y_test[col])
        print(f'======= {col} =======')
        print(rep)


def save_model(model, model_filepath):
    """Saves model as pickle.

    :param model sklearn.pipeline.Pipeline model pipeline
    :param model_filepath str path to save the pickle to
    :return: None
    """
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)


def main():
    """Main function.
    """
    if len(sys.argv) == 3:
        database_filepath = sys.argv[1]
        model_filepath = sys.argv[2]
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
