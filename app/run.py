"""
.. module:: run
   :platform: Unix, Windows
   :synopsis: Runs flask app that hosts webpage to display plots and
              classify new messages.

.. moduleauthor:: Matthias Budde
"""
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('CategorizedMessages', engine)
# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """Index function.

    Generates index page with plotly visualizations.
    """

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    category_counts = df.iloc[:, 4:].sum()

    # create visuals
    graphs = [
        {
            'data': [
                Scatter(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=category_counts.index.values,
                    values=category_counts.values)
            ],

            'layout': {
                'title': 'Distribution of Message Labels',
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = [f"graph-{i}" for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """Function that gets user input, passes it to classifier and renders
       classification results.
    """
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """Main function.
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
