import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''
        Function to clean the input string by: normalizing, tokenizing, 
        removing stopwords, lemmatizing for nouns and for verbs, and stemming.
        Output is a list of cleaned words.
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Lemmatize for nouns and for verbs, and stem
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    lemmed_stemmed = [PorterStemmer().stem(w) for w in lemmed]
    return lemmed_stemmed

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)
df_plot = pd.read_sql_table('words', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create 2nd chart
    y_cats = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products',\
           'search_and_rescue', 'security', 'military', 'child_alone', 'water',\
           'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees',\
           'death', 'other_aid', 'infrastructure_related', 'transport',\
           'buildings', 'electricity', 'tools', 'hospitals', 'shops',\
           'aid_centers', 'other_infrastructure', 'weather_related', 'floods',\
           'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']
    
    list_words = list(pd.unique(df_plot["word"]))
    word_count_list = []
    for cat in y_cats:
        df_plot[df_plot["category"] == cat][["word", "count"]]
        df_temp = pd.DataFrame(list_words, columns = ["word"])
        df_temp = df_temp.merge(df_plot[df_plot["category"] == cat][["word", "count"]], how = "left")
        word_count_list.append(list(df_temp["count"]))

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
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
        },
        {
            'data': [
                Heatmap(
                    x=list_words,
                    y=y_cats,
                    z=word_count_list
                )
            ],

            'layout': {
                'title': 'Top 10 cleaned words per category and their count',
                'yaxis': {
					'title': "Categories"
                },
				'xaxis': {
					'title': "Cleanned, stemmed words"
                },
				'height': 800
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()