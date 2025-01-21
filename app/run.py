import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request 

from plotly.graph_objs import Bar
import joblib
import pickle
from sqlalchemy import create_engine

import os 
import sys 
sys.path.append('../')
import re

from wrangle_metrics import return_metrics 

import plotly.graph_objs as go

def tokenize(text):
    '''removes non alphanumeric characters from text input parameter, then returns that cleaned text'''
    text = re.sub(r'[^a-zA-Z0-9\s]',' ',text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = []
    for tok in tokens:
        clean_tok = WordNetLemmatizer().lemmatize(tok,).lower().strip()
        cleaned_tokens.append(clean_tok)
        
    return cleaned_tokens

app = Flask(__name__)

# Open the file in binary mode
with open('data/db_name', 'rb') as file:
    # Deserialize and retrieve the variable from the file
    sql_db_path = pickle.load(file)

#what if someone uses a different database name in the process_data.py script??? 
engine = create_engine(f'sqlite:///{sql_db_path}')

try:
    df = pd.read_sql_table('cleaned_messages', engine)
except Exception as e:
    print(e)

# load model
#what if someone uses a different model name in the train_classifier.py script? 
# current_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(current_dir, '..', 'models', 'my_model.pkl')

# Open the file in binary mode
with open('models/model_name', 'rb') as file:
    # Deserialize and retrieve the variable from the file
    model_path = pickle.load(file)

model = joblib.load(f"{model_path}")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    '''
    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    figures = return_metrics()

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]
    #temp changed to a single image 
    #ids = figures    

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html',
                           ids=ids,
                           figuresJSON=figuresJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    '''
    

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
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

@app.route('/metrics')
def metrics():
    '''
    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
   
    figures = return_metrics()

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('metrics.html',
                           ids=ids,
                           figuresJSON=figuresJSON)


@app.route("/about")
def about():
	
	return render_template("about.html")


def main():

    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()


