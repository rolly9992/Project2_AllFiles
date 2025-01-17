
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify, flash
from plotly.graph_objs import Bar
import joblib
import pickle
from sqlalchemy import create_engine
import os 



from wrangle_metrics import return_metrics

app = Flask(__name__)

def tokenize(text):
    '''
    

    Parameters
    ----------
    text : TYPE
        DESCRIPTION.

    Returns
    -------
    clean_tokens : TYPE
        DESCRIPTION.

    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
database_path = r'disaster_project.db'
engine = create_engine(f'sqlite:///{database_path}')
df = pd.read_sql_table('cleaned_messages', engine)

# load model
model_path ='LogisticRegression.pkl'
model = joblib.load(f"{model_path}")
#with open(model_path, "rb") as f:
#    model = pickle.load(f)

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

    return render_template('metrics.html',
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
