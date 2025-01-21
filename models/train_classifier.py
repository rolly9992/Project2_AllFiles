

# import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
from nltk import word_tokenize
from nltk import sent_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import pickle 
import joblib
import re 
import nltk
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import os 

#from models.tokenizer import tokenize 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')



def load_data(database_filepath):
    
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')

    sql = 'select * from cleaned_messages'
    #table is cleaned_messages
    df = pd.read_sql_query(sql,engine)
    
    Y = df.drop('message',axis=1) 
    Y = Y.drop('id',axis=1) 
    Y = Y.drop('original',axis=1) 
    Y = Y.drop('genre',axis=1)
    Y = Y.drop('child_alone',axis=1) #no values are populated in the data
    Y = Y.astype('int')
    X = df['message'] 
    category_names = list(Y.columns.tolist())  
    return X, Y, category_names

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
   
################

def build_model(X_train,Y_train):
   
    #multi model pipelines with different ML models.  
    pipe_lr = Pipeline([('vect',CountVectorizer(tokenizer=tokenize,token_pattern=None)),
                        ('Tfidf',TfidfTransformer()),
                        ('rf',MultiOutputClassifier( LogisticRegression(max_iter=1000,random_state=42)))])

    pipe_lr.fit(X_train, Y_train)
 
    return pipe_lr



def evaluate_model(model, X_test, Y_test, category_names=None):

    y_pred = model.predict(X_test)
    ycols = Y_test.columns.tolist()
    #category_names =ycols
    y_pred2 = pd.DataFrame(y_pred,columns=ycols)
    l = []
    for i in range(len(ycols)):
        
        accuracy=accuracy_score(Y_test[ycols[i]],y_pred2[ycols[i]])
        precision=precision_score(Y_test[ycols[i]],y_pred2[ycols[i]],average='weighted',zero_division=1)
        recall=recall_score(Y_test[ycols[i]],y_pred2[ycols[i]],average='weighted')
        f1score = f1_score(Y_test[ycols[i]],y_pred2[ycols[i]],average='weighted')    
        temp ={'Column':[ycols[i]]
                        ,'Accuracy':[accuracy]
                        ,'Precision':[precision],
                        'Recall':[recall],
                        'F1 Score':[f1score]}
        
        temp = pd.DataFrame(temp)
        #temp = {k:[v] for k,v in temp.items()}  
        l.append(temp)

    
    dfmetrics=pd.concat(l,axis=0)
    
    metrics_file_path = os.path.join('data', 'metrics_file.xlsx')
    dfmetrics.to_excel(metrics_file_path)
    
    #looking at average accuracy, precision, recalls since over 30 output variables. 
    summary = dfmetrics.describe()
    metrics_summary_file_path = os.path.join('data', 'metrics_summary_file.xlsx')
    summary.to_excel(metrics_summary_file_path)
    


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)
    with open('models/model_name', 'wb') as file:
        # save name of model so it can get picked up by the 3rd script that has no sys args. 
        pickle.dump(model_filepath, file)
   


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('input received')
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
                
        print('Building model...')
        model = build_model(X_train,Y_train)
        
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