
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np 
from nltk import word_tokenize
from nltk import sent_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import pickle 
import re 
import nltk
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV 


database_filepath = sys.argv[1]
def load_data(database_filepath):
    # load data from database
    engine = create_engine(database_filepath) 
    #'sqlite:///disaster_project.db')
    
    sql = 'select * from cleaned_messages'
    #table is cleaned_messages
    df = pd.read_sql_query(sql,engine)
    y = df.drop('message',axis=1) 
    y = y.drop('id',axis=1) 
    y = y.drop('original',axis=1) 
    y = y.drop('genre',axis=1)
    y = y.astype('int')
    
    #replacing 2s with 0s for related field. 
    y['related']=y['related'].replace(2,0)
    category_names = y.columns.tolist()
    X = df['message']
    return X, y, category_names
    


def tokenize(text):
    text = re.sub(r'[^a-zA-Z0-9\s]',' ',text)
    
    tokens = word_tokenize(text)

    cleaned_tokens = []
    for tok in tokens:
        clean_tok = WordNetLemmatizer().lemmatize(tok,).lower().strip()
        cleaned_tokens.append(clean_tok)
        
    return cleaned_tokens


def build_model():
    X,y, category_names= load_data(database_filepath)
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize,token_pattern=None)),
                    ('Tfidf',TfidfTransformer()),
                    ('rf',MultiOutputClassifier(RandomForestClassifier()))])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    pipeline.fit(X_train,y_train)
    model = pipeline
    return model
    


def evaluate_model(model, X_test, Y_test, category_names):
    X,y, category_names= load_data(database_filepath)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)    
    model = build_model()
  
    y_pred = model.predict(X_test)
    ycols = y_test.columns.tolist()
    df_y_pred = pd.DataFrame(y_pred,columns=ycols)
    l = []
    for i in range(len(ycols)):
        
        accuracy=accuracy_score(y_test[ycols[i]],df_y_pred [ycols[i]])
        precision=precision_score(y_test[ycols[i]],df_y_pred [ycols[i]],average='weighted',zero_division=1)
        recall=recall_score(y_test[ycols[i]],df_y_pred [ycols[i]],average='weighted')
        f1score = f1_score(y_test[ycols[i]],df_y_pred [ycols[i]],average='weighted')    
        temp ={'Column':[ycols[i]]
                         ,'Accuracy':[accuracy]
                         ,'Precision':[precision],
                         'Recall':[recall],
                          'F1 Score':[f1score]}
        
        temp = pd.DataFrame(temp)
        #temp = {k:[v] for k,v in temp.items()}  
        l.append(temp)
    #print(len(l))
    dfmetrics=pd.concat(l,axis=0)
    dfmetrics
    summary = dfmetrics.describe()
    dfmetrics.to_excel('metrics_file.xlsx')
    print('metrics file saved')
    summary.to_excel('metrics_summary_file.xlsx')
    print('metrics summary file saved')
    return dfmetrics, summary
    
    
    
  


def save_model(model, model_filepath):
    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)
    


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