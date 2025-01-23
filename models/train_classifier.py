
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
#using this instead of GridSearchCV since there are SO MANY COMBINATIONS!!!
from sklearn.model_selection import RandomizedSearchCV 
import os 

#from models.tokenizer import tokenize 

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')



def load_data(database_filepath):
    '''INPUT - filepath of database created by process_data.py script
    OUTPUT - separates into X, Y, category_names
    '''
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
    '''INPUT
    X_train data, Y_train data
    OUTPUT a fitted machine learning pipeline. 
    '''
   
    #multi model pipelines with different ML models.  
    pipe_lr = Pipeline([('vect',CountVectorizer(tokenizer=tokenize,token_pattern=None)),
                        ('Tfidf',TfidfTransformer()),
                        #changed name from rf to lr. I started with a RandomForestClassifier, then later switched. pickle size was much larger 
                        ('lr',MultiOutputClassifier( LogisticRegression(max_iter=1000,random_state=42)))])
    
    #Note: with 4 variables here and a few values for each , there are 24 combinations.  
    # It will take awhile if we do all of them.
       
    # Parameter grid. adding a couple for the count vectorizer and the lr model. not going too crazy with a million combinations. 
    param_grid = {   
                  'vect__max_df': [0.5, 0.75],
                  'vect__min_df': [0.01, 0.05],
                  'lr__estimator__C': [0.01, 0.1, 1],
                  'lr__estimator__max_iter': [100, 200],
             
                 }
     
    lr_grid_search = GridSearchCV(pipe_lr, param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    lr_grid_search.fit(X_train, Y_train)
    return lr_grid_search



def evaluate_model(model, X_test, Y_test, category_names=None):
    '''INPUT
    the machine learning model we built earlier
    X_test data
    Y_test data
    the category names (pulled from the Y_test dataframe)
    
    OUTPUT 
    excel metric files on how well each category performed, including 
    -accuracy
    -precision
    -recall
    -the F1 score
    
    the output files are later used in some visuals in the Flask app
       
    '''

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
    '''
    INPUT 
    model name
    file path of where there model will be saved
    OUTPUT
    the model is saved to the file path given in the INPUT
    '''
    #save model
    joblib.dump(model, model_filepath)
    #save model filepath for 3rd script to avoid a hardcoded path
    with open('models/model_name', 'wb') as file:
        pickle.dump(model_filepath, file)
   


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('input received')
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
                
        print('Building model...feel free to grab a cup of coffee. This may take a bit')
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