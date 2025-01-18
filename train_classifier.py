#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y



# import libraries
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
from models.tokenizer import tokenize 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# load data from database
engine = create_engine('sqlite:///data/disaster_project.db')

sql = 'select * from cleaned_messages'
#table is cleaned_messages
df = pd.read_sql_query(sql,engine)
y = df.drop('message',axis=1) 
y = y.drop('id',axis=1) 
y = y.drop('original',axis=1) 
y = y.drop('genre',axis=1)
y = y.drop('child_alone',axis=1) #no values are populated in the data
y = y.astype('int')
X = df['message'] 


# ### 2. Write a tokenization function to process your text data



# def tokenize(text):
#     '''removes non alphanumeric characters from text input parameter, then returns that cleaned text'''
#     text = re.sub(r'[^a-zA-Z0-9\s]',' ',text)
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()
#     cleaned_tokens = []
#     for tok in tokens:
#         clean_tok = WordNetLemmatizer().lemmatize(tok,).lower().strip()
#         cleaned_tokens.append(clean_tok)
        
#     return cleaned_tokens
        
 




# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.


#multi model pipelines with different ML models.  
pipe_lr = Pipeline([('vect',CountVectorizer(tokenizer=tokenize,token_pattern=None)),
                    ('Tfidf',TfidfTransformer()),
                    ('rf',MultiOutputClassifier( LogisticRegression(max_iter=1000,random_state=42)))])


pipelist = [pipe_lr]


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


def fit_your_pipeline(modelname,x_train,y_train):
    '''a definition to insert in a loop to fit multiple models (if added)
    inputs: 
    the name of the model
    the X_train data
    the y_train_data
    output the fitted model. 
    there is no need for a return statement, but it prints out when done. 
    NOTE: this may take some time. particularly if you train more than 1 model'''
    try:
        modelname.fit(x_train, y_train)
        #print(f'fitted model: {modelname}') 
    except Exception as e:
        print(f'exception for model: {modelname} of {e}')  



#for i in range(len(pipelist)):
print('fitting pipeline...')
fit_your_pipeline(pipelist[0],X_train,y_train)
print('fitting completed')
pipelist = [pipe_lr]


#save model to pickle file
model_file_path = os.path.join('models', 'my_model.pkl')

joblib.dump(pipelist[0], model_file_path)
# for i in range(len(pipelist)):
#     pipe = pipelist[i]
#     temp2 = pipe.steps[-1][1].estimator.__class__.__name__
#     print(temp2)
#     #model_name = type(temp[2]).__name__
#     #print(model_name)
#     #name = str(pipe)
#     #print(pipelist[i])
#     joblib.dump(pipe, f'{str(temp2)}.pkl')
# #joblib.dump(pipe, 'pipe_lr.pkl')

y_pred = pipe_lr.predict(X_test)

ycols = y_test.columns.tolist()
y_pred2 = pd.DataFrame(y_pred,columns=ycols)

l = []
for i in range(len(ycols)):
    
    accuracy=accuracy_score(y_test[ycols[i]],y_pred2[ycols[i]])
    precision=precision_score(y_test[ycols[i]],y_pred2[ycols[i]],average='weighted',zero_division=1)
    recall=recall_score(y_test[ycols[i]],y_pred2[ycols[i]],average='weighted')
    f1score = f1_score(y_test[ycols[i]],y_pred2[ycols[i]],average='weighted')    
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
metrics_file_path = os.path.join('data', 'metrics_file.xlsx')
dfmetrics.to_excel(metrics_file_path)
dfmetrics


#looking at average accuracy, precision, recalls since over 30 output variables. 
summary = dfmetrics.describe()
metrics_summary_file_path = os.path.join('data', 'metrics_summary_file.xlsx')
summary.to_excel(metrics_summary_file_path)






