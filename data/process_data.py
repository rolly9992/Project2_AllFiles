import sys
import pandas as pd
import numpy as np 
from sqlalchemy import create_engine
import os 
import pickle

#TODO add comments to each definition, using the STANDARD PRACTICE
#Note to self: NO AD HOC. NO DEVIATION FROM STANDARD PRACTICE. NO PRAGMATIC CODING.


message_filepath = os.path.join('data', 'messages.csv')
category_filepath = os.path.join('data', 'categories.csv')


def load_data(messages_filepath, categories_filepath):
    ''' 
    INPUT 
    message csv file
    categories csv file
    
    OUTPUT 
    combined dataframe of the message and categories files
    '''
    #load messages dataset
    messages = pd.read_csv(message_filepath,sep=',')
    print('messages data loaded')

    # load categories dataset
    categories = pd.read_csv(category_filepath,sep=',')
    print('categories data loaded')
    # merge datasets
    df = messages.merge(categories,on='id',how='inner')
    return df



def clean_data(df):
    '''
    INPUT
    The merged dataframe we created in the previous definition 
    
    OUTPUT 
    Cleaned dataframe 
    
    What this does -- splits the categories column into separate y variables. 
    '''

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [x[:-2] for x in row]

    # rename the columns of `categories`
    categories.columns = category_colnames



    # ### 4. Convert category values to just numbers 0 or 1.
    # - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, `related-0` becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.
    # - You can perform [normal string actions on Pandas Series](https://pandas.pydata.org/pandas-docs/stable/text.html#indexing-with-str), like indexing, by including `.str` after the Series. You may need to first convert the Series to be of type string, which you can do with `astype(str)`.

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
    # ### 5. Replace `categories` column in `df` with new category columns.
    # - Drop the categories column from the df dataframe since it is no longer needed.
    # - Concatenate df and categories data frames.

    df = df.drop('categories',axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # ### 6. Remove duplicates.
    df = df.drop_duplicates()
    return df

    

def save_data(df, database_filename):
    '''INPUT 
    dataframe that we just cleaned
    name of the database to save the cleaned data to 
    OUTPUT 
    a sqlite database to be later used in machine learning
    '''
   
    engine = create_engine(f'sqlite:///{database_filename}')
    try:
        df.to_sql('cleaned_messages', engine, index=False)
        serialized = pickle.dumps(database_filename) 
        with open('data/db_name', 'wb') as file:
                 # store name that user chooses to be able to use in 3rd script
                pickle.dump(database_filename, file)
 
    except Exception as e:
        print('exception of :',e)
      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()