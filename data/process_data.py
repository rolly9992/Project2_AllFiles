
import sys
import pandas as pd
from sqlalchemy import create_engine




file1 = r'C:\Users\Milo_Sand\Documents\GitHub\Project2\data\categories.csv' 
#file1
#sys.argv[2] =r'C:\Users\Milo_Sand\Documents\GitHub\Project2\data\messages.csv' 
#sys.argv[3] = r'sqlite:///disaster_project.db'

print(sys.argv)
#sys.exit()

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath,sep=',')
    categories = pd.read_csv( categories_filepath,sep=',')
    # merge datasets
    df = messages.merge(categories,on='id',how='inner')
  
    return df
    


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = [x[:-2] for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
    # drop the original categories column from `df`
    df = df.drop('categories',axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    #drop duplicates 
    df = df.drop_duplicates()
    return df 
    


def save_data(df, database_filename):
    engine = create_engine(database_filename) 
    #'sqlite:///disaster_project.db')
    df.to_sql('cleaned_messages', engine, index=False)
    
    #engine = create_engine(database_filename)
    #df.to_sql('cleaned_messages', engine, index=False)
    
    #pass  


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