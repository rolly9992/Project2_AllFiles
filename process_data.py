
# # ETL Pipeline Preparation
# Follow the instructions below to help you create your ETL pipeline.
# ### 1. Import libraries and load datasets.
# - Import Python libraries
# - Load `messages.csv` into a dataframe and inspect the first few lines.
# - Load `categories.csv` into a dataframe and inspect the first few lines.


# import libraries
import pandas as pd
import numpy as np 
from sqlalchemy import create_engine
import os 
import sys 

# load messages dataset
message_file_path = os.path.join('data', 'messages.csv')

messages = pd.read_csv(message_file_path,sep=',')
print('messages data loaded')
#print(messages.shape)
#messages.head()


# load categories dataset
category_file_path = os.path.join('data', 'categories.csv')
categories = pd.read_csv(category_file_path,sep=',')
print('categories data loaded')
# print(categories.shape)
# categories.head()


# ### 2. Merge datasets.
# - Merge the messages and categories datasets using the common id
# - Assign this combined dataset to `df`, which will be cleaned in the following steps


# merge datasets
df = messages.merge(categories,on='id',how='inner')
# df.head()


# ### 3. Split `categories` into separate category columns.
# - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.
# - Use the first row of categories dataframe to create column names for the categories data.
# - Rename columns of `categories` with new column names.


# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(';',expand=True)
# categories.head(2)



# select the first row of the categories dataframe
row = categories.iloc[0]
#print(row)
# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
category_colnames = [x[:-2] for x in row]
# print(category_colnames)


# rename the columns of `categories`
categories.columns = category_colnames
categories.head(2)


# ### 4. Convert category values to just numbers 0 or 1.
# - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, `related-0` becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.
# - You can perform [normal string actions on Pandas Series](https://pandas.pydata.org/pandas-docs/stable/text.html#indexing-with-str), like indexing, by including `.str` after the Series. You may need to first convert the Series to be of type string, which you can do with `astype(str)`.

for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
    #categories[column] = categories[column].astype('int') 
# categories.head()


# ### 5. Replace `categories` column in `df` with new category columns.
# - Drop the categories column from the df dataframe since it is no longer needed.
# - Concatenate df and categories data frames.


# drop the original categories column from `df`

df = df.drop('categories',axis=1)
# df.head()

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df,categories],axis=1)
# df.head()


# ### 6. Remove duplicates.
# - Check how many duplicates are in this dataset.
# - Drop the duplicates.
# - Confirm duplicates were removed.

# check number of duplicates
dupes = df.duplicated()
# print(dupes.sum())

# drop duplicates
df = df.drop_duplicates()

# check number of duplicates
dupes = df.duplicated()
# print(dupes.sum())


# ### 7. Save the clean dataset into an sqlite database.
# You can do this with pandas [`to_sql` method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html) combined with the SQLAlchemy library. Remember to import SQLAlchemy's `create_engine` in the first cell of this notebook to use it below.
print('data cleaned')
#sys.exit()

#db_path = os.path.join('data', 'disaster_project.db')
engine = create_engine('sqlite:///data/disaster_project.db')
try:
    df.to_sql('cleaned_messages', engine, index=False)
    print('sqlite database created with cleaned, processed data')

except Exception as e:
    print('exception of :',e)



# ### 8. Use this notebook to complete `process_data.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database based on new datasets specified by the user. Alternatively, you can complete `process_data.py` in the classroom on the `Project Workspace IDE` coming later.





