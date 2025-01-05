# Data Science Project #2 - Working to Classify Text into Different Emergency Categories.

## The Dataset
For this project, I as using a public dataset from the World Bank: 
https://data.worldbank.org/


## Acknowledgements 
I am utilizing some of the framework of the Udacity HTML code in my website. 


## Motivation
The motivation for this study was to fill a requirement for a Udacity course. A secondary reason was to make an attempt to build a website. Not everyone who takes this course is a web designer.


## Files Used
There are multiple files and directories in for this project. These are outlined by folder name, then files are subcatories under the directory.

A) #the app folder 
 1) HTML template sub folder
   a) master.HTML - the HTML file for the main page of the website. Also has some visuals 
   b) go.HTML - the HTML file for the page allowing the user to enter text and see which categories the model assigns to it
   c) metrics.HTML - this HTML file also has the same visuals as the master tab. I did this to experiment with labels at the top
   d) about.HTML - a small text blurb about me
 2) the __init__.py file. A mandatory python file needed in order for an app to run
 3) run.py - the python file that uses the Flask module that ties everything together
    
B) #the data folder
 1) categories.csv - one of 2 initial data files used to create a sqlite database. Has categories of each row. (note 1 row can be assigned to multiple cateogories).
 2) messages.csv - the actual text messages used as the raw data that would be assigned to a category (in the previous line item above in #1)
 3) process_data.py - a python script that merges and cleans the 2 csv files listed above, then stores as a sqlite database.
 4) disaster_project.db - this does not yet exist until someone runs the process_data.py script in the previous step.
    
C) #the models folder
 1) disaster_project.db - a copy of the sqlite database for speed's sake so the python script (in the next step) can be run independently of the process_data.py
 2) train_classifier.py - a python script that takes data from the sqlite database then builds a machine learning model
 3) metrics_file.xlsx - output from a machine learning creation that is used for visuals in the website.
 4) model name - a pre trained pickle file that is small enough that GitHub allows for uploading to be used in the website.
    
D) #wrangling_scripts folder
 1) wrangling_metrics.py - a script used to assist in the creation of a few visuals for the website.
    
E) #the work_out_files folder - a folder of jupyter notebooks used for sketching out ideas. NOTE: these files are not connected to anything else nor are they used for the app. They are just used to show some of my thought processes.
 1) ETL_Pipeline_Preparation.ipynb - a workbook sketching out how to clean up the data
 2) ML_Pipeline_Prepartion.ipynb - the intial sketch creating a machine learning model to save to a pickle file
 3) ML_Pipeline_withGridSearch.ipynb - toying around with improving the model. Grid search, Random Grid Search, etc. 
  


## Python Libraries Used
pandas, 
numpy,
sklearn, 
scipy,
matplotlib
sys, 
sqlalchemy,
nltk, 
pickle,
re,
json,
plotly,
joblib,
os,
flask

## Questions Asked
In the process of working on this project, I considered how to improve a machine learning model that had multiple outputs. 


## Method
To answer these questions, I am utilizing python to do machine learning. I am using python version 3.8

First, I tried using a RandomForest classifier first wrapped in a MultiOutputClassifier function and inserted into a pipeline with a Count Vectorizer and a Tfidf transformer. 
Saving this to a pickle file ended up being a rather large file. GitHub has restrictions on the size of files allowed to be uploaded and the pickle file was beyond that allowed threshold. 

Second, I used a Logistic Regression classifier in the pipeline, again using the Count Vectorizer and Tfidf transformer in the pipeline. 
One issue I found was that at least one y output column only had 1 type of value (always "no") which caused issues with this particular type of model. However the size of the pickle file was much smaller. So I dropped this particular column from the model. This may not be the best approach, but, I did it from a practical standpoint, trying to get a working model uploaded. 

The accurancy seemed pretty high for the no frills model. But being the skeptic, there is likely some overfitting. 

Next I worked on improving the model. First I considered using a grid search. However, when using potential parameters for the Count Vectorizer, the Tfidf transformer, along with the model itself, the number of combinations would be well over 1000. And I found that each pass took my computer over 30 minutes. 
So I bagged this approach, then considered a random grid search, using the same set as the regular grid search, but only used 15 iterations. The accuracy of the best set from this was only 0.26. Hardly impressive. What this means is that there is room for improvement. If I could dedicate a computer to running more iterations over a multiday period, I suspect we'd get some outputs with better accuracy than 0.26. 

Another idea (that I have not yet implemented): work on each of the 30 y output categories seperately with a grid search. Get the top 10 combos for each. Then take some kind of vote. Will that be perfect? Unlikely. Some categories will perform better than others. But, it might increase the number of categories that could get a boost. 


## Summary

This was an interesting project. The front end was very new to me, trying to learn HTML, CSS, Bootstrap, Plotly and Flask all at the same time then trying to get them to work together. 
After spending time on this, I see why this is an important challenge, trying to assign emergency texts to the correct categories. 

To see more details, please reference the code in this repo. Thanks! 

 




