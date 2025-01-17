# Data Science Project #2 - Working to Classify Text into Different Emergency Categories.

## The Dataset
For this project, I as using a public dataset from the World Bank: 
https://data.worldbank.org/


## Acknowledgements 
I am utilizing some of the framework of the Udacity HTML code in my website. And, at the point of being totally obvious, I am also using data made available from the Worldbank.


## Motivation
The motivation for this study was to fill a requirement for a Udacity course. A secondary reason was to make an attempt to build a website. Not everyone who takes this course is a web designer.


## Files Used

There are multiple files and directories in for this project. These are outlined by folder name, then files are subcategories under the directory.

A) The app folder 
 1) HTML template sub folder
   a) master.HTML - the HTML file for the main page of the website. Also has some visuals 
   b) go.HTML - the HTML file for the page allowing the user to enter text and see which categories the model assigns to it
   c) metrics.HTML - this HTML file also has the same visuals as the master tab. I did this to experiment with labels at the top
   d) about.HTML - a small text blurb about me
 2) the static sub folder - not used
 3) the __init__.py file. A mandatory python file needed in order for an app to run
 4) run.py - the python file that uses the Flask module that ties everything together
 5) the disaster.db - a database with cleaned text data
 6) wrangle_metrics.py - used to show the visuals 
    
B) The data folder
 1) categories.csv - one of 2 initial data files used to create a sqlite database. Has categories of each row. (note 1 row can be assigned to multiple cateogories).
 2) messages.csv - the actual text messages used as the raw data that would be assigned to a category (in the previous line item above in #1)
 3) process_data.py - a python script that merges and cleans the 2 csv files listed above, then stores as a sqlite database.
 4) disaster_project.db - this does not yet exist until someone runs the process_data.py script in the previous step.
    
C) The models folder
 1) create_small_pickle_model_for_web.py - creates a fast, small model. There may be better models with longer scripts, but there is a trade off to everything.
 2) LogisiticRegression.pkl - output from the script above.
 3) disaster_project.db - a copy of the sqlite database for speed's sake so the python script (in the next step) can be run independently of the process_data.py
 4) train_classifier.py - a python script that takes data from the sqlite database then builds a machine learning model
 5) metrics_file.xlsx - output from a machine learning creation that is used for visuals in the website.
 6) model name - a pre trained pickle file that is small enough that GitHub allows for uploading to be used in the website.
 7) reduce_size_of_pickle_file.py - experimenting with trying to shrink the size. Not a necessary file. Including for auditing purposes.   
    
D) The wrangling_scripts folder
 1) wrangling_metrics.py - a script used to assist in the creation of a few visuals for the website.
    
E) The work_out_files folder - a folder of jupyter notebooks used for sketching out ideas. NOTE: these files are not connected to anything else nor are they used for the app. They are just used to show some of my thought processes and for an audit by the reviewer. This folder is not used to run the app, but included for reference and auditing purposes. I tried to make copious amounts of comments. 
 1) ETL_Pipeline_Preparation.ipynb - a workbook that cleaned and stored the data used in this model
 2) ML_Pipeline_Prepartion.ipynb - the intial sketch creating a machine learning model to save to a pickle file
 3) ML_Pipeline_withGridSearch.ipynb - toying around with improving the model. Grid search, Random Grid Search, etc.
 4) other files (some duplicates from other folders): 2 output models using pickle and joblib, another copy of the database, output metric files. 
  


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
In addition, being new to the front end, another question was how to get a webpage up and running, complete with some visuals.  
Admittedly, this took some time to digest. 


## Method
To answer these questions, I am utilizing python to do machine learning. I am using python version 3.11

In the workout folder I show how I created a small model for the website, and also other scripts showed a way to create other models. In yet another script I began a grid search, however the number of permutations was very large so I switched to a random grid search. Again, we may see a trade off. For the purpose of this project I kept the number low. Naturally if you have time and a spare computer, you could run it for longer periods to grab the most optimal combination of parameters. 

I also have scripts to create other models (not included but could be run using scripts in the workout folder)


## To run the app (locally)

Download this directory
Set up your python environment - I used python 3.11 for this. 
Navitage to the app subdirectory in your terminal machine
type python run.py
go to the URL showing in the terminal. 

## Summary

This was an interesting project. The front end was very new to me, trying to learn HTML, CSS, Bootstrap, Plotly and Flask all at the same time then trying to get them to work together. 
After spending time on this, I see why this is an important challenge, trying to assign emergency texts to the correct categories. 

To see more details, please reference the code in this repo. Thanks! 

 




