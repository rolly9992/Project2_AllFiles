# Data Science Project #2 - Working to Classify Text into Different Emergency Categories.


## The Dataset

For this project, I am using a public dataset from the World Bank: 
https://data.worldbank.org/


## Acknowledgements 

I am utilizing some of the framework of the Udacity HTML code in my website. And, as mentioned above, I am also using data made available from the Worldbank.


## Motivation

The motivation for this study was to fill a requirement for a Udacity course. A secondary reason was to make an attempt to build a website. Not everyone who takes this course is a web designer.


## Files Used

There are multiple files and directories in for this project. These are outlined by folder name, then files are subcategories under the directory. Instructions on how to run the code are below

A) The app folder 

 1) HTML template sub folder

   a) master.HTML - the HTML file for the main page of the website. 
   
   b) go.HTML - the HTML file for the page allowing the user to enter text and see which categories the model assigns to it
   
   c) metrics.HTML - this HTML file also has the same visuals as the master tab. I did this to experiment with labels at the top
   
   d) about.HTML - a small text blurb about me
 
 3) the static sub folder - not used
 
 4) the __init__.py file. A mandatory python file needed in order for an app to run
 
 5) run.py - the python file that uses the Flask module that ties everything together, pulling HTML files in.
 
    

    
B) The data folder

 1) categories.csv - one of 2 initial data files used to create a sqlite database. Has categories of each row. (note 1 row can be assigned to multiple cateogories).
 
 2) messages.csv - the actual text messages used as the raw data that would be assigned to a category (in the previous line item above in #1)
 
 
    
C) The models folder
 
 1) tokenizer.py - the tokenizer saved seperately to be used in the model
 

D) Wrangling_scripts

 1) Wrangle_metrics.py - uses the output of the model metric to create visuals for F1 Scores, accuracy, precision and recall for the y outputs. 

    
E) The work_out_files folder [OPTIONAL REFERENCE SCRIPTS] - a folder of jupyter notebooks used for sketching out ideas. NOTE: these files are not connected to anything else nor are they used for the app. 

They are just used to show some of my thought processes and for an audit by the reviewer. 

 1) ETL_Pipeline_Preparation.ipynb - a workbook that cleaned and stored the data used in this model
 
 2) ML_Pipeline_Preparation.ipynb - the intial sketch creating a machine learning model to save to a pickle file
 
 3) ML_Pipeline_withGridSearch.ipynb - using a random grid search. if you want to scan many combinations, this takes a long time. I tested it with a smaller set of 5, but the trade off is while that runs faster, 5 is really not enough to find the best combination.
 


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

To run the code and see the local webapp, please read the next section. 



## How to Run the Code and See the Web App (locally)

First, set up the python environment using the requirements.txt file with python version = 3.11

If someone was to follow along with the code, the order would be:

1) Navigate to the root folder of your directory and run the following scripts in order. 

2) First, run python process_data.py - this manipulates and cleans the data and stores it into a sqlite database.  
  
4) Second, run train_classifier.py - this generates a logistic regression NLP model fairly quickly (give it a minute or so). This generates the model that will be used in the website

5) Third run main.py - this will generate a website. Once you see the output (similar to the image below) in the terminal, you can press control, then left click on the website to open it up in your browser.


![image](https://github.com/user-attachments/assets/04cd5f71-0a62-47de-976f-3df94a44a536)

 



## Summary

This was an interesting project. The front end was very new to me, trying to learn HTML, CSS, Bootstrap, Plotly and Flask all at the same time then trying to get them to work together. 
After spending time on this, I see why this is an important challenge, trying to assign emergency texts to the correct categories. 

To see more details, please reference the code in this repo. Thanks! 

 




