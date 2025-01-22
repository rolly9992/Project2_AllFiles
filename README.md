# Data Science Project #2 - Working to Classify Text into Different Emergency Categories.

<pre>


</pre>

## The Dataset

For this project, I am using a public dataset from the World Bank: 
https://data.worldbank.org/

<pre>


</pre>

## Acknowledgements 

I am utilizing some of the framework of the Udacity HTML code in my website. And, as mentioned above, I am also using data made available from the Worldbank.


<pre>


</pre>

## Motivation

The motivation for this study was to fill a requirement for a Udacity course. A secondary reason was to make an attempt to build a website. Not everyone who takes this course is a web designer.

<pre>


</pre>

## Files Used

There are multiple files and directories in for this project. These are outlined by folder name, then files are subcategories under the directory. Instructions on how to run the code are below

A) The app folder 

 1) HTML template sub folder

   a) master.HTML - the HTML file for the main page of the website. 
   
   b) go.HTML - the HTML file for the page allowing the user to enter text and see which categories the model assigns to it
   
   c) metrics.HTML - the HTML file which has the visuals used in the website
   
   d) about.HTML - an HTML file with a small text blurb about me
 
 3) the static sub folder - not used
 
 4) the __init__.py file. A mandatory python file needed in order for an app to run
 
 5) run.py - the python file that uses the Flask module that ties everything together, pulling HTML files in.

 6) wrangle_metrics.py - a script used to build the visuals 
 


    
B) The data folder

 1) categories.csv - one of 2 initial data files used to create a sqlite database. Has categories of each row. (note 1 row can be assigned to multiple cateogories).
 
 2) messages.csv - the actual text messages used as the raw data that would be assigned to a category (in the previous line item above in #1)

 3) process_data.py - a script used to modify data from the 2 csv files above, then store in a sqlite database.

 4) what_version.py - a simple script that shows the python version being used (including the micro version).
     

    
C) The models folder
 
 1) train_classifier.py - a script that loads data from the database, trains, evaluates and saves a machine learning model that will be used in the Flask web app we will be using.


    
D) The work_out_files folder [OPTIONAL REFERENCE SCRIPTS] - a folder of jupyter notebooks used for sketching out ideas. NOTE: these files are not connected to anything else nor are they used for the app. 

They are just used to show some of my thought processes and for an audit by the reviewer. 

 1) ETL_Pipeline_Preparation.ipynb - a workbook that cleaned and stored the data used in this model
 
 2) ML_Pipeline_Preparation.ipynb - the intial sketch creating a machine learning model to save to a pickle file
 
 3) ML_Pipeline_withGridSearch.ipynb - using a random grid search. if you want to scan many combinations, this takes a long time. I tested it with a smaller set of 5, but the trade off is while that runs faster, 5 is really not enough to find the best combination.
 <pre>


</pre>


## Python Libraries Used

pandas, 
numpy,
sklearn, 
scipy,
matplotlib,
sys, 
sqlalchemy,
nltk, 
pickle,
re,
json,
plotly,
joblib,
os,
flask,
openpyxl
<pre>


</pre>

## Questions Asked

In the process of working on this project, I considered how to improve a machine learning model that had multiple outputs. 
In addition, being new to the front end, another question was how to get a webpage up and running, complete with some visuals.  
Admittedly, this took some time to digest. 
<pre>


</pre>


## Method

To answer these questions, I am utilizing python to do machine learning. I am using python version 3.11

To run the code and see the local webapp, please read the next section. 

<pre>


</pre>



## Here are the steps to run the code and launch the Flask Web App (locally)

1) Download this repo to your local computer
2) Extract the data from this repo to a folder. Name the folder whatever you like. For the instructions, I'm naming my folder FlaskFiles
3) Open the terminal
4) Navigate to the root folder of the directory that we just downloaded and extracted. If you also named your folder FlaskFiles, your terminal location should be in ../FlaskFiles. We will be staying here to run all our code to get the app up and running. 
5) Create a new python environment. I am using conda and naming my new environment myenv. You can name your environment whatever you like. Note that I am using python version 3.11. Type the following command in the terminal:
<pre>
conda create –name myenv python==3.11
</pre>
6) It will ask you if you want to proceed with a y/n option. Press y
   
7) Activate the new python evironment you just created with the following code in the terminal:

<pre>
conda activate myenv
</pre>

8) Please enter the following to install the requirements.txt file:

<pre>
conda install --yes --file requirements.txt
</pre>

9) After the requirements are installed, we can proceed to running the first script. This script pulls data from some csv files, prepares the data, then stores the output in a database. There are 4 pieces to the command line we will enter into the terminal: the name of the first python script, the name of 2 input csv files (kindly keep in the same order as listed below) and finally the folder and name of the output database. If you'd like you can change the name (but not the folder) of the database but it's not required. Please copy or type the following in the terminal:

<pre>
python data/process_data.py data/messages.csv data/categories.csv data/disaster_project.db
</pre>

   
10) Next, we can run the next script which pulls from the database we created in the previous step, then creates, trains, evalulates and saves a machine learning model. There are 3 pieces to the command we put in the terminal: the script that creates the machine learning model, the database it pulls from (should be the same database name as what was entered in the previous step) and the name we call the saved machine learning model. I'm calling the output model my_model.pkl. You can rename it if you like but that is not required. Please copy or type the following into the terminal:
 
 <pre>
 python models/train_classifier.py data/disaster_project.db models/my_model.pkl
 </pre>
 

12) Finally, to start the Flask app and get the URL we can go to, copy or type the following in the terminal:
This will generate a URL website. Once you see the output (similar to the image below) in the terminal, you can press control, then left click on the website URL to open it up in your browser.


<pre>
python app/run.py
</pre>


![image](https://github.com/user-attachments/assets/04cd5f71-0a62-47de-976f-3df94a44a536)

 
<pre>


</pre>

## Structure of the Flask App
There are 4 pages on this web app. 

On the main page, there is a query where you can enter some text and press the button to see how the model would classify the text. Note it can choose more than one category. 

The second page are visuals for the model metrics, showing accuracy, precision, recall and F1 scores for the y output variables. 

The third page is a small blurb about me

The 4th is actually not part of the website but a link to this GitHub repo. 


## Summary

This was an interesting project. The front end was very new to me, trying to learn HTML, CSS, Bootstrap, Plotly and Flask all at the same time then trying to get them to work together. 
After spending time on this, I see why this is an important challenge, trying to assign emergency texts to the correct categories. 

To see more details, please reference the code in this repo. Thanks! 

 




