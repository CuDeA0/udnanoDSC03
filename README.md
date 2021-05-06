# Disaster Response Pipeline Project
This is a project done for Udacitys Nanodegree in Data Science
see 
- https://classroom.udacity.com/nanodegrees/nd025/dashboard/overview

The objective of this exercise was to look at the Data Engineering 
pipeline specifically

- ETL
  - Extract - Transform - Load
- NLP Pipelines
- ML - Pipelines

The code herein is forked from the project repo on udacity. 

## Objectives
The objective is to explore a Disaster Response Pipeline, building
a ML pipeline to categorize emergency messages based on their needs.

## Data
The source of the datsaet comes from 
- https://www.figure-eight.com

The raw datset goes through the ETL pipeline doing the following steps
in file `data/process_data.py`

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the datasets
- Stored the output in a SQLite database

The machine learning pipeline, found in file `models/train_classifier.py` does
the following

- Loads the data from SQLite database
- Splits the dataset into train/test sets
- Defines a text processing and ML pipeline (using sklearn)
- Outputs the results
- Exports model as a pickle


Additionally a flask web app is used to visualize the dataset. 
That code is explicitly from Udacity.

## Folder Structure

```
├── app                            |Flask apps folder
│   ├── run.py                     |Call this to run the app
│   └── templates                  |Store templates for html setup
│       ├── go.html                |
│       └── master.html            |
├── data                           |Store and process data
│   ├── disaster_categories.csv    |Categorization of data
│   ├── disaster_messages.csv      |Raw text data
│   └── process_data.py            |Process file for the datasets
├── LICENSE                        |MIT license file
├── models                         |Folder for the models
│   └── train_classifier.py        |Runfile for training a classifier (and outputtg data)
└── README.md                      | This information 
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
