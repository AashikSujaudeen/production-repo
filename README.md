# production-repo

This repository consists of the code for Stack-overflow tag prediction using deep learning algorithm.

Steps to run the code:

Pre-Execution preparation:

Install all the required libraries.

Download 'Train.zip' from the link https://www.kaggle.com/reintegrated/4-stackoverflow-training?select=Train.zip by scrolling all the way down and expanding 'Facebook Recruiting III - Keyword Extraction' folder under 'Data Sources'.

Extract the zip file and place the file in a folder 'Datasets'

Execution Steps:

Navigate to master branch and execute preprocess.py file [File that imports the train and test csv files of stack-overflow data, does data wrangling and export the cleaned data as pickle files]

Execute train.py file [File that trains the data using neural network/deep learning algorithm and build the sequential nn model]

Execute model_precition.py [File that predicts the tag for the given test inputs and calculates the prediction accuracy using the sequential deep learning model]
