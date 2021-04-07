# production-repo
This repository consists of the code for Stack-overflow tag prediction using deep learning algorithm.

Steps to run the code:

1. Install all the required libraries.
2. Execute preprocess.py file [File that imports the train and test csv files of stack-overflow data, does data wrangling and export the cleaned data as pickle files]
3. Execute train.py file [File that trains the data using neural network/deep learning algorithm and build the sequential nn model]
4. Execute model_precition.py [File that predicts the tag for the given test inputs and calculates the accuracy using the sequential deep learning model]
