This project is part of the Udacity program for Data Science Nanodegree. The project categorizes disaster data from Figure Eight to build a model for an API that classifies disaster messages into 36 categories. Input data consists of real messages that were sent during disaster events and their categorization to an appropriate disaster relief agency. 
The project includes:
- A machine learning pipeline to categorize the input messages to a disaser relief agency
- A web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the input data


-----
I. INSTALLATION

The project uses the following libraries:<br>
*numpy==1.12.1 <br>
plotly==2.0.15 <br>
nltk==3.2.5 <br>
SQLAlchemy==1.2.18 <br>
Flask==0.12.4 <br>
pandas==0.23.3 <br>
scikit_learn==0.21.3*

1. Run the following commands in the project's root directory to set up database and model.
	- To upgrade scikit-learn to latest version (scikit_learn==0.21.3; the provided Project Workspace IDE had an earlier version of scikit_learn)
		pip install --upgrade pip
		pip install --upgrade scikit-learn
	- To run ETL pipeline that cleans data and stores in database
		python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
	- To run ML pipeline that trains classifier and saves model
		python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2. Run the following command in the app's directory to run the web app.
	python run.py

3. Go to http://0.0.0.0:3001/


-----
II. FILE DESCRIPTIONS

| app <br>
| - templates <br>
|   | - master.html # main page of web app <br>
|   | - go.html # classification result page of web app <br>
| - run.py # Flask file that runs app <br>
| - custom_scorer_module.py # the used custom scoring function for the model <br>

| data <br>
| - disaster_categories.csv # data to process <br>
| - disaster_messages.csv # data to process <br>
| - process_data.py # ETL Pipeline preparation script <br>
| - DisasterResponse.db # database with cleaned data <br>
| - output.txt # captured output in one run of the process_data.py <br>

| models <br>
| - train_classifier.py # script that creates the machine learning model <br>
| - (classifier.pkl) # saved model; missing due to GitHub size requirements <br>
| - output.txt # captured output in one run of the train_classifier.py <br>
| - custom_scorer_module.py # the used custom scoring function for the model <br>

| README.md # this file


-----
III. RESULTS

The model achieves an accuracy rate of ~95%, correctly predicting no category for 90% of the cases and correctly classifying the messages for one of the agencies in 5% of the cases. The model used the accuracy ratio for scoring ((TP + TN)/Total Population). In real-life, different weights would be assigned to different prediction categories and the model would be optimized for that (e.g. TP = 100; FP = 10; FN = 50; TN = 1).

Achieved overall confusion matrix in %:<br>

| 		|	| Predicted category	|	|
|---------------|------:|:---------------------:|:-----:|
| 		|	| False			|True	|
| Real category	|False	| 90.0% 		| 1.1%	|
|               |True	| 3.9%	 		| 4.9%	|


IV. KEY TECHNICALITIES

1. The ETL Pipeline
	Loads the messages and categories datasets.
	Merges the two datasets.
	Cleans the data.
	Calculates the top 10 tokenized words present in each category.
	Stores the results in a SQLite database with two tables: 1) "messages" 2) "words".

2. The ML Pipeline
	Loads data from the SQLite database.
	Splits the dataset into training and test sets.
	Builds a text processing and machine learning pipeline using a RandomForest MultiOutputClassifier. The evaluation of the clasifier is done using a custom scoring function (Accuracy ratio = (TP + TN)/ Total Population; F-score was not used as it does not take the True Negatives rate into account
	Trains and tunes a model using GridSearchCV. To save processing time, parameters were optimized individually. (This does not guarantee the best global parameters, but should be close).
	Outputs results on the test set.
	Exports the final model as a pickle file.

3. Flask Web App
	Visualises the data using Plotly.
	Processes an input message and outputs classification results.

V. LICENSING, AUTHORS, ACKNOWLEDGEMENTS

Author: Dumitru Furtuna.

Dataset: Udacity and Figure Eight.

Templates: Python code structure and much of the flask web app was provided by Udacity.
