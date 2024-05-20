# Sentiment Classification with Bagging and SVM
This project demonstrates how to build a sentiment classification model using the Bagging ensemble method with Support Vector Machine (SVM) as the base learner. The model is trained to classify sentiment into categories like positive, negative, and neutral based on textual data. The implementation is done in Python using scikit-learn and includes steps for data preprocessing, model training, evaluation, and saving the trained model.
# Project Structure
* model.py: Contains the code for training the Bagging model with SVM.
* test.py: Example code to load the saved model and make predictions on new data
* bagging_svm_classifier.pkl: Saved Bagging model with SVM.
* vectorizer.pkl: Saved vectorizer for transforming input text data.
# Requirements
* Python 3.6+
* scikit-learn
* pandas
* nltk
* pickle
# Installation
1- Clone the repository:
```
git clone https://github.com/yourusername/sentiment-classification-bagging.git
cd sentiment-classification-bagging
```
2- Create a virtual environment and activate it
```
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```
3- Install the required packages:
```
pip install -r requirements.txt
```
# Usage
### Training the Model
- Place your dataset file (all-data.csv) in the project directory.
- Run the model.py script to train the model:
```
python model.py
```
This will train the Bagging model with SVM and save it to bagging_svm_classifier.pkl
### Making Predictions
- Run the test.py script to load the saved model and make predictions on new data:
```
python test.py
```
# Additional Information
The text data is preprocessed using the following steps:
* Tokenization
* Removal of stop words
* Stemming (if applicable)
# Model Training
The Bagging model with SVM is trained using the preprocessed data. The training process includes:
* Splitting the data into training and testing sets
* Training the model on the training data
* Evaluating the model on the testing data
# Evaluation Metrics
The model's performance is evaluated using accuracy, precision, recall, and F1-score.
# Contributing
If you wish to contribute to this project, please fork the repository and submit a pull request.
# Contact
For any questions or feedback, please contact [magdyabdo484@gmail.com]
