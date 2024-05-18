import os
import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pandas as pd

# Download necessary resources for NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def read_data(filepath):
    data = {'text': [], 'label': []}
     

        # Open each file and read its content
    with open(filepath, encoding='utf-8') as file:
        lines = file.readlines() 
    
    # Extract emotional states and labels from each line
    for line in lines:
        parts = line.strip().split(':')
        parts = [remove_punctuation(part).lower() for part in parts]
        if len(parts) == 2:
            text, label = parts
            data['text'].append(text)
            data['label'].append(label)
    print(data['text'][:5])        
    return data

def read_data_csv(filepath):
    data = {'text': [], 'label': []}
    # Read data From csv
    read_data = pd.read_csv(filepath , encoding='latin1')
    
    # Extract labels and text from each row
    data['text'] = read_data.iloc[:,1]
    data['label'] = read_data.iloc[:,0]

    print(data['text'][:5])
    print(data['label'][:5])
    return data

def remove_punctuation(text):
    # Remove punctuation characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    
    # Removing punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text



# Directory containing the data files
filepath = 'emotions.txt'
csv_path = 'all-data.csv'

# Read the data
# data = read_data(filepath)

# Read the data from csv
data = read_data_csv(csv_path)

# Apply preprocessing to each text in the dataset
data['preprocessed_text'] = [preprocess_text(text) for text in data['text']]

#save data to csv
# Convert data to a DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv('data.csv', index=False)

# Vectorize the preprocessed text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['preprocessed_text'])
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Support Vector Classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Train a Bagging Classifier
bagging_svm = BaggingClassifier(estimator=classifier, n_estimators=10, random_state=42)
bagging_svm.fit(X_train, y_train)

# Save the trained model and vectorizer
with open('classifier.pkl', 'wb') as model_file:
    pickle.dump(bagging_svm, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Predict on the test data
y_pred = classifier.predict(X_test)
y_pred_bagging = bagging_svm.predict(X_test)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("=====================")
print(classification_report(y_test, y_pred_bagging))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("=====================")
print(confusion_matrix(y_test, y_pred_bagging))