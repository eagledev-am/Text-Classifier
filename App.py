import tkinter as tk
from tkinter import messagebox
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pickle

# Net sales decreased to EUR160 .3 m from EUR179 .9 m and pretax profit rose 
# Download necessary resources for NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and vectorizer
with open('classifier.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

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
    
    # stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

def classify_text():
    input_text = entry.get()
    preprocessed_text = preprocess_text(input_text)
    text_vector = vectorizer.transform([preprocessed_text])
    print(text_vector)
    prediction = classifier.predict(text_vector)
    print(prediction)
    messagebox.showinfo("Prediction", f"The predicted class is: {prediction[0]}")

# Create the main application window
root = tk.Tk()
root.title("Text Classifier")

# Create and place the input field
entry_label = tk.Label(root, text="Enter text:")
entry_label.pack()

entry = tk.Entry(root, width=50)
entry.pack()

# Create and place the classify button
classify_button = tk.Button(root, text="Classify", command=classify_text)
classify_button.pack()

# Start the Tkinter event loop
root.mainloop()