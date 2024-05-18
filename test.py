import tkinter as tk
from tkinter import messagebox
import pickle
import nltk
from nltk.stem import PorterStemmer

# Ensure NLTK resources are downloaded
nltk.download('punkt')

# Load the trained model and vectorizer
with open('classifier.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocessing function
def preprocess_text(text):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text.lower())  # Lowercasing and tokenization
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)  # Joining back into a single string

# Function to predict sentiment
def predict_sentiment():
    input_text = text_entry.get()
    if not input_text:
        messagebox.showerror("Input Error", "Please enter some text.")
        return
    
    preprocessed_text = preprocess_text(input_text)
    text_vector = vectorizer.transform([preprocessed_text])  # Vectorizer handles raw text
    prediction = classifier.predict(text_vector)
    sentiment_label.config(text=f"Predicted Sentiment: {prediction[0]}")

# Create GUI
root = tk.Tk()
root.title("Sentiment Analysis GUI")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack(padx=10, pady=10)

text_label = tk.Label(frame, text="Enter text:")
text_label.grid(row=0, column=0, pady=(0, 10))

text_entry = tk.Entry(frame, width=50)
text_entry.grid(row=0, column=1, pady=(0, 10))

predict_button = tk.Button(frame, text="Predict Sentiment", command=predict_sentiment)
predict_button.grid(row=1, columnspan=2, pady=10)

sentiment_label = tk.Label(frame, text="Predicted Sentiment:")
sentiment_label.grid(row=2, columnspan=2, pady=10)

root.mainloop()