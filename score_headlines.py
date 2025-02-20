#!/usr/bin/env python
# coding: utf-8

# #### **ADSP 32026 Project 1**

# #### Simran Karamchandani

# #### Importing Libraries

# In[1]:


import sys
import os
import joblib
from datetime import datetime
from sentence_transformers import SentenceTransformer
from joblib import load


# #### Check Validity of Input

# In[2]:


def validate_input():
    if len(sys.argv) != 3:
        print("Invalid number of inputs. Usage: python score_headlines.py <HEADLINE_FILE> <SOURCE_NAME>")
        sys.exit(1)

    headline_file = sys.argv[1]
    source_name = sys.argv[2]

    if not os.path.exists(headline_file):
        print(f"Error: Unable to find '{headline_file}'.")
        sys.exit(1)

    if not headline_file.endswith('.txt'):
        print(f"Error:'{headline_file}' is not a text file.")
        sys.exit(1)

    return headline_file, source_name


# #### Read Headline

# In[3]:


def read_headlines_from_file(headline_file):
    try:
        with open(headline_file, 'r', encoding='utf-8') as file:
            headlines = [line.strip() for line in file if line.strip()]
        return headlines
    except Exception as e:
        print(f"Error reading file '{headline_file}': {e}")
        sys.exit(1)


# #### Load Transformer Model/Sentiment Classifier

# In[ ]:


# def load_model_and_transformer(model_path: str, transformer_name: str):
#      try:
#         model = load(model_path)
#         transformer = SentenceTransformer(transformer_name)
#         return model, transformer
#     except Exception as e:
#         print(f"Error loading model from '{model_path}' or transformer '{transformer_name}': {e}")
#         sys.exit(1)


# #### Analyze Headline 

# In[4]:


def analyze_headline(headlines: list, model, transformer) -> list:
    try:
        vectors = transformer.encode(headlines)
        predictions = model.predict(vectors)
        results = [(pred, headline) for pred, headline in zip(predictions, headlines)]
        return results
    except Exception as e:
        print(f"Error during sentiment prediction: {e}")
        sys.exit(1)


# In[4]:


# def analyze_headline_sentiment(headlines, transformer_model, sentiment_classifier):
#     try:
#         if isinstance(headlines, str):
#             headlines = [headlines]
        
#         print("Generating embeddings for headlines...")
#         embeddings = transformer_model.encode(headlines)
        
#         if embeddings.ndim == 1: 
#             embeddings = embeddings.reshape(1, -1)
#         elif embeddings.ndim == 2 and embeddings.shape[1] != 384:
#             raise ValueError(f"Expected embeddings with 384 features, but got {embeddings.shape[1]}.")
        
#         print("Predicting sentiment...")
#         predictions = sentiment_classifier.predict(embeddings)
        
#         label_mapping = {0: "neutral", 1: "positive", -1: "negative"}
#         readable_predictions = [label_mapping.get(pred, "unknown") for pred in predictions]
        
#         results = list(zip(readable_predictions, headlines))
#         return results
    
#     except Exception as e:
#         print(f"Error during sentiment prediction: {e}")
#         sys.exit(1)


# In[15]:


#sample_embedding = transformer_model.encode(["Breakthrough in Cancer Treatment Shows Promising Results"])
#print("Prediction from SVM model:", sentiment_classifier.predict(sample_embedding))


# #### Save File

# In[5]:


def save_file(results, source):
    today = datetime.today().strftime("%Y_%m_%d")
    final_file = f"headline_scores_{source}_{today}.txt"

    try:
        with open(final_file, "w", encoding="utf-8") as file:
            for label, headline in results:
                file.write(f"{label}, {headline}\n")
        print(f"Final File: {final_file}")
    except Exception as e:
        print(f"Error while saving final file: {e}")
        sys.exit(1)
    try:
        with open(final_file, "r", encoding="utf-8") as file:
            for line in file:
                print(line.strip())  # .strip() to remove trailing newline characters
    except Exception as e:
        print(f"Error reading the file: {e}")


# #### Main Function

# In[6]:


def main():
    headline_file, source = validate_input()
    headlines = read_headlines_from_file(headline_file)

    print("Loading models...")
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')  
    sentiment_classifier = joblib.load("svm.joblib")  

    print("Analyzing headline sentiment...")
    results = analyze_headline(headlines, sentiment_classifier, transformer_model)

    save_file(results, source)


# In[6]:


if __name__ == "__main__":
    main()
# This is a test comment to trigger pre-commit
