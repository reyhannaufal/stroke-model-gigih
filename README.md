# Sentiment Analysis - Tweet

A Web App to perform sentiment analysis on tweets. 

App is deployed in Heroku, click the link to access it : [Open in Heroku](https://toDeploy/) 

## Summary

Natural Language Processing (NLP): The discipline of computer science, artificial intelligence and linguistics that is concerned with the creation of computational models that process and understand natural language. These include: making the computer understand the semantic grouping of words (e.g. cat and dog are semantically more similar than dog and spoon), text to speech, language translation and many more

Sentiment Analysis: It is the interpretation and classification of emotions (positive, negative and neutral) within text data using text analysis techniques. Sentiment analysis allows organizations to identify public sentiment towards certain words or topics.

In this notebook, we'll develop a Sentiment Analysis model to categorize a tweet as Positive or Negative.

## Data

The data for the following problem is [available on Kaggle.](https://www.kaggle.com/kazanova/sentiment140) 
Since the data has been added to the `data/` directory, cloning this repository would suffice.
Note : Please unzip the csv file before running the notebook.

## Pre-requisites

The project was developed using python 3.8.3 with the following packages.
- Pandas
- Numpy
- matplotlib
- Scikit-learn
- NLTK
- wordcloud
- Streamlit
- pickle

Installation with pip:

```bash
pip install -r requirements.txt
```

## Getting Started
Open the terminal in you machine and run the following command to access the web application in your localhost.
```bash
streamlit run app.py
```


## Files
- notebook/Sentiment_Analysis_Tweet_LemmatizingWithPOS.ipynb : Jupyter Notebook with all the workings including pre-processing using Lemmatizing with POS taggers, modelling and inference using NLTK library and classification algorithm like Naive Bayes and logistic regression.
- app.py : Streamlit App script
- requirements.txt : pre-requiste libraries for the project
- models/ : TFIDF vectoriser object and trained model (the vectorizer and model is created based on this file "Sentiment_Analysis_Tweet_LemmatizingWithPOS.ipynb")
- data/ : source data
- setup.sh : Setup file for Heroku.
- Procfile : To trigger the app in Heroku.


## Acknowledgements

[Kaggle](https://kaggle.com/), for providing the data for this problem statement.