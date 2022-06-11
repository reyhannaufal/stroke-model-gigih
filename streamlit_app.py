import streamlit as st
import pandas as pd

st.title('Diagon Alley')

st.write('Diagon Alley is an end-to-end ML model project that uses a combination of machine learning techniques to predict stroke ✊')

#import stroke dataset
df = pd.read_csv('./stroke_dataset.csv')

# display data
st.write(df.head())


