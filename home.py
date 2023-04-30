import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load data
data_carstq = "/Users/nelson/Desktop/BI/data/caracteristique.csv"
df_carstq = pd.read_csv(data_carstq, delimiter=",")


# Title
st.markdown(
    "<h1 style='text-align: center;'>Projet Business Intelligences sur les emplois en France</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h2 style='text-align: center;'>Présentation des données</h2>",
    unsafe_allow_html=True
)






