import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load data
data_carstq = "/Users/nelson/Desktop/BI/data/caracteristique.csv"
data_BMO = "/Users/nelson/Desktop/BI/data/BMO.csv"
df_carstq = pd.read_csv(data_carstq, delimiter=",")
df_BMO = pd.read_csv(data_BMO, delimiter=",")



# Title
st.title("Datasets sur les emplois en France")

# data
st.write(df_carstq)
st.write(df_BMO)