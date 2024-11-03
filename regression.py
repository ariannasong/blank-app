import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.title("The relationship between the possum's age and its tail length using a scatter plot")
# Load data
df = pd.read_csv('possum.csv')

# Plot scatter plot
fig, ax = plt.subplots()
ax.scatter(df['age'], df['tailL'], alpha=0.3)
ax.set_xlabel("Age")
ax.set_ylabel("Tail Length")
ax.set_title("Tail Length vs Age")

st.pyplot(fig)
