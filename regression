import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
df = pd.read_csv('possum.csv')

# Plot the scatter plot
fig, ax = plt.subplots()
ax.scatter(df['age'], df['tailL'], alpha=0.3)
ax.set_xlabel("Age")
ax.set_ylabel("Tail Length")
ax.set_title("Tail Length vs Age")

# Display the plot with Streamlit
st.pyplot(fig)
