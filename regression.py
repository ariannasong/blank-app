import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.title("Supervised Learning, Linear Models, and Loss Functions")
st.markdown("""
### Overview
The purpose of this project is to explore the relationship between different variables in a dataset of possums from Australia and New Guinea.
""")
st.subheader("Data Set")
st.markdown("""
The dataset contains 46 observations on the following 6 variables:
""")
st.write("""
- **sex**: Sex, either m (male) or f (female).
- **age**: Age in years.
- **headL**: Head length, in mm.
- **skullW**: Skull width, in mm.
- **totalL**: Total length, in cm.
- **tailL**: Tail length, in cm.
""")

st.subheader("The relationship between the possum's age and its tail length using a scatter plot")
# Load data
df = pd.read_csv('possum.csv')

# Plot scatter plot
fig, ax = plt.subplots()
ax.scatter(df['age'], df['tailL'], alpha=0.3)
ax.set_xlabel("Age")
ax.set_ylabel("Tail Length")
ax.set_title("Tail Length vs Age")

st.pyplot(fig)

import streamlit as st
import numpy as np

st.title("Linear Model Prediction Function")

st.markdown("""
This example demonstrates how to compute predictions using a linear model. The predictions are calculated with the formula:

\\[ y_p = X \\cdot b \\]

where:
- \\( X \\) is the design matrix, which includes a column of ones.
- \\( b \\) is the coefficient vector.

The function `linearModelPredict` will take a coefficient vector `b` and a design matrix `X` and return the predictions.
""")

# Define the prediction function
def linearModelPredict(b, X):
    # calculate prediction
    yp = np.dot(X, b.T)
    return yp

# Sample data
X = np.array([[1, 0], [1, -1], [1, 2]])
b = np.array([0.1, 0.3])

# Calculate predictions
yp = linearModelPredict(b=b, X=X)

# Display results
st.subheader("Testing the linear model prediction function")
st.write("Coefficient vector `b`:", b)
st.write("Design matrix `X`:\n", X)
st.write("Predicted values `y_p`:", yp)
st.write("Dimensionality of `y_p`:", yp.shape)

st.markdown("""
### Observation
- Since `X` has 3 rows (data points), the output `y_p` also has 3 values.
- If `b` were a 2D array, the output dimensions would change accordingly.
""")
