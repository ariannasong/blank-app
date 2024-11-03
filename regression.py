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



# Title and Explanation
st.header("Linear Model Prediction Function")

st.markdown(r"""
### Task: Implement a Linear Model Prediction Function
To obtain predictions using a linear model, we use the following formula:

\[
\hat{y} = \mathbf{X} \hat{\beta}
\]

where:
- \( \mathbf{X} \) is the design matrix, which includes a column of ones,
- \( \hat{\beta} \) is the coefficient vector,
- \( \hat{y} \) represents the predicted outcomes.

Your task is to create a function `linearModelPredict` that:
- Takes `b`, a 1D array of coefficients, and `X`, a 2D array representing the design matrix.
- Returns the predictions `y_p` calculated as \( \hat{y} = \mathbf{X} \hat{\beta} \).

### Implementation
Below is the code for `linearModelPredict` and a test case to verify its accuracy.
""")

# Define the prediction function
def linearModelPredict(b, X):
    # calculate prediction
    yp = np.dot(X, b.T)
    return yp

# Display code snippet
st.code("""
def linearModelPredict(b, X):
    # calculate prediction
    yp = np.dot(X, b.T)
    return yp
""", language="python")

# Test data
X = np.array([[1, 0], [1, -1], [1, 2]])
b = np.array([0.1, 0.3])

# Run function and calculate predictions
yp = linearModelPredict(b=b, X=X)

# Display results
st.subheader("Test Results")
st.write("Coefficient vector `b`:", b)
st.write("Design matrix `X`:\n", X)
st.write("Predicted values `y_p`:", yp)
st.write("Dimensionality of `y_p`:", yp.shape)

st.markdown(r"""
### Observation
- The dimensionality of `y_p` is 3, corresponding to the 3 rows (data points) in \( \mathbf{X} \).
- If `b` were a 2D array, the output dimensions of `y_p` would vary depending on its shape.
""")


