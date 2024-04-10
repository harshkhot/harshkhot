import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
data = pd.DataFrame({
    'x': range(20),
    'y1': np.random.randn(20).cumsum(),
    'y2': np.random.randn(20).cumsum()
})

# Streamlit app
def main():
    st.title('Basic Streamlit Dashboard')

    # Line Chart
    st.subheader('Line Chart')
    st.line_chart(data.set_index('x'))

    # Bar Chart
    st.subheader('Bar Chart')
    st.bar_chart(data.set_index('x'))

    # Scatter Plot
    st.subheader('Scatter Plot')
    st.write("Random scatter plot with normally distributed data")
    st.write(data)
    st.write(plt.scatter(data['y1'], data['y2']))
    st.pyplot()

if __name__ == "__main__":
    main()
