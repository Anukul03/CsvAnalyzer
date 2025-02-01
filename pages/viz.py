import pandas as pd
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def visuals(df):
    st.subheader("Histogram")
    column_to_plot = st.selectbox("Select a column to plot", df.columns)

    try :
        fig, ax = plt.subplots()
        df[column_to_plot].plot(kind='hist', ax=ax)
        st.pyplot(fig)
    except:
        st.write("Please select a neumerical column to plot.")
