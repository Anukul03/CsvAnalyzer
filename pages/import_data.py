import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt

def sample_data(df):
    st.subheader("Top 5 Rows")
    st.dataframe(df.head(), hide_index=True, use_container_width=True)

    st.subheader("Bottom 5 Rows")
    st.dataframe(df.tail(), hide_index=True, use_container_width=True)

    st.subheader("Random Sample Data")
    st.dataframe(df.sample(10), hide_index=True, use_container_width=True)
