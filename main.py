import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from analyze import analysis
from viz import visuals
from custom_viz import custom_visuals


st.set_page_config(page_title=None, page_icon="images/favicon.png", layout="wide", initial_sidebar_state="auto")

st.title("CSV Data Analyser")
sidebar = st.sidebar
sidebar.title("File Uploader")
uploaded_file = sidebar.file_uploader("Upload a CSV file", type=["csv"], key="123")

sidebar.subheader("Page")
selected_option = sidebar.selectbox("Select Page", ["Sample Data", "Analysis", "Visuals", "Custom Visuals"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if selected_option == "Sample Data":
        from import_data import sample_data
        sample_data(df)  # Pass the dataframe instead of importing uploaded_file
    elif selected_option == "Analysis":
        analysis(df)
    elif selected_option == "Visuals":
        visuals(df)
    elif selected_option == "Custom Visuals":
        custom_visuals(df)
else:
    st.write("Please upload a CSV file to get started.")
