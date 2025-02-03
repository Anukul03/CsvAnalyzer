import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title=None, page_icon="images/favicon.png", layout="wide", initial_sidebar_state="auto")


############################# Sidebar #################################
st.title("CSV Data Analyser")
sidebar = st.sidebar
sidebar.title("File Uploader")

uploaded_file = sidebar.file_uploader("Upload a CSV file", type=["csv"], key="123")

sidebar.subheader("Page")
selected_option = sidebar.selectbox("Select Page", ["Sample Data", "Analysis", "Visuals", "Custom Visuals"])

############################### Main ##################################
def sample_data(df):
    st.subheader("Top 5 Rows")
    st.dataframe(df.head(), hide_index=True, use_container_width=True)

    st.subheader("Bottom 5 Rows")
    st.dataframe(df.tail(), hide_index=True, use_container_width=True)

    st.subheader("Random Sample Data")
    st.dataframe(df.sample(10), hide_index=True, use_container_width=True)

############################## Analyze #################################

def analysis(df):
    st.subheader("Data Preview")
    st.dataframe(df.head(),hide_index=True, use_container_width=True)

    # Combine Data Types, Summary Statistics, and Null Values
    st.subheader("Data Types and Summary Statistics")

    # Get data types
    data_types = df.dtypes.reset_index()
    data_types.columns = ["Column Name", "Data Type"]

    # Get summary statistics
    summary_stats = df.describe(include="all").T

    # Get null value counts and percentages
    null_counts = df.isna().sum().reset_index()
    null_counts.columns = ["Column Name", "Null Count"]
    null_percent = (df.isna().mean() * 100).reset_index()
    null_percent.columns = ["Column Name", "Null %"]

    # Merge data types, summary statistics, null counts
    combined_df = data_types.merge(summary_stats, how="left", left_on="Column Name", right_index=True)
    combined_df = combined_df.merge(null_counts, how="left", on="Column Name")
    combined_df = combined_df.merge(null_percent, how="left", on="Column Name")

    # Display the combined DataFrame
    st.dataframe(
        combined_df,hide_index=True,
        height=int(35.2 * (combined_df.shape[0] + 1)),use_container_width=True
    )


############################## Visualizer ##############################

def visuals(df):
    fig = sns.pairplot(df)
    st.pyplot(fig)

######################## Custom Visualizer ##############################
def custom_visuals(df):
    st.write("Custom Visuals coming soon!")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if selected_option == "Sample Data":
        sample_data(df)  
    elif selected_option == "Analysis":
        analysis(df)
    elif selected_option == "Visuals":
        visuals(df)
    elif selected_option == "Custom Visuals":
        custom_visuals(df)
else:
    st.write("Please upload a CSV file to get started.")
