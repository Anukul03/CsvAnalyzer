import pandas as pd
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

    # Merge data types, summary statistics, null counts, and null percentages
    combined_df = data_types.merge(summary_stats, how="left", left_on="Column Name", right_index=True)
    combined_df = combined_df.merge(null_counts, how="left", on="Column Name")
    combined_df = combined_df.merge(null_percent, how="left", on="Column Name")

    # Display the combined DataFrame
    st.dataframe(
        combined_df,hide_index=True,
        height=int(35.2 * (combined_df.shape[0] + 1)),use_container_width=True
    )

