import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer


############################# Page Config #################################
st.set_page_config(page_title=None, page_icon="images/favicon.png", layout="wide", initial_sidebar_state="auto")

st.title("CSV Data Analyser")

############################# Sidebar #################################

sidebar = st.sidebar

sidebar.title("File Uploader")

uploaded_file = sidebar.file_uploader("Upload a CSV file", type=["csv"], key="123")

sidebar.subheader("Page")

selected_option = sidebar.selectbox("Select Page", ["Sample Data", "Analysis", "Visuals", "Custom Visuals", "Clean & Save"])

############################# INFO #################################
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    cols, (nrows, ncols) = data.columns, (data.shape)
    obj_cols = data.select_dtypes(include='object').columns.tolist()
    nobj_cols = data.select_dtypes(include=["int64", "float64"], exclude='object').columns.tolist()

############################### Main ##################################
def sample_data(df):
    st.subheader("Top 5 Rows")
    st.dataframe(df.head(), hide_index=True, use_container_width=True)

    st.subheader("Bottom 5 Rows")
    st.dataframe(df.tail(), hide_index=True, use_container_width=True)

    st.subheader("Random Sample Data")
    samp = st.selectbox("Select Sample Count", [5,10,15,20,50])
    st.dataframe(df.sample(samp),height=int(35.2 * (df.sample(samp).shape[0] + 1)), hide_index=True, use_container_width=True)

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
        combined_df,hide_index=True,use_container_width=True,
        height=int(35.2 * (combined_df.shape[0] + 1))
    )





############################## Visualizer ##############################

def visuals(df):
    
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    st.subheader("Numerical Column Distributions")
    if not numerical_columns:
        st.warning("No numerical columns found for visualization.")
        return
    
    legend_column = None
    if categorical_columns:
        legend_column = st.selectbox("Select a categorical column for legend (optional)", [None] + categorical_columns)
    
    hue_param = df[legend_column].astype(str) if legend_column else None
    
    for col in numerical_columns:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fig = px.histogram(df, x=col, color=legend_column, nbins=30, title=f"Histogram of {col}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x=col, color=legend_column, title=f"Box Plot of {col}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            if len(numerical_columns) > 1:
                next_col = numerical_columns[(numerical_columns.index(col) + 1) % len(numerical_columns)]
                fig = px.scatter(df, x=col, y=next_col, color=legend_column, title=f"Scatter Plot: {col} vs {next_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        with col4:
                fig = px.histogram(df, x=col, color=legend_column, title=f"Count Plot of {col}", category_orders={col: df[col].value_counts().index.tolist()})
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Categorical Column Distributions")
    col1, col2, col3 = st.columns(3)
    for i, col in enumerate(categorical_columns):
        with [col1, col2, col3][i % 3]:
            fig = px.histogram(df, x=col, color=legend_column, title=f"Count Plot of {col}", category_orders={col: df[col].value_counts().index.tolist()})
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Correlation Heatmap")
    fig = px.imshow(
        df[numerical_columns].corr(), 
        text_auto=True, 
        color_continuous_scale='RdBu', 
        title="Correlation Heatmap",
        height=2000,  # Increase height
        width=2000   # Increase width
    )
    st.plotly_chart(fig, use_container_width=True)

    
######################## Custom Visualizer ##############################
def custom_visuals(df):
    st.subheader("Create Your Custom Visualization")
    
    # Column selection
    chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Box Plot", "Violin Plot", "Scatter Plot", "Histogram", "Line Plot", "Pair Plot", "Heatmap"])
    col1, col2 = st.columns(2)
    with col1:
        x_column = st.selectbox("Select X-axis Column", df.columns)
    with col2:
        y_column = st.selectbox("Select Y-axis Column (if applicable)", [None] + df.columns.tolist())
    
    # Additional customization options
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    hue_column = st.selectbox("Select Hue (Grouping) Column", [None] + categorical_columns)
    
    # Chart rendering
    fig = None
    if chart_type == "Bar Chart" and y_column:
        fig = px.bar(df, x=x_column, y=y_column, color=hue_column, title=f"Bar Chart: {x_column} vs {y_column}")
    elif chart_type == "Box Plot" and y_column:
        fig = px.box(df, x=x_column, y=y_column, color=hue_column, title=f"Box Plot: {x_column} vs {y_column}")
    elif chart_type == "Violin Plot" and y_column:
        fig = px.violin(df, x=x_column, y=y_column, color=hue_column, title=f"Violin Plot: {x_column} vs {y_column}")
    elif chart_type == "Scatter Plot" and y_column:
        fig = px.scatter(df, x=x_column, y=y_column, color=hue_column, title=f"Scatter Plot: {x_column} vs {y_column}")
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_column, color=hue_column, nbins=30, title=f"Histogram of {x_column}")
    elif chart_type == "Line Plot" and y_column:
        fig = px.line(df, x=x_column, y=y_column, color=hue_column, title=f"Line Plot: {x_column} over {y_column}")
    elif chart_type == "Pair Plot" and len(numerical_columns) > 1:
        fig = sns.pairplot(df[numerical_columns], hue=hue_column)
        st.pyplot(fig)
    elif chart_type == "Heatmap" and len(numerical_columns) > 1:
        fig = px.imshow(df[numerical_columns].corr(), text_auto=True, color_continuous_scale='RdBu', title="Correlation Heatmap")
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)

############################# Clean & Save ##################################

# Function to clean categorical columns
def clean_categorical(df):
    # Fill missing values in categorical columns with 'Unknown'
    cat_columns = df.select_dtypes(include=['object']).columns
    for col in cat_columns:
        df[col] = df[col].fillna('Unknown')  # Replace NaN with 'Unknown'
        df[col] = df[col].str.lower().str.strip()  # Standardize text (lowercase and remove spaces)
    
    # Encode categorical columns using LabelEncoder
    le = LabelEncoder()
    for col in cat_columns:
        df[col] = le.fit_transform(df[col])  # Convert categorical to numerical labels
    return df

# Function to clean numerical columns
def clean_numerical(df):
    # Handle missing values in numerical columns by replacing them with the median
    num_columns = df.select_dtypes(include=[np.number]).columns
    for col in num_columns:
        imputer = SimpleImputer(strategy='median')
        df[col] = imputer.fit_transform(df[[col]])  # Fill missing values with the median
    
    # Detect and remove outliers using IQR method
    for col in num_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    scaler = StandardScaler()  
    for col in num_columns:
        df[col] = scaler.fit_transform(df[[col]])  
    
    return df


def main(df):
    st.subheader("Raw Dataset")
    st.write(df.head())

    # Cleaning process
    st.subheader("Cleaning Categorical Columns")
    df = clean_categorical(df)
    st.write("Categorical columns have been cleaned.")

    st.subheader("Cleaning Numerical Columns")
    df = clean_numerical(df)
    st.write("Numerical columns have been cleaned.")

    # Display cleaned data
    st.subheader("Cleaned Dataset")
    st.write(df.head())

    # Show correlation heatmap (optional)
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(20, 15)) 
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


    # Save the cleaned dataset
    st.download_button(
        label="Download Cleaned Dataset",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

############################# Main ##################################
if __name__ == "__main__":
    if uploaded_file is not None:
        if selected_option == "Sample Data":
            sample_data(data)  
        elif selected_option == "Analysis":
            analysis(data)
        elif selected_option == "Visuals":
            visuals(data)
        elif selected_option == "Custom Visuals":
            custom_visuals(data)
        elif selected_option == "Clean & Save":
            main(data)
        
    else: 
        st.write("Please upload a CSV file to get started.")


