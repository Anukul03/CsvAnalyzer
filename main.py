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
    st.subheader("Create Your Custom Catplot Visualization")
    
    numerical_columns = df.columns.tolist()
    categorical_columns = df.columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        kind = st.selectbox(
            "Catplot Kind",
            ["strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"]
        )
        
        x = st.selectbox("X-axis", categorical_columns + numerical_columns)
        height = st.slider("Height", 2, 10, 5)
        hue = st.selectbox("Hue (optional)", [None] + categorical_columns)
        row = st.selectbox("Row Facet", [None] + categorical_columns)
        legend = st.selectbox("Legend", ["auto", "brief", "full", False])
        legend_out = st.checkbox("Legend Outside", True)

    with col2:
        aggfunc = st.selectbox("Aggregation", ["mean", "median", "sum", "min", "max",  None])
        y = st.selectbox("Y-axis", [None] + numerical_columns)
        aspect = st.slider("Aspect Ratio", 0.5, 3.0, 1.0)
        palette = None
        if hue:
            palette = st.selectbox(
                "Palette",
                ["deep", "muted", "pastel", "bright", "dark", "colorblind"]
            )
        col = st.selectbox("Column Facet", [None] + categorical_columns)
        orient = st.selectbox("Orientation", ["v", "h", "x", "y", None])
        margin_titles = st.checkbox("Margin Titles", False)
        

    estimator_map = {
        "mean": "mean",
        "median": "median",
        "sum": sum,
        "min": min,
        "max": max,
        # "count":"count",
        # "distinct count": "nunique"
    }

    catplot_args = {
        "data": df,
        "kind": kind,
        "x": x,
        "y": y,
        "hue": hue,
        "row": row,
        "col": col,
        "height": height,
        "aspect": aspect,
        "orient": orient,
        "legend": legend,
        "legend_out": legend_out,
        "margin_titles": margin_titles
    }

    if kind in ["bar", "point"] and aggfunc:
        catplot_args["estimator"] = estimator_map[aggfunc]

    if hue and palette:
        catplot_args["palette"] = palette

    catplot_args = {k: v for k, v in catplot_args.items() if v is not None}

    if st.button("Submit"):
        fig = sns.catplot(**catplot_args).tight_layout()
        st.pyplot(fig)

############################# Clean & Save ##################################

def clean_categorical(df):
    cat_columns = df.select_dtypes(include=['object']).columns
    for col in cat_columns:
        df[col] = df[col].fillna('Unknown')  
        df[col] = df[col].str.lower().str.strip()  
    
    le = LabelEncoder()
    for col in cat_columns:
        df[col] = le.fit_transform(df[col])  
    return df

def clean_numerical(df):
    num_columns = df.select_dtypes(include=[np.number]).columns
    for col in num_columns:
        imputer = SimpleImputer(strategy='median')
        df[col] = imputer.fit_transform(df[[col]])  
    
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


