import streamlit as st
import io
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go

# Streamlit page configuration
st.set_page_config(page_title="Heart to Say", page_icon=":heartbeat:")

# Function to load and resize image
def resize_image(image, width):
    # Calculate new height to maintain aspect ratio
    aspect_ratio = image.height / image.width
    new_height = int(width * aspect_ratio)
    return image.resize((width, new_height))

def show_login_image():
    image_path_2 = "heart_to_say.png"
    image_2 = Image.open(image_path_2)  # Load image
    resized_image_2 = resize_image(image_2, width=300)  # Resize image
    st.image(resized_image_2, caption="", use_column_width=False)

# Login function
def login(username=None, password=None):
    # Successful login regardless of what is input
    st.session_state["logged_in"] = True

# Login page
def login_page():

    show_login_image()  
    st.title("The Global Pioneer in Cardiac Language Translation")
    st.markdown("---")

    with st.container():
        st.subheader("Login to Your Account")
        st.write("Please choose your preferred login method")

        if st.button("Login with Username & Password"):
            st.session_state["login_method"] = "username_password"

        if st.button("Login with QR Code"):
            st.session_state["login_method"] = "qr_code"

    # Display different login interfaces based on the selection
    if "login_method" in st.session_state:
        if st.session_state["login_method"] == "username_password":
            show_username_password_login()
        elif st.session_state["login_method"] == "qr_code":
            show_qr_code_login()

def show_username_password_login():
    with st.container():
        st.subheader("Please enter your credentials")

        # Create a form layout
        with st.form(key="login_form"):
            col1, col2 = st.columns([2, 1])

            with col1:
                username = st.text_input("Username", key="username_input")
                password = st.text_input("Password", type="password", key="password_input")
                
            # Place the login button in the blank area on the right
            with col2:
                st.write("")  # Blank area
                st.write("")  
                st.image("https://www.watchman.com/en-us/home/_jcr_content/root/container/container/container_1543905876/container_copy_96760380/image_copy.coreimg.90.1600.jpeg/1721155726782/0-0-wm-how-wm-works-960x541.jpeg", width=200)
            
            login_button = st.form_submit_button("Login")

            if login_button:
                login(username, password)
                if st.session_state.get("logged_in", False):
                    st.success("Login successfully!")
                    st.session_state["show_dashboard_button"] = True

        if st.session_state.get("show_dashboard_button", False):
            if st.button("Proceed to Dashboard"):
                st.session_state["show_dashboard_button"] = False
                st.session_state["logged_in"] = True  # Ensure that it is marked as logged in


def show_qr_code_login():
    with st.container():
        # Display the QR code on the right for logging in
        image_path = "qrcode_heat_to_say.png"
        image = Image.open(image_path)# Load image
        # Resize image to a specific width
        resized_image = resize_image(image, width=100)  # Set the desired width
        # Display the resized image
        st.image(resized_image, caption="Please scan the QR code to login",use_column_width=False)

# Main function to handle the navigation between steps
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login_page()
    else:
        st.session_state["page"] = "dashboard"
        show_dashboard()

if __name__ == "__main__":
    # Initialize session state variables if they don't exist
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'show_dashboard_button' not in st.session_state:
        st.session_state['show_dashboard_button'] = False
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'

# Logout function
def logout():
    # Reset session state and automatically refresh by clearing the logged_in flag
    st.session_state["logged_in"] = False
    st.session_state["show_dashboard_button"] = False
    st.session_state["page"] = "login"

def show_contact_us():
    st.subheader("📞 Contact Us")
    st.markdown("""
    For any inquiries or support, please reach out to us at:
    
    - **Email**: [heart_to_say_team@dsv.su.se](mailto:heart_to_say_team@dsv.su.se)  
    - **Phone**: +46 123456789
    **Group Members**:  
    - Ifani Pinto Nada  
    - Mahmoud Elachi  
    - Nan Jiang  
    - Sahid Hasan Rahimm  
    - Zhao Chen  

    **Data Resource**:  
    [Heart Failure Clinical Data](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)  
   
    **Problem Description**:
    The Heart to Say project aims to build a web-based medical dashboard that supports physicians to predict the risk of mortality due to heart failure. Physicians will be able to reassess treatment plans, explore alternative therapies, and closely monitor patients to help mitigate the risk of mortality. Prescriptive analytics will be used on patient data to help physicians identify specific factors contributing to elevated mortality risk. Thus, it will provide recommendations based on existing medical guidelines to guide in clinical decision-making on an individual basis for prevention and/or mitigation of mortality due to heart failure.           
    
    **Design Process**: 
    1. Team Rules: Document, Paper prototype
    2. Project Charter: Document, Digital prototype and Preprocessing dataset
    3. Text Mining: Jupyter notebook and Report
    4. Project Delivery: Web medical dashboard, Video showcase and Final project report.      
                
    **References**:  
    1. Chicco D, Jurman G. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Med Inform Decis Mak. 2020 Feb 3;20(1):16.  
    2. Kaggle. Heart Failure Prediction [Internet]. San Francisco, CA: Kaggle; [date unknown]. [cited 2024 Sep 11]. Available from: https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data  

    Got some thoughts or suggestions? Don't hesitate to reach out to us. We'd love to hear from you! 
    """)

    st.markdown("---")  # Add a horizontal line for better separation

def show_dashboard():
    st.title("Medical Data Analysis Dashboard")

    if st.sidebar.button("Logout"):
        logout()

    search_query = st.text_input("Search", "")

    with st.sidebar:
        image_path = "heart_to_say.png"
        image = Image.open(image_path)
        resized_image = resize_image(image, width=300)
        st.image(resized_image, use_column_width=True)

        st.header("Upload Your Dataset")

         # File uploader logic
        uploaded_file = st.file_uploader("Choose a CSV file (or skip to use default dataset)", type="csv")
        if uploaded_file is not None:
            st.session_state["uploaded_file"] = uploaded_file
            st.session_state["use_default"] = False
            st.success("File uploaded successfully!")

            if "preprocessing_needed" not in st.session_state:
                st.subheader("Would you like to perform Data Preprocessing before analysis?")
                preprocessing_choice = st.radio(
                    "Select an option:", 
                    ("Yes, I need", "No, my dataset is ready to be analyzed")
                )

                if preprocessing_choice == "Yes, I need":
                    st.session_state["preprocessing_needed"] = True
                    st.sidebar.warning("Please click 'Save and Proceed' after preprocessing!")
                    try:
                        df = pd.read_csv(uploaded_file)
                        show_preprocessing(df) 
                    except pd.errors.EmptyDataError:
                        st.error("The uploaded file is empty. Please upload a valid CSV file.")
                else:
                    st.session_state["preprocessing_needed"] = False
                    try:
                        df = pd.read_csv(uploaded_file)
                        show_main_options(df, search_query)
                    except pd.errors.EmptyDataError:
                        st.error("The uploaded file is empty. Please upload a valid CSV file.")
        else:
            if st.button("Use Default Dataset"):
                st.session_state["use_default"] = True
                st.success("Using the default dataset!")

    # Handle session state for uploaded or default file
    if "uploaded_file" in st.session_state:
        try:
            df = pd.read_csv(st.session_state["uploaded_file"])
            show_main_options(df, search_query)
        except pd.errors.EmptyDataError:
            st.error('There is no file to waif for analysing. Please click "Save and Proceed" after preprocessing!')
    elif "use_default" in st.session_state and st.session_state["use_default"]:
        try:
            df = pd.read_csv("d:/KI/project management_SU/PROHI-dashboard-class-exercise/heart_failure_clinical_records_dataset.csv")
            show_main_options(df, search_query)
        except pd.errors.EmptyDataError:
            st.error("The default dataset could not be loaded. Please check the file path.")
    else:
        st.error("No file uploaded or selected for use. Please upload your dataset.")


def show_main_options(df, search_query):
    st.markdown("---")
    with st.sidebar:
        st.subheader(":guide_dog: Navigation")
        option = st.radio("Select an option:", ["Data Overview", "Exploratory Data Analysis", "Data Modeling","Contact Us"])

    # Filter data based on search query
    if search_query:
        filtered_df = df[df.apply(lambda row: search_query.lower() in row.to_string().lower(), axis=1)]
    else:
        filtered_df = df

    if option == "Data Overview":
        show_data_overview(filtered_df)
    elif option == "Exploratory Data Analysis":
        show_eda(filtered_df)
    elif option == "Data Modeling":
        show_modeling(filtered_df)
    elif option == "Contact Us":
        show_contact_us()
        
def show_preprocessing(df):
    st.subheader("Data Preprocessing Steps")

    # 1. Check Missing Values
    st.markdown(" 1. Check Missing Values")
    df.fillna(df.median(numeric_only=True), inplace=True)
    st.success("Missing values filled with median!")

    # 2. Encode Categorical Variables
    st.markdown(" 2. Encode Categorical Variables")
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_columns:
        if st.button(f"Encode {col}"):
            df[col] = pd.Categorical(df[col]).codes
        st.success(f"Encoded {col} successfully!")
    st.success("Encoded Categorical Variables successfully!")

    # 3. Transform DataFrame into Numpy Array
    st.markdown(" 3. Transform DataFrame into Numpy Array")
    np_array = df.to_numpy()
    st.success("DataFrame transformed to Numpy array!")

    # Save and Proceed Button
    st.markdown("---")
    if st.button("Save and Proceed"):
        st.session_state["processed_data"] = df
        st.success("Data preprocessing complete! Proceeding to main options...")
        show_main_options(df, "")  # Return to main options after preprocessing



def show_data_overview(df):
    st.subheader("Dataset Overview")
    
    # dataset basic info
    total_records = len(df)
    positive_cases = df['DEATH_EVENT'].value_counts().get(1, 0)
    negative_cases = df['DEATH_EVENT'].value_counts().get(0, 0)
    missing_values = df.isnull().sum().sum()
    total_features = df.shape[1]

    col1, col2, col3, col4 = st.columns(4)

    # card style
    with col1:
        st.markdown(f"""
            <div style="background-color: #1abc9c; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h3>Total Records</h3>
                <p style="font-size: 24px; font-weight: bold;">{total_records}</p>
                <p>This represents the total number of patient records in the dataset.</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background-color: #e74c3c; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h3>Positive Cases</h3>
                <p style="font-size: 24px; font-weight: bold;">{positive_cases}</p>
                <p>Number of patients who experienced a death event (1).</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div style="background-color: #3498db; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h3>Negative Cases</h3>
                <p style="font-size: 24px; font-weight: bold;">{negative_cases}</p>
                <p>Number of patients who did not experience a death event (0).</p>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div style="background-color: #9b59b6; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h3>Missing Values</h3>
                <p style="font-size: 24px; font-weight: bold;">{missing_values}</p>
                <p>Indicates how many data entries are missing across all features.</p
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("### Sample of the Dataset")
    st.write("Here are the first 15 records from the dataset for your review:")
    st.dataframe(df.head(15))

    st.write("")
    st.write("### Data Types and Missing Values")
    st.write("This table provides information on each feature, including its data type and how many missing values there are:")
    data_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    st.dataframe(data_info)

    st.write("")
    st.write("### Statistical Summary")
    st.write("The statistical summary includes key metrics like mean, standard deviation, min, and max for numerical features:")
    st.dataframe(df.describe())

    st.write("")
    st.write("### Distribution of Target Variable")
    fig = px.pie(values=df['DEATH_EVENT'].value_counts(), names=['Negative (0)', 'Positive (1)'],
                 )
    st.plotly_chart(fig, use_container_width=True)

    # Distribution of Selected Features
    st.write("")
    st.write("### Distribution of Selected Feature")
    st.write("Select a feature from the dataset to visualize its distribution. The histogram will color-code the data based on the DEATH_EVENT status.")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'DEATH_EVENT' in numeric_columns:
        numeric_columns.remove('DEATH_EVENT')
    
    selected_column = st.selectbox("Select a feature to visualize", numeric_columns)

    fig = px.histogram(df, x=selected_column, color='DEATH_EVENT', barmode='overlay',
                       title=f'Distribution of {selected_column}')
    st.plotly_chart(fig, use_container_width=True)


def show_eda(df):
    st.subheader("Exploratory Data Analysis ")
    st.write("This section provides a brief overview of the dataset, including the total number of records and the features available.")
    st.markdown("---")

    # EDA 
    eda_option = st.selectbox("Choose analysis option:", [ "Correlation Coefficient","Basic Feature Relationships", "Clustering Analysis"])

    if eda_option == "Correlation Coefficient":
        show_correlation(df)
    elif eda_option == "Basic Feature Relationships":
        basic_feature_relationships(df)
    elif eda_option == "Clustering Analysis":
        show_clustering_analysis(df)

def basic_feature_relationships(df):

    col1, col2 = st.columns([4, 1])

    with col1:

        chart_type = st.radio("Select Chart Type:", ["Line Plot", "Box Plot","KDE Plot"])
        
        if chart_type == "Line Plot":

            x_axis = st.selectbox('Select X-axis feature:', df.columns.tolist())
            y_axis = st.selectbox('Select Y-axis feature:', df.columns.tolist())
            
            st.write("### Line Plot Visualization")
            st.write("A line plot will show the relationship between two (continuous) features. Please select the X and Y axes.")
            
            if x_axis == y_axis:
                st.warning("Please select different features for X-axis and Y-axis.")
            else:
                st.write(f"Selected Features: **X-axis:** {x_axis}, **Y-axis:** {y_axis}")

                # Feature Stats
                st.subheader("Feature Stats")
                stats_dict = {
                    f"{x_axis}": df[x_axis].describe(),
                    f"{y_axis}": df[y_axis].describe()
                }
                stats_df = pd.concat(stats_dict, axis=1).T
                st.dataframe(stats_df)


                st.markdown("<br>"*1, unsafe_allow_html=True)

                st.subheader(f"Line Plot: {x_axis} vs {y_axis}")
                df_grouped = df.groupby(x_axis)[y_axis].mean().reset_index()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_grouped[x_axis],
                    y=df_grouped[y_axis],
                    mode='lines+markers',
                    name='Line Plot',
                    line=dict(color='blue')
                ))

                fig.update_layout(
                    xaxis_title=x_axis,
                    yaxis_title=y_axis,
                    width=1000,
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=50)
                )
                st.plotly_chart(fig)

        elif chart_type == "Box Plot":

            x_axis = st.selectbox('Select X-axis feature:', df.columns.tolist())
            y_axis = st.selectbox('Select Y-axis feature:', df.columns.tolist())
            color_feature = st.selectbox('Select feature for color grouping:', df.columns.tolist())
            st.write("### Box Plot Visualization")
            st.write("A box plot is useful for displaying the distribution of a dataset based on a continuous variable, grouped by another variable.")
            if x_axis and y_axis and color_feature:
                st.write(f"Selected Features: **X-axis:** {x_axis}, **Y-axis:** {y_axis}, **Color:** {color_feature}")

                # Feature Stats
                st.subheader("Feature Stats")
                stats_dict = {
                    f"{x_axis}": df[x_axis].describe(),
                    f"{y_axis}": df[y_axis].describe(),
                    f"{color_feature}": df[color_feature].describe()
                }
                stats_df = pd.concat(stats_dict, axis=1).T
                st.dataframe(stats_df)


                st.markdown("<br>"*3, unsafe_allow_html=True)

            st.subheader(f"Box Plot: {x_axis} vs {y_axis} grouped by {color_feature.capitalize()}")
            fig = px.box(df, x=x_axis, y=y_axis, color=color_feature,
                             labels={x_axis: x_axis, y_axis: y_axis},
                             )
            fig.update_layout(
                    width=1000,  # Adjust width to match selectbox
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=50)  # Adjust top margin here
                )
            st.plotly_chart(fig)

        elif chart_type == "KDE Plot":

            x_axis = st.selectbox('Select X-axis feature:', df.columns.tolist())
            y_axis = st.selectbox('Select Y-axis feature:', df.columns.tolist())
            color_feature = st.selectbox('Select feature for color grouping:', df.columns.tolist())
            st.write("### KDE Plot Visualization")
            st.write("A Kernel Density Estimate (KDE) plot is useful for visualizing the probability density of a continuous variable.")
            if x_axis and y_axis and color_feature:
                st.write(f"Selected Features: **X-axis:** {x_axis}, **Y-axis:** {y_axis}, **Color:** {color_feature}")

                # Feature Stats
                st.subheader("Feature Stats")
                stats_dict = {
                    f"{x_axis}": df[x_axis].describe(),
                    f"{y_axis}": df[y_axis].describe(),
                    f"{color_feature}": df[color_feature].describe()
                }
                stats_df = pd.concat(stats_dict, axis=1).T
                st.dataframe(stats_df)


                st.markdown("<br>"*3, unsafe_allow_html=True)

            st.subheader(f"KDE Plot: {x_axis} vs {y_axis} grouped by {color_feature.capitalize()}")
            custom_colors = px.colors.qualitative.Set1
            fig = px.density_contour(
                    df, x=x_axis, y=y_axis, color=color_feature,
                    marginal_x="histogram", marginal_y="histogram",  # Optional histograms on margins
                    color_discrete_sequence=custom_colors
                )

            fig.update_layout(
                    width=1000,  # Adjust width to match selectbox
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=50)  # Adjust top margin here
                )
            st.plotly_chart(fig)

    with col2:

        st.subheader("Data Overview")
        st.write("**Total Records:** ", len(df))
        st.write("**Features:**", len(df.columns))
        st.write("**Columns:**", ', '.join(df.columns))
        

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def handle_highly_correlated_features(df, threshold=0.85):

    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    st.write(f"Highly correlated features (correlation > {threshold}): {to_drop}")

    df_reduced = df.drop(to_drop, axis=1)
    return df_reduced

def show_clustering_analysis(df):
    st.subheader("Clustering Analysis")
    st.markdown("In this section, you can select features for clustering and explore patterns in your data.")


    remove_correlated = st.checkbox("Remove Highly Correlated Features")
    st.markdown("""
    When enabled, features that have a correlation greater than the specified threshold will be removed. 
    This can help reduce redundancy and improve the clustering results.
    """)


    if remove_correlated:
        corr_threshold = st.slider("Correlation threshold:", 0.0, 1.0, 0.85)
        st.markdown(f"Selected correlation threshold: {corr_threshold}. Features with a correlation higher than this will be removed.")

        corr_matrix = df.corr()
        to_remove = set()


        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                    colname = corr_matrix.columns[i]
                    to_remove.add(colname)


        if to_remove:
            st.write(f"Removed features due to high correlation (correlation > {corr_threshold}): {to_remove}")
        else:
            st.write(f"No features removed due to high correlation (correlation > {corr_threshold}).")


    feature1 = st.selectbox('Select First Feature for Clustering:', df.columns.tolist())
    feature2 = st.selectbox('Select Second Feature for Clustering:', df.columns.tolist())


    if feature1 == feature2:
        st.warning("Please select different features for clustering.")
    else:

        additional_features = st.multiselect("Select Additional Features for Clustering:", df.columns.tolist(), default=[])

        selected_features = [feature1, feature2] + additional_features
        if len(selected_features) > 0:
            selected_df = df[selected_features]


            if remove_correlated:
                selected_df = selected_df.drop(columns=to_remove.intersection(selected_features), errors='ignore')

            n_clusters = st.slider("Select number of clusters:", min_value=2, max_value=10, value=3)


            kmeans = KMeans(n_clusters=n_clusters)
            df['Cluster'] = kmeans.fit_predict(selected_df)


            if selected_df.shape[1] >= 2:
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(selected_df)
                fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1], color=df['Cluster'].astype(str), title="K-Means Clustering")
                fig.update_layout(title_x=0.5)
                st.plotly_chart(fig)
            else:
                st.warning("Please select at least two different features for clustering.")


            st.subheader(f"Clustering Results with {n_clusters} Clusters")

            selected_cluster = st.selectbox("Select a Cluster to View Details:", range(n_clusters))

            st.write(f"Statistics for Cluster {selected_cluster}")
            cluster_data = df[df['Cluster'] == selected_cluster][selected_features]
            st.write(cluster_data.describe())


            st.markdown("""
            The table above shows the statistical summary of the selected features for each cluster. 
            This includes metrics such as count, mean, standard deviation, minimum, and maximum values.
            Understanding these statistics can help you identify the characteristics of each cluster and how they differ from one another.
            """)
        else:
            st.warning("Please select at least one feature for clustering.")



import seaborn as sns
import matplotlib.pyplot as plt

def show_correlation(df):
    st.subheader("Correlation Matrix")

    selected_features = st.multiselect("Select features for correlation analysis:", df.columns.tolist(), default=df.columns.tolist())

    if len(selected_features) > 0:

        corr_df = df[selected_features].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}, ax=ax)

        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)

        st.pyplot(fig)

        # Detailed Interpretation
        st.markdown("### Interpretation of the Correlation Matrix")
        st.markdown("""
            The correlation matrix shows the pairwise relationships between the selected features.
            
            #### Key Points to Understand:
            - **Values close to 1**: Strong positive correlation. This indicates that as one feature increases, the other feature tends to increase as well. For example, if `feature A` and `feature B` have a correlation of 0.9, when `feature A` increases, `feature B` likely increases.
            - **Values close to -1**: Strong negative correlation. This suggests that as one feature increases, the other decreases. For instance, a correlation of -0.8 between `feature X` and `feature Y` suggests that when `feature X` increases, `feature Y` decreases.
            - **Values close to 0**: Little or no linear relationship. Features with correlations near zero are not linearly related, meaning one does not predict or affect the other in a linear way.
            
            #### How to Use This Information:
            - **Identifying Predictors**: High correlation values (either positive or negative) suggest that one feature could be a good predictor for another. This is particularly useful in feature selection for predictive modeling, where multicollinearity should be considered.
            - **Dropping Redundant Features**: If two features are highly correlated, one may be redundant and could be dropped from your model to avoid multicollinearity.
            - **Feature Engineering**: Insights from the correlation matrix can help in designing new features by combining or transforming existing ones (e.g., creating interaction terms or scaling features).
        """)

        st.markdown("""
            #### Next Steps:
            After analyzing the correlation matrix, you may consider:
            - **Further Statistical Analysis**: Performing regression or clustering with highly correlated features.
            - **Visual Exploration**: Plotting scatter plots for pairs of highly correlated features to visualize their relationships.
        """)
    else:
        st.warning("Please select at least one feature for correlation analysis.")

import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import plotly.express as px
import io
import pickle
import joblib
import shap
import matplotlib.pyplot as plt

def load_model(uploaded_file): 
    return joblib.load(uploaded_file)

def show_modeling(df):
    with st.sidebar:
        option = st.radio("Select an option:", ["Upload your model file", "Use the provided model"])

    if option == "Upload your model file":
        upload_your_model_file(df)
    elif option == "Use the provided model":
        use_the_provided_model(df)

def upload_your_model_file(df):
    # Select features
    st.sidebar.subheader("Feature Selection")
    all_features = df.columns.tolist()
    all_features.remove("DEATH_EVENT")
    selected_features = st.sidebar.multiselect(
        "Select features for modeling:",
        options=all_features,
        default=all_features
    )
    
    if not selected_features:
        st.error("Please select at least one feature.")
        return

    # Preparing data
    X = df[selected_features]
    y = df["DEATH_EVENT"]

    # Choose scaling method
    st.sidebar.subheader("Select Scaling Method")
    scaling_method = st.sidebar.selectbox(
        "Choose a scaling method:",
        ("StandardScaler", "MinMaxScaler", "RobustScaler")
    )

    # Scaling based on user's choice
    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
    elif scaling_method == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaling_method == "RobustScaler":
        scaler = RobustScaler()

    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features)

    # Upload trained model
    st.sidebar.subheader("Upload Pre-trained Model")
    uploaded_file = st.sidebar.file_uploader("Choose the model file", type=["pkl", "sav"])

    if uploaded_file is not None:
        model = load_model(uploaded_file)
        st.success("Model loaded successfully!")

        # Make predictions
        predictions = model.predict(X_scaled)
        predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
        
        df = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)


        # User selection for evaluation or SHAP
        analysis_option = st.sidebar.selectbox(
            "Select Analysis Type:",
            ("View Predictions", "Model Performance Metrics", "SHAP Analysis")
        )

        if analysis_option == "Model Performance Metrics":
            st.subheader("Model Performance Metrics")
            
            # Calculate accuracy
            accuracy = accuracy_score(y, predictions)
            st.write(f"**Accuracy:** {accuracy:.4f}")
        
           
            # Classification Report
            report = classification_report(y, predictions, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.write("### Classification Report")
            st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
            st.write("""
            The classification report provides a comprehensive overview of the model's performance across different classes. Here are the key metrics included:

            - **Precision**: This metric indicates the accuracy of positive predictions. It is calculated as the ratio of true positive predictions to the total predicted positives. High precision means that most positive predictions are correct.

            - **Recall**: Also known as sensitivity or true positive rate, recall measures the ability of the model to identify all relevant instances. It is calculated as the ratio of true positive predictions to the total actual positives. High recall indicates that the model is good at capturing positive cases.

            - **F1-Score**: This is the harmonic mean of precision and recall, providing a single score that balances both metrics. It is particularly useful when you want to find an optimal balance between precision and recall. A high F1-score indicates a good balance between precision and recall.

            - **Support**: This indicates the number of actual occurrences of each class in the specified dataset. It helps in understanding the distribution of classes and the relevance of other metrics.

            Overall, the classification report is crucial for evaluating the model’s effectiveness, especially in cases where class imbalance exists. By analyzing these metrics, we can gain insights into areas for improvement and make informed decisions about model optimization.
            """)
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            conf_matrix = confusion_matrix(y, predictions)
            fig = px.imshow(
                conf_matrix, 
                text_auto=True, 
                color_continuous_scale="Blues",
                labels={'color': 'Count'},
                aspect="auto"
            )
            fig.update_layout(
                xaxis_title='Predicted Label', 
                yaxis_title='True Label',
                title_font=dict(size=20, color='black'),
                xaxis=dict(tickmode='linear'),
                yaxis=dict(tickmode='linear')
            )
            st.plotly_chart(fig, use_container_width=True)
            st.write("""
            The confusion matrix provides a visual representation of the performance of the classification model. It summarizes the counts of true positive, true negative, false positive, and false negative predictions.

            - **True Positive (TP)**: The number of instances correctly predicted as positive.
            - **True Negative (TN)**: The number of instances correctly predicted as negative.
            - **False Positive (FP)**: The number of instances incorrectly predicted as positive (also known as Type I error).
            - **False Negative (FN)**: The number of instances incorrectly predicted as negative (also known as Type II error).

            From the confusion matrix, you can also derive several important metrics:
            - **Accuracy**: \((TP + TN) / (TP + TN + FP + FN)\) - Overall correctness of the model.
            - **Precision**: \(TP / (TP + FP)\) - How many of the predicted positives are actually positive.
            - **Recall**: \(TP / (TP + FN)\) - How many of the actual positives were correctly identified.
            - **F1-Score**: The harmonic mean of precision and recall.

            The confusion matrix is a powerful tool for understanding the strengths and weaknesses of the classification model, helping to identify specific areas where the model may need improvement. By analyzing the counts, we can gain insights into class-specific performance and potential biases in the model.
            """)

        elif analysis_option == "SHAP Analysis":
            st.subheader("SHAP Analysis")
            
            X_scaled_np = np.array(X_scaled)
            y_np = y.values
            X_sample = shap.sample(X_scaled_np, 50)
            class_labels = np.unique(y_np)

            positive_class_index = np.where(class_labels == 1)[0]
            if len(positive_class_index) == 0:
                st.error("There is no positive labels.")
                return
            else:
                positive_class_index = positive_class_index[0]

            # SHAP explainer and shap_values determination
            if model.__class__.__name__ in ["RandomForestClassifier", "DecisionTreeClassifier"]:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_scaled_np)

                if isinstance(shap_values, list):
                    shap_values = shap_values[positive_class_index]

            elif model.__class__.__name__ == "LogisticRegression":
                explainer = shap.LinearExplainer(model, X_sample)
                shap_values = explainer.shap_values(X_scaled_np)

            else:
                explainer = shap.KernelExplainer(model.predict_proba, X_sample)
                shap_values = explainer.shap_values(X_scaled_np)

            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, positive_class_index]
            elif shap_values.ndim == 1:
                shap_values = shap_values.reshape(-1, 1)

            if shap_values.ndim == 2:
                feature_importances = np.abs(shap_values).mean(axis=0)
            else:
                st.error("SHAP values should be 2D.")
                return

            sorted_indices = np.argsort(feature_importances)[::-1]
            sorted_feature_names = [selected_features[i] for i in sorted_indices]

            # SHAP Summary Plot
            shap.summary_plot(shap_values, X_scaled_np, feature_names=sorted_feature_names)
            plt.title("SHAP Summary Plot")
            st.pyplot(plt)

            # Feature Importance from SHAP
            st.subheader("Feature Importance from SHAP")
            shap_importance_df = pd.DataFrame({
                "Feature": sorted_feature_names,
                "Importance": feature_importances[sorted_indices]
            })

            fig = px.bar(
                shap_importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                text="Importance", 
            )
            fig.update_layout(
                yaxis=dict(
                    categoryorder="total ascending",
                    showgrid=True 
                ),
                xaxis=dict(
                    title="Feature Importance",
                    showgrid=True,  
                    gridcolor="lightgray" 
                ),
                font=dict(size=16),  
                plot_bgcolor='white',  
                width=900,  
                height=500  
            )
            fig.update_traces(
                marker_color="#00BFFF",  
                texttemplate="%{text:.4f}", 
                textposition="auto"  
            )
            st.plotly_chart(fig, use_container_width=True)

            # Conclusion on Derived Insights
            st.subheader("Conclusions from SHAP Analysis")
            st.write("""
                The SHAP analysis provides clear insights into how each feature contributes to the model's predictions. 
                Features with higher SHAP values indicate a stronger influence on the predicted outcome. For example, 
                if 'Feature A' consistently shows high importance, it suggests that variations in 'Feature A' significantly 
                affect the likelihood of a patient being classified as high risk. 
                
                By understanding these relationships, we can validate the model's decision-making process and ensure 
                that derived conclusions from the predictions align with domain knowledge and clinical expectations. 
                This transparency is critical for clinical applications, where understanding the underlying reasons for predictions 
                can guide effective interventions and improve patient outcomes.
            """)

        elif analysis_option == "View Predictions":

            st.subheader("Data with Predictions")
            st.dataframe(df)  
            # Ensure patient index is within the valid range
            max_index = len(df) - 1
            patient_index = st.number_input("Enter Patient Index:", min_value=0, max_value=max_index, step=1)

            if 0 <= patient_index <= max_index:
                # Extract prediction for the selected patient index
                pred = df.loc[patient_index, 'Predictions']
                
                # Generate recommendation based on the prediction
                if pred == 1:
                    recommendation = "Patient is at high risk of death. Immediate intervention is advised."
                else:
                    recommendation = "Patient is at low risk of death. Regular monitoring is recommended."

                # Display Patient Information
                st.markdown(f"""
                <div style="
                    background-color: #ffffff;
                    border-radius: 10px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                    padding: 20px;
                    margin-bottom: 20px;
                    color: #333;
                    border: 1px solid #ddd;
                ">   
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 20px;
                    ">
                        <h2 style="
                            margin: 0;
                            font-size: 22px;
                            color: #2c3e50;
                            border-bottom: 2px solid #3498db;
                            width: 150px;
                            padding-bottom: 10px;
                            flex: 1;
                        ">
                            Patient Index: <br>
                            {patient_index}
                        </h2>
                        <h2 style="
                            margin: 0;
                            font-size: 22px;
                            color: #2c3e50;
                            border-bottom: 2px solid #3498db;
                            padding-bottom: 10px;
                            flex: 1;
                            text-align: right;
                        ">
                            Model: <br>
                            {model}
                        </h2>
                    </div>
                    <p style="
                        font-size: 20px;
                        margin: 10px 0;
                        font-weight: bold;
                    ">
                        Prediction: 
                        <span style="
                            font-weight: bold;
                            color: {'#e74c3c' if pred == 1 else '#27ae60'};
                        ">
                            {'High Risk' if pred == 1 else 'Low Risk'}
                        </span>
                    </p>
                    <p style="
                        font-size: 20px;
                        margin: 10px 0;
                        font-weight: bold;
                    ">
                        Recommendation: 
                        <span style="
                            color: #2980b9;
                            background-color: #ecf0f1;
                            border-radius: 5px;
                            padding: 5px 10px;
                        ">
                            {recommendation}
                        </span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("""
            In this section, you can view the model's predictions for individual patients. By selecting a patient index, 
            you can see whether the model has classified the patient as high or low risk based on the available data.
            
            **High Risk Prediction**: The model predicts that the patient is at high risk, suggesting that immediate intervention may be necessary.
            
            **Low Risk Prediction**: The model predicts a low risk for the patient, implying that regular monitoring should suffice for now.
            
            This tool is useful for identifying and understanding individual predictions and how the model assesses risk. Additionally, based on the prediction, a recommendation is generated to guide further actions for the patient.
             """)
            
            # Create an Excel file with patient data and recommendations
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Patient Data', index=False)
                # Write recommendations in a separate sheet
                recommendations_df = pd.DataFrame({
                    'Patient Index': [patient_index],
                    'Prediction': [pred],
                    'Recommendation': [recommendation]
                })
                recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
            st.download_button(label="Download Patient Data and Recommendations as Excel",
                                data=output.getvalue(),
                                file_name=f"{model}_patient_data_and_recommendations.xlsx",
                                mime="application/vnd.ms-excel")
    else:
        st.warning("Please upload a trained model file.")

def use_the_provided_model(df):
    # select features
    st.sidebar.subheader("Feature Selection")
    all_features = df.columns.tolist()
    all_features.remove("DEATH_EVENT")
    selected_features = st.sidebar.multiselect(
        "Select features for modeling:",
        options=all_features,
        default=all_features
    )
    
    if not selected_features:
        st.error("Please select at least one feature.")
        return

    # preparing data
    X = df[selected_features]
    y = df["DEATH_EVENT"]

   # Choose scaling method
    st.sidebar.subheader("Select Scaling Method")
    scaling_method = st.sidebar.selectbox(
        "Choose a scaling method:",
        ("StandardScaler", "MinMaxScaler", "RobustScaler")
    )

    # scaling based on user's choice
    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
    elif scaling_method == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaling_method == "RobustScaler":
        scaler = RobustScaler()

    if scaler is not None:
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=selected_features)
    else:
        X_scaled = X  # No scaling applied

    # choose to show the scaled features box plot
    show_boxplot = st.sidebar.checkbox("Show Box Plot of Scaled Features")

    if show_boxplot:
        # box plot
        st.subheader("Box Plot of Scaled Features")
        fig = px.box(X_scaled)
        fig.update_layout(xaxis_title='Feature', yaxis_title='Scaled Value')
        st.plotly_chart(fig, use_container_width=True)

    # choose a model
    st.sidebar.subheader("Select a Model for Training")
    model_option = st.sidebar.selectbox(
        "Choose a model:",
        ("Logistic Regression", "Decision Tree", "Random Forest", "K-Nearest Neighbors", "Support Vector Machine", "Artificial Neural Network")
    )

    def train_and_evaluate_model(model_name):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=25)

        # select model to train
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(min_samples_split=2, random_state=25)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100)
        elif model_name == "K-Nearest Neighbors":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_name == "Support Vector Machine":
            model = SVC(probability=True)
        elif model_name == "Artificial Neural Network":
            model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=25)
        else:
            st.error("Invalid model selected.")
            return None, None, None, None, None, None

        # train model and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Check if the model supports predict_proba
        try:
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        except KeyError as e:
            st.warning(f"Prediction error: {str(e)}. Model may not support probability prediction.")
            y_prob = None

        # Model Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        return model, accuracy, auc_score, report, conf_matrix, X_test, X_train, y_test, y_train

    def show_model_results(model_name):
        st.markdown(f"""
        ## Model: {model_name}
        """)
        st.markdown("---")
        model, accuracy, auc_score, report, conf_matrix, X_test, X_train, y_test, y_train= train_and_evaluate_model(model_name)

        if model is None:
            st.error("The model could not be trained. Please check the model selection.")
            return None

        # Save the model and predictions to session state
        st.session_state.model = model
        filename = save_model(model, model_name)
        st.success(f"Model saved as {filename}")
        # Save the model and predictions to session state
        st.session_state.model = model

        # Classification Report
        st.subheader("Classification Report")
        st.write("The classification report provides precision, recall, and F1-score for each class.")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

        # Model Performance Metrics
        st.subheader("Model Performance Metrics")
        st.write("These metrics help to understand the performance of the model.")
        
        # Layout for performance metrics with cards
        st.markdown("""
        <style>
        .card-container {
            display: flex;
            justify-content: space-between;
        }
        .card {
            background-color: #f0f0f5; /* Default background color */
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 16px;
            width: 48%; /* Adjust width to fit side-by-side */
        }
        .card-title {
            font-size: 18px;
            font-weight: bold;
            color: #fff; /* White text */
        }
        .card-value {
            font-size: 24px;
            font-weight: bold;
            color: #fff; /* White text */
        }
        .card-red {
            background-color: #e74c3c; /* Red background */
        }
        .card-blue {
            background-color: #3498db; /* Blue background */
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display Accuracy in red card on the left
        st.markdown(f"""
        <div class="card-container">
            <div class="card card-red">
                <div class="card-title">Accuracy</div>
                <div class="card-value">{accuracy:.4f}</div>
            </div>
            <div class="card card-blue">
                <div class="card-title">ROC-AUC Score</div>
                <div class="card-value">{auc_score:.4f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # If ROC-AUC Score is not applicable
        if auc_score == "N/A":
            st.markdown(f"""
            <div class="card-container">
                <div class="card card-red">
                    <div class="card-title">Accuracy</div>
                    <div class="card-value">{accuracy:.4f}</div>
                </div>
                <div class="card card-blue">
                    <div class="card-title">ROC-AUC Score</div>
                    <div class="card-value">Not Applicable</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Specificity Card
        st.markdown("")
        tn = report['0']['support'] if '0' in report else 'N/A'  # True Negative count
        fp = report['1']['support'] if '1' in report else 'N/A'  # False Positive count
        specificity = tn / (tn + fp) if tn != 'N/A' and fp != 'N/A' else 'N/A'
        specificity_card = f"""
        <div style="
            background-color: #00bcd4; 
            color: white; 
            padding: 10px; 
            border-radius: 5px; 
            font-size: 18px; 
            font-weight: bold; 
            text-align: center;
            margin-bottom: 10px;">
            Specificity (SP): {specificity:.4f}
        </div>
        """
        st.markdown(specificity_card, unsafe_allow_html=True)
        
        # Confusion Matrix
        st.markdown("")
        st.subheader("Confusion Matrix")
        st.write("The confusion matrix shows the counts of true vs predicted labels.")
        fig = px.imshow(conf_matrix, text_auto=True, color_continuous_scale="Blues",
                        labels={'color': 'Count'})
        fig.update_layout(xaxis_title='Predicted Label',
                        yaxis_title='True Label')
        st.plotly_chart(fig, use_container_width=True)
        
            # Initialize SHAP
        shap.initjs()

        # SHAP Analysis
        st.subheader("SHAP Analysis")

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        X_sample = shap.sample(X_train, 50)
        y = df["DEATH_EVENT"].values
        class_labels = np.unique(y)


        positive_class_index = np.where(class_labels == 1)[0]
        if len(positive_class_index) == 0:
            st.error("There is no positive labels.")
            return
        else:
            positive_class_index = positive_class_index[0]

        # Determine the appropriate SHAP explainer and shap_values
        if model_name in ["Random Forest", "Decision Tree"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[positive_class_index]  # Only get SHAP values for the positive class

        elif model_name == "Logistic Regression":
            explainer = shap.LinearExplainer(model, X_sample, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_test)

        elif model_name in ["K-Nearest Neighbors", "Support Vector Machine", "Artificial Neural Network"]:
            explainer = shap.KernelExplainer(model.predict_proba, X_sample)
            shap_values = explainer.shap_values(X_test)

        else:
            st.error("SHAP analysis not supported for this model.")
            return

        # Print shapes for debugging
        print("X_test shape:", X_test.shape)
        print("shap_values shape:", shap_values.shape)

        ## Ensure shap_values is 2D (for the positive class)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, positive_class_index]  
        elif shap_values.ndim == 1:
            shap_values = shap_values.reshape(-1, 1)  

    
        if shap_values.ndim == 2:
            feature_importances = np.abs(shap_values).mean(axis=0)
        else:
            st.error("SHAP values should be 2D.")
            return

        # Debugging: print shape of feature_importances
        print("Feature importances shape:", feature_importances.shape)

        sorted_indices = np.argsort(feature_importances)[::-1]

        # Create sorted_feature_names
        if len(selected_features) == X_test.shape[1]:
            sorted_feature_names = [selected_features[i] for i in sorted_indices]
        else:
            st.error("selected_features length does not match X_test features.")
            return

        # SHAP Summary Plot
        shap.summary_plot(shap_values, X_test, feature_names=sorted_feature_names)
        plt.title("SHAP Summary Plot")
        st.pyplot(plt)
        
        sorted_importances = feature_importances[sorted_indices]
        sorted_importances = np.round(sorted_importances, 4)
        shap_importance_df = pd.DataFrame({
                'Feature': sorted_feature_names,
                'Importance': sorted_importances
            })
        
        fig = px.bar(
            shap_importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            text='Importance', 
            title='Feature Importance from SHAP',
            color_discrete_sequence=['#00BFFF']  
        )
        fig.update_layout(
            yaxis=dict(
                categoryorder='total ascending',  
                showgrid=True  
            ),
            xaxis=dict(
                title="SHAP Feature Importance",
                showgrid=True,  
                gridcolor='lightgray'  
            ),
            yaxis_title="Feature",
            font=dict(size=16),  
            plot_bgcolor='white',  
            width=900,  
            height=500  
        )

        st.plotly_chart(fig, use_container_width=True)
      # Save the model to a file
        with open("saved_model.pkl", "wb") as f:
            joblib.dump(model, f)
            st.success("Model has been saved successfully!")  
        return model
        
    def save_model(model, model_name):
        filename = f"{model_name.replace(' ', '_').lower()}_model.pkl"
        with open(filename, 'wb') as file:
            joblib.dump(model, file)
        return filename
    
    # Clear previous model results if model changes
    if 'current_model_option' in st.session_state:
        if st.session_state.current_model_option != model_option:
            st.session_state.df_with_predictions = None
            st.session_state.model = None
            st.session_state.current_model_option = model_option
            st.warning("You've changed the model so you need to train and evaluate the model again.")
    else:
        st.session_state.current_model_option = model_option 
        
    # Train and Evaluate Model
    if st.sidebar.button("Train and Evaluate Model"):
        model = show_model_results(model_option)

        if model:
            # Make predictions with the trained model on the full dataset
            X_full = df[selected_features]
            X_full_scaled = scaler.transform(X_full)
            predictions = model.predict(X_full_scaled)

            # Add predictions to the original DataFrame
            df['Predictions'] = predictions
            st.session_state.df_with_predictions = df

        if 'df_with_predictions' in st.session_state and st.session_state.df_with_predictions is not None:
            df_with_predictions = st.session_state.df_with_predictions
            st.subheader("Data with Predictions")
            st.dataframe(df_with_predictions)   
        
    # Upload new data and predict
    st.sidebar.subheader("Upload New Data for Recommendations")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the uploaded data
        new_data = pd.read_csv(uploaded_file)

        # Check if the necessary features are present
        missing_features = [feat for feat in selected_features if feat not in new_data.columns]
        if missing_features:
            st.error(f"Missing features in uploaded data: {missing_features}")
        else:
            st.success("All necessary features are present.")

            # Load the saved model
            if "model" not in st.session_state or st.session_state.model is None:
                st.warning("Please train and evaluate a model first.")
            else:
                loaded_model = st.session_state.model
                st.success(f"{model_option} loaded successfully!")

                try:
                    new_data_scaled = scaler.transform(new_data[selected_features])

                    # Make predictions
                    new_predictions = loaded_model.predict(new_data_scaled)

                    # Add 'Predictions' column to new_data
                    new_data['Predictions'] = new_predictions

                    st.subheader("Predictions for New Data")
                    st.dataframe(new_data)

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
       
            try: 
                # Ensure patient index is within the valid range
                max_index = len(new_data) - 1
                patient_index = st.number_input("Enter Patient Index:", min_value=0, max_value=max_index, step=1)

                if 0 <= patient_index <= max_index:
                    # Extract prediction for the selected patient index
                    pred = new_data.loc[patient_index, 'Predictions']
                    
                    # Generate recommendation based on the prediction
                    if pred == 1:
                        recommendation = "Patient is at high risk of death. Immediate intervention is advised."
                    else:
                        recommendation = "Patient is at low risk of death. Regular monitoring is recommended."

                    # Display Patient Information
                    st.markdown(f"""
                    <div style="
                        background-color: #ffffff;
                        border-radius: 10px;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                        padding: 20px;
                        margin-bottom: 20px;
                        color: #333;
                        border: 1px solid #ddd;
                    ">   
                        <div style="
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                            margin-bottom: 20px;
                        ">
                            <h2 style="
                                margin: 0;
                                font-size: 26px;
                                color: #2c3e50;
                                border-bottom: 2px solid #3498db;
                                padding-bottom: 10px;
                                flex: 1;
                            ">
                                Patient Index: {patient_index}
                            </h2>
                            <h2 style="
                                margin: 0;
                                font-size: 26px;
                                color: #2c3e50;
                                border-bottom: 2px solid #3498db;
                                padding-bottom: 10px;
                                flex: 1;
                                text-align: right;
                            ">
                                Model: {model_option}
                            </h2>
                        </div>
                        <p style="
                            font-size: 20px;
                            margin: 10px 0;
                            font-weight: bold;
                        ">
                            Prediction: 
                            <span style="
                                font-weight: bold;
                                color: {'#e74c3c' if pred == 1 else '#27ae60'};
                            ">
                                {'High Risk' if pred == 1 else 'Low Risk'}
                            </span>
                        </p>
                        <p style="
                            font-size: 20px;
                            margin: 10px 0;
                            font-weight: bold;
                        ">
                            Recommendation: 
                            <span style="
                                color: #2980b9;
                                background-color: #ecf0f1;
                                border-radius: 5px;
                                padding: 5px 10px;
                                display: inline-block;
                            ">
                                {recommendation}
                            </span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Create an Excel file with patient data and recommendations
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        new_data.to_excel(writer, sheet_name='Patient Data', index=False)
                        # Write recommendations in a separate sheet
                        recommendations_df = pd.DataFrame({
                            'Patient Index': [patient_index],
                            'Prediction': [pred],
                            'Recommendation': [recommendation]
                        })
                        recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
                    st.download_button(label="Download Patient Data and Recommendations as Excel",
                                    data=output.getvalue(),
                                    file_name=f"{model_option}_patient_data_and_recommendations.xlsx",
                                    mime="application/vnd.ms-excel")
            except KeyError:
                pred = "Need to train and evaluate model again."
            
if __name__ == "__main__":
    main()