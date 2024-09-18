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

def main():
    # Check which page to show based on session state
    if st.session_state.get("page") == "dashboard":
        show_dashboard()
    else:
        login_page()

if __name__ == "__main__":
    # Initialize session state variables if they don't exist
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'show_dashboard_button' not in st.session_state:
        st.session_state['show_dashboard_button'] = False
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'


# Function to show the dashboard after file upload
def show_dashboard():
    st.title("Medical Data Analysis Dashboard")

    # Sidebar for image and file upload
    with st.sidebar:
        # Load and display the image
        image_path = "heart_to_say.png"
        image = Image.open(image_path)
        resized_image = resize_image(image, width=300)  # Resize image to fit the sidebar
        st.image(resized_image, use_column_width=True)
        
        st.header("Upload Your Dataset")

        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file (or skip to use default dataset)", type="csv")
        if uploaded_file is not None:
            # Save the uploaded file to session state
            st.session_state["uploaded_file"] = uploaded_file
            st.session_state["use_default"] = False
            st.success("File uploaded successfully!")
        else:
            # Option to proceed without uploading a file
            if st.button("Use Default Dataset"):
                st.session_state["use_default"] = True
                st.success("Using the default dataset!")

        # Add a separator
        st.markdown("---")

        # Display basic dataset information if file is uploaded or default is used
        if "uploaded_file" in st.session_state or "use_default" in st.session_state:
            if "uploaded_file" in st.session_state:
                df = pd.read_csv(st.session_state["uploaded_file"])
            else:
                df = pd.read_csv("d:/KI/project management_SU/PROHI-dashboard-class-exercise/heart_failure_clinical_records_dataset.csv")

    # Load data from the uploaded file or use default data
    if "uploaded_file" in st.session_state:
        df = pd.read_csv(st.session_state["uploaded_file"])
    elif "use_default" in st.session_state and st.session_state["use_default"]:
        df = pd.read_csv("d:/KI/project management_SU/PROHI-dashboard-class-exercise/heart_failure_clinical_records_dataset.csv")
    else:
        st.error("No file uploaded or selected for use. Please go back and choose an option.")
        return
    
    st.markdown("---")
   
    with st.sidebar: 
        st.subheader(":guide_dog: Navigation")
        option = st.sidebar.radio("Select an option:", [ "Data Overview", "Exploratory Data Analysis","Data Modeling"])
    
    if option == "Data Overview":
        show_data_overview(df)
    elif option == "Exploratory Data Analysis":
        show_eda(df)
    elif option == "Data Modeling":
        show_modeling(df)


def show_data_overview(df):
    st.subheader("Dataset Overview")
    
    # First section: Show the dataset
    total_records = len(df)
    positive_cases = df['DEATH_EVENT'].value_counts().get(1, 0)
    negative_cases = df['DEATH_EVENT'].value_counts().get(0, 0)

   
    col1, col2, col3 = st.columns(3)

    # Statistic Card
    with col1:
        st.markdown(f"""
            <div style="background-color: #F8F9FA; padding: 10px; border-radius: 10px; text-align: center;">
                <h4 style="color: #2C3E50;">Total Records</h4>
                <p style="font-size: 24px; color: #16A085;">{total_records}</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background-color: #F8F9FA; padding: 10px; border-radius: 10px; text-align: center;">
                <h4 style="color: #2C3E50;">Positive Cases</h4>
                <p style="font-size: 24px; color: #E74C3C;">{positive_cases}</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div style="background-color: #F8F9FA; padding: 10px; border-radius: 10px; text-align: center;">
                <h4 style="color: #2C3E50;">Negative Cases</h4>
                <p style="font-size: 24px; color: #3498DB;">{negative_cases}</p>
            </div>
        """, unsafe_allow_html=True)
    st.write("")
    st.write(df.head(15))


def show_eda(df):
    st.subheader("Exploratory Data Analysis ")
    st.markdown("---")
    
    # EDA 
    eda_option = st.selectbox("Choose analysis option:", ["Select Features for Analysis", "Correlation Coefficient"])
    
    if eda_option == "Select Features for Analysis":
        select_features_for_analysis(df)
    elif eda_option == "Correlation Coefficient":
        show_correlation(df)

def select_features_for_analysis(df):
    
    col1, col2 = st.columns([4, 1])

    with col1:
        
        chart_type = st.radio("Select Chart Type:", ["Line Plot", "Box Plot"])

        if chart_type == "Line Plot":
            
            x_axis = st.selectbox('Select X-axis feature:', df.columns.tolist())
            y_axis = st.selectbox('Select Y-axis feature:', ['DEATH_EVENT'])

            if x_axis and y_axis:
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
                    yaxis_title='Death Event',
                    width=1000,
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=50) 
                )
                st.plotly_chart(fig)

        elif chart_type == "Box Plot":
            
            x_axis = st.selectbox('Select X-axis feature:', df.columns.tolist())
            y_axis = st.selectbox('Select Y-axis feature:', ['DEATH_EVENT'])
            color_feature = st.selectbox('Select feature for color grouping:', df.columns.tolist())

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
                             labels={x_axis: x_axis, y_axis: "Death Event"},
                             )
            fig.update_layout(
                    width=1000,  # Adjust width to match selectbox
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=50)  # Adjust top margin here
                )
            st.plotly_chart(fig)

    with col2:
        # Data Overview
        st.subheader("Data Overview")
        st.write("**Total Records:** ", len(df))
        st.write("**Features:**", len(df.columns))
        st.write("**Columns:**", ', '.join(df.columns))


def show_correlation(df):
    st.subheader("Correlation Matrix")
    
    selected_features = st.multiselect("Select features for correlation analysis:", df.columns.tolist(), default=df.columns.tolist())

    if len(selected_features) > 0:
        corr_df = df[selected_features].corr()
        fig = px.imshow(corr_df, text_auto=True, title="Correlation Matrix")
    
        # Adjust the area of figture
        fig.update_layout(
            width=1000,  
            height=1000,  
            margin=dict(l=0, r=0, t=0, b=500) 
        )
        
        st.plotly_chart(fig)
    else:
        st.warning("Please select at least one feature for correlation analysis.")

import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import svm 
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
from keras import callbacks
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

def show_modeling(df):
    X = df.drop(["DEATH_EVENT"], axis=1)
    y = df["DEATH_EVENT"]

    # Setting up a standard scaler for the features and analyzing it thereafter
    st.subheader("Scale for the features")
    col_names = list(X.columns)
    s_scaler = preprocessing.StandardScaler()
    X_scaled = s_scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=col_names)
    st.write(X_scaled.describe().T)

   # Plotting the scaled features using box plots with Plotly
    st.subheader("Plot the scaled features using box plots")
    fig = px.box(X_scaled, title="Box Plot of Scaled Features")
    fig.update_layout(xaxis_title='Feature', yaxis_title='Scaled Value')
    st.plotly_chart(fig, use_container_width=True)

    def train_and_evaluate_model():
        # Create and train the SVM model
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=25)
        model1 = svm.SVC()
        model1.fit(X_train, y_train)
        # Predicting the test variables
        y_pred = model1.predict(X_test)
        # Getting the classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        return report, y_test, y_pred
    
    def show_model_results():
        st.title("SVM Model Evaluation")
        # Get the classification report
        report, y_test, y_pred = train_and_evaluate_model()
        # Display the classification report as a data frame
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
    # Call the function to show model results
    show_model_results()

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
    main()
