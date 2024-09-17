import streamlit as st
import io
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image

# Login function
def login(username=None, password=None):
    # Successful login regardless of what is input
    st.session_state["logged_in"] = True

# Login page
def login_page():
    st.title("The Global Pioneer in Cardiac Language Translation")
    # Choose the way to login in 
    st.markdown("---")

    with st.container():
        st.subheader(":key: Login to Your Account")

        # Layout
        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("Login with Username and Password"):
                st.session_state["login_method"] = "username_password"

        with col2:
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
        with st.form(key='login_form'):
            col1, col2 = st.columns([3, 1])

            with col1:
                username = st.text_input("Username", key="username_input")
                password = st.text_input("Password", type="password", key="password_input")
            
            # Place the login button in the blank area on the right
            with col2:
                st.write("")  # Blank area
            
            login_button = st.form_submit_button("Login")

            if login_button:
                login(username, password)
                if st.session_state.get("logged_in", False):
                    st.success("Login successfully")
                    st.session_state["show_dashboard_button"] = True

        if st.session_state.get("show_dashboard_button", False):
            if st.button("Proceed to Dashboard"):
                st.session_state["show_dashboard_button"] = False
                st.session_state["logged_in"] = True  # Ensure that it is marked as logged in

# Function to load and resize image
def resize_image(image, width):
    # Calculate new height to maintain aspect ratio
    aspect_ratio = image.height / image.width
    new_height = int(width * aspect_ratio)
    return image.resize((width, new_height))

def show_qr_code_login():
    with st.container():
        # Display the QR code on the right for logging in
        image_path = "qrcode_heat_to_say.png"
        image = Image.open(image_path)# Load image
        # Resize image to a specific width
        resized_image = resize_image(image, width=300)  # Set the desired width
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
    st.title("Dashboard")
    st.write(":coffee: Welcome to the dashboard ")

    # Sidebar for file upload and dataset information
    with st.sidebar:
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
        option = st.sidebar.radio("Select an option:", [ "Data Overview", "EDA","Modeling"])
    
    if option == "Data Overview":
        show_data_overview(df)
    elif option == "EDA":
        show_eda(df)
    elif option == "Modeling":
        show_modeling(df)

def show_data_overview(df):
    st.subheader("Dataset Overview")
    st.write(df.head(15))
    # First section: Show the dataset
    st.write(f"**Total Records:** {len(df)}")
    st.write(f"**Positive Cases:** {df['DEATH_EVENT'].value_counts().get(1, 0)}")
    st.write(f"**Negative Cases:** {df['DEATH_EVENT'].value_counts().get(0, 0)}")

def show_eda(df):
    # Columns layout for multiple data analysis sections
    col1, col2 = st.columns(2)
    # First column: Age Distribution of Death Events
    death = df[df['DEATH_EVENT'] == 1]
    fig_age = px.histogram(death, x='age', color='sex', nbins=10, 
                           labels={'age': 'Age', 'sex': 'Sex'}, )
    col1.subheader("Age Distribution of Death Events by Sex")
    col1.plotly_chart(fig_age, use_container_width=True)

    # Second column: Correlation Matrix
    fig_corr = px.imshow(df.corr())
    col2.subheader("Correlation Matrix")
    col2.plotly_chart(fig_corr, use_container_width=True)

    # Additional section for more analysis
    st.subheader("Additional Analysis")
    fig_bp = px.box(df, x='DEATH_EVENT', y='serum_creatinine', color='sex',
                    labels={'serum_creatinine': 'Serum Creatinine', 'DEATH_EVENT': 'Death Event'},
                    )
    st.plotly_chart(fig_bp, use_container_width=True)

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
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        login_page()  # Show login page
    else:
        show_dashboard()  # Show dashboard after login

# Streamlit page configuration
st.set_page_config(page_title="Heart to Say", page_icon=":heartbeat:")

image_path_2 = "heart_to_say.png"
image_2 = Image.open(image_path_2)# Load image
# Resize image to a specific width
resized_image_2 = resize_image(image_2, width=400)  # Set the desired width
# Display the resized image
st.image(resized_image_2, caption="",use_column_width=False)

if __name__ == "__main__":
    main()
