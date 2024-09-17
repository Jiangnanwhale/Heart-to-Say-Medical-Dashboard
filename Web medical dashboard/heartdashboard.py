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
    st.title("The Global Pioneer in Cardiac Language Translation")
    show_login_image()  
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
    # EDA Navigation
    eda_option = st.selectbox("Select EDA Option:", ["Select Features for Analysis", "Correlation Coefficient"])
    
    if eda_option == "Select Features for Analysis":
        select_features_for_analysis(df)
    elif eda_option == "Correlation Coefficient":
        show_correlation(df)
        
def select_features_for_analysis(df):
    # Sidebar for selecting features and chart type
    st.sidebar.header('Select Features for Analysis')
    
    # Select the type of chart to display
    chart_type = st.sidebar.radio("Select the type of chart:", ["Box Plot", "Line Plot"])
    
    if chart_type == "Box Plot":
        # Single select for x-axis (independent variables)
        x_axis = st.sidebar.selectbox('Select a feature for X-axis', df.columns.tolist())
        # Choose y-axis, setting death event as the default y-axis
        y_axis = st.sidebar.selectbox('Select feature for Y-axis', ['DEATH_EVENT'])
        # Choose a categorical variable for color grouping (e.g., sex)
        color_feature = st.sidebar.selectbox('Select feature for color grouping', df.columns.tolist())

        if x_axis and y_axis and color_feature:
            st.write(f"You have selected: {x_axis}, {color_feature}, and {y_axis}")
            stats_dict = {}
            stats_dict[f"{x_axis}"] = df[x_axis].describe()
            stats_dict[f"{y_axis}"] = df[y_axis].describe()
            stats_dict[f"{color_feature}"] = df[color_feature].describe()
            stats_df = pd.concat(stats_dict, axis=1).T
            st.dataframe(stats_df)

            st.subheader(f"Box Plot of {x_axis} and {y_axis}, grouped by {color_feature.capitalize()}")
            # Box plot for the selected columns
            fig = px.box(df, x=x_axis, y=y_axis, color=color_feature,
                        labels={x_axis: x_axis, y_axis: 'Death Event'},
                        title=f'Box Plot of {x_axis} and {y_axis}, grouped by {color_feature.capitalize()}')
            st.plotly_chart(fig)
        else:
            st.warning("Please select all required features for Box Plot.")

    elif chart_type == "Line Plot":
        # Single select for x-axis (independent variables)
        x_axis = st.sidebar.selectbox('Select a feature for X-axis', df.columns.tolist())
        # Choose y-axis, setting death event as the default y-axis
        y_axis = st.sidebar.selectbox('Select feature for Y-axis', ['DEATH_EVENT'])
        
        if x_axis and y_axis:
            st.write(f"You have selected: {x_axis} and {y_axis}")
            stats_dict = {}
            stats_dict[f"{x_axis}"] = df[x_axis].describe()
            stats_dict[f"{y_axis}"] = df[y_axis].describe()
            stats_df = pd.concat(stats_dict, axis=1).T
            st.dataframe(stats_df)

            st.subheader(f"Line Plot of {x_axis} vs {y_axis}")
            # Line plot for the selected columns
            df_grouped = df.groupby(x_axis)[y_axis].mean().reset_index()
            
            fig = go.Figure()
            # Add line plot trace
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
                width=800,
                height=600
            )
            st.plotly_chart(fig)
        else:
            st.warning("Please select both X-axis and Y-axis features for Line Plot.")
            
def show_correlation(df):
    st.subheader("Correlation Matrix")
    # Choose features for correlation
    selected_features = st.multiselect("Select features for correlation", df.columns.tolist(), default=df.columns.tolist())
    if len(selected_features) > 0:
        corr_df = df[selected_features].corr()
        st.write(corr_df)
        fig = px.imshow(corr_df, text_auto=True)
        st.plotly_chart(fig)
    else:
        st.warning("Please select at least one feature for correlation.")

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
