import streamlit as st
import io
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


# Streamlit page configuration
st.set_page_config(page_title="Heart to Say", 
                   page_icon=":heartbeat:",
                   layout="wide")

# Custom CSS to adjust the layout and add scrollbars
st.markdown("""
    <style>
        /* Main content styling */
        .main {
            max-width: 1400px;  /* Set maximum width for the main content area */
            margin: auto;  /* Center the content */
            overflow-y: auto;  /* Enable vertical scrolling */
            height: 120vh;  /* Set height to allow scrolling */
            padding-right: 10px;  /* Add padding to prevent content from sticking to the scrollbar */
        }

        /* Increase font size for all text */
        body {
            font-size: 16px;  /* Adjust the font size as needed */
        }

        /* Specific styling for headers */
        h1, h2, h3{
            font-size: 2em;  /* Make headers larger */
        }
        
        /* Adjust paragraph and link font size */
        p, a {
            font-size: 1.2em;  /* Make paragraph and link text larger */
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        /* Centering the form */
        .login-form {
            max-width: 400px;  /* Set the maximum width of the form */
            margin: 0 auto;  /* Center the form horizontally */
            padding: 2rem;  /* Add some padding around the form */
            background-color: #f9f9f9;  /* Add a background color */
            border-radius: 10px;  /* Rounded corners */
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);  /* Add a subtle shadow */
        }

        /* Adjust the font sizes for the login elements */
        .login-form h2 {
            font-size: 1.5em;
            text-align: center;  /* Center the title text */
            margin-bottom: 1.5rem;  /* Add space below the title */
        }

        .login-form label {
            font-size: 1.1em;
        }

        .login-form input {
            font-size: 1.1em;
        }

        .login-form button {
            width: 100%;  /* Make the button full width */
            padding: 0.75rem;  /* Add padding inside the button */
            font-size: 1.1em;  /* Adjust the button font size */
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }

        .login-form button:hover {
            background-color: #0056b3;  /* Darken the button on hover */
        }
    </style>
""", unsafe_allow_html=True)

def show_login_image():
    image_path_2 = r"assets/heart_to_say.png"
    image_2 = Image.open(image_path_2)  # Load image
    resized_image_2 = resize_image(image_2, width=300)  # Resize image
    st.image(resized_image_2, caption="", use_column_width=False)
    print(image_path_2)

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

        st.markdown("<br>"*6, unsafe_allow_html=True)

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

# Function to load and resize image
def resize_image(image, width):
    # Calculate new height to maintain aspect ratio
    aspect_ratio = image.height / image.width
    new_height = int(width * aspect_ratio)
    return image.resize((width, new_height))

def show_contact_us():
    st.title("📞 Contact Us")
    st.markdown("""
    For any inquiries or support, please reach out to us at:
    
    **Email**: [heart_to_say_team@dsv.su.se](mailto:heart_to_say_team@dsv.su.se)  
                
    **Phone**: +46 123456789
                
    **Group Members**:  
    - Ifani Pinto Nada  
    - Mahmoud Elachi  
    - Nan Jiang  
    - Sahid Hasan Rahimm  
    - Zhao Chen  
    
      
    **Data Resource**:  
    [Heart Failure Clinical Data](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)
                  
    **Project Information**:  
    [Project GitHub](https://github.com/Jiangnanwhale/Heart-Health-Caring-Team)
                
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

    st.markdown("---") 

def show_dashboard():
    if st.sidebar.button("Logout"):
        logout()
    
    with st.sidebar:
        image_path = "assets/heart_to_say.png"
        image = Image.open(image_path)
        resized_image = resize_image(image, width=300)
        st.image(resized_image, use_column_width=True)

        st.subheader(" Home ")
    show_input_data()
    
def show_input_data():

    with st.sidebar:
        st.subheader(":guide_dog: Navigation")
        option = st.radio("Select an option:", ["Home","Descriptive analytics", "Diagnostic analytics","Predictive analytics", "Contact Us"])
    
    df = pd.read_csv("d:/KI/project management_SU/PROHI-dashboard-class-exercise/heart_failure_clinical_records_dataset.csv")
    if option == "Descriptive analytics":
        show_data_overview(df)
    elif option == "Diagnostic analytics":
        show_eda(df)
    elif option == "Contact Us":
        show_contact_us()
    elif option == "Predictive analytics":
        with st.sidebar:
            sub_option = st.radio("Choose an action:", ["Input your data", "Show model performance"])
        if sub_option == "Input your data":
            upload_pre_model()
        elif sub_option == "Show model performance":
            show_model_performance(df)
    elif option == "Home":
        show_home()

def show_home():
    st.title("💖 Welcome to Heart to Say Dashboard")
    st.markdown("---")
    
    st.markdown(
        """
        **This dashboard supports physicians in predicting the risk of mortality due to heart failure.** 
        By utilizing patient data and advanced analytics, we aim to provide insights for better clinical decision-making.
        """
    )
    
    st.markdown("### Dashboard Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            - **🏠 Home**: 
              Provides an overview of the dashboard's functionality.
            - **📊 Descriptive Analytics**: 
              Explore heart failure patient data, enabling healthcare professionals to view trends and prevalence based on:
              - Age and gender
              - Smoking status
              - Comorbidities
              - Laboratory test results
            """
        )
    
    with col2:
        st.markdown(
            """
            - **🔍 Diagnostic Analytics**: 
              Analyze correlations and patterns between heart failure risk factors and mortality to provide a comprehensive overview.
            - **🤖 Predictive Analytics**: 
              Input patient data on heart failure risk factors to predict the risk level of mortality.
            - **📞 Contact Us**: 
              Get in touch for more information about the project, our team, and how to reach us.
            """
        )
    
    st.sidebar.info("Navigate through the tabs to explore different analytics and features.")


def upload_pre_model():
    st.title("Input Your Medical Data")
    st.markdown("")
    
    # Initialize the session state key if it doesn't exist
    if "input_history" not in st.session_state:
        st.session_state["input_history"] = []

    
    # Create a container for the input fields to improve layout
    with st.form(key='input_form'):
        col1, col2, col3 = st.columns(3)  # Create two columns for better organization

        with col1:
            age = st.number_input("**Age (years)**", min_value=0, step=1, value=st.session_state.get("age", 0))
            creatinine_phosphokinase = st.number_input("**Creatinine Phosphokinase (mcg/L)**", min_value=0.0, format="%.2f", value=st.session_state.get("creatinine_phosphokinase", 0.0))
            ejection_fraction = st.number_input("**Ejection Fraction (%)**", min_value=0.0, max_value=100.0, format="%.2f", value=st.session_state.get("ejection_fraction", 0.0))
            platelets = st.number_input("**Platelets (kiloplatelets/mL)**", min_value=0, value=st.session_state.get("platelets", 0))
        with col2:
            serum_creatinine = st.number_input("**Serum Creatinine (mg/dL)**", min_value=0.0, format="%.2f", value=st.session_state.get("serum_creatinine", 0.0))
            serum_sodium = st.number_input("**Serum Sodium (mEq/L)**", min_value=0.0, format="%.2f", value=st.session_state.get("serum_sodium", 0.0))
            anaemia = st.selectbox("**Anaemia (Yes/No)**", options=["Yes", "No"], index=0 if st.session_state.get("anaemia") != "Yes" else 1)
            diabetes = st.selectbox("**Diabetes (Yes/No)**", options=["Yes", "No"], index=0 if st.session_state.get("diabetes") != "Yes" else 1)
        with col3:
            high_blood_pressure = st.selectbox("**High Blood Pressure (Yes/No)**", options=["Yes", "No"], index=0 if st.session_state.get("high_blood_pressure") != "Yes" else 1)
            sex = st.selectbox("**Sex**", options=["Male", "Female"], index=0 if st.session_state.get("sex") != "Female" else 1)
            smoking = st.selectbox("**Smoking (Yes/No)**", options=["Yes", "No"], index=0 if st.session_state.get("smoking") != "Yes" else 1)
            time = st.number_input("**Follow-up Period (days)**", min_value=0, value=st.session_state.get("time", 0))
           

        # Submit button for the form
        submit_button = st.form_submit_button("Calculate Prediction")
    
    scaler = StandardScaler()

    if submit_button:
         # Store input values in session state for reset functionality
        st.session_state["age"] = age
        st.session_state["creatinine_phosphokinase"] = creatinine_phosphokinase
        st.session_state["ejection_fraction"] = ejection_fraction
        st.session_state["platelets"] = platelets
        st.session_state["serum_creatinine"] = serum_creatinine
        st.session_state["serum_sodium"] = serum_sodium
        st.session_state["anaemia"] = anaemia
        st.session_state["diabetes"] = diabetes
        st.session_state["high_blood_pressure"] = high_blood_pressure
        st.session_state["sex"] = sex
        st.session_state["smoking"] = smoking
        st.session_state["time"] = time
        
        prediction = None
        model = joblib.load('D:\KI\project management_SU\PROHI-dashboard-class-exercise\logistic_regression_model.pkl')
        # Check if the loaded model is indeed a valid model
        if hasattr(model, 'predict'):
            # Prepare input data for the model
            input_data = pd.DataFrame({
                'age': [age],
                'anaemia': [1 if anaemia == "Yes" else 0],
                'creatinine_phosphokinase': [creatinine_phosphokinase],
                'diabetes': [1 if diabetes == "Yes" else 0],
                'ejection_fraction': [ejection_fraction],
                'high_blood_pressure': [1 if high_blood_pressure == "Yes" else 0],
                'platelets': [platelets],
                'serum_creatinine': [serum_creatinine],
                'serum_sodium': [serum_sodium],
                'sex': [1 if sex == "Male" else 0],
                'smoking': [1 if smoking == "Yes" else 0],
                'time': [time]
            })
            # Fit the scaler if it's not fitted
            if 'scaler' not in st.session_state:
                scaler.fit(input_data)  
                st.session_state['scaler'] = scaler 
            else:
                scaler = st.session_state['scaler']  

            input_data_scaled = scaler.transform(input_data)
            
            prediction = model.predict(input_data_scaled)
    
            if prediction is not None and len(prediction) > 0:
                recommendation = (
                    "Patient is at high risk of death. Immediate intervention is advised."
                    if prediction[0] == 1
                    else "Patient is at low risk of death. Regular monitoring is recommended."
                )
            
            st.subheader("Recommendation Card")
            # Create a custom styled recommendation box
            st.markdown(f"""
            <div style="
                background-color: #ffffff;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                padding: 20px;
                color: #333;
                border: 1px solid #ddd;
                max-width: 800px;
            ">   
                <h2 style="
                    margin: 0;
                    font-size: 26px;
                    color: #2c3e50;
                    padding-bottom: 10px;
                ">
                    Model: {model}
                </h2>
                <hr style="
                    border: 0;
                    height: 2px;
                    background: #3498db;
                    margin: 20px 0;
                ">
                <p style="
                    font-size: 20px;
                    margin: 10px 0;
                    font-weight: bold;
                ">
                    Prediction: 
                    <span style="
                        font-weight: bold;
                        color: {'#e74c3c' if prediction[0] == 1 else '#27ae60'};
                    ">
                        {'High Risk' if prediction[0] == 1 else 'Low Risk'}
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
            """, unsafe_allow_html=True)  # Display the recommendation with custom styling


                # Append the current inputs to the input history, including the prediction if available    
        st.session_state["input_history"].append({
            "age": age,
            "creatinine_phosphokinase": creatinine_phosphokinase,
            "ejection_fraction": ejection_fraction,
            "platelets": platelets,
            "serum_creatinine": serum_creatinine,
            "serum_sodium": serum_sodium,
            "anaemia": anaemia,
            "diabetes": diabetes,
            "high_blood_pressure": high_blood_pressure,
            "sex": sex,
            "smoking": smoking,
            "time": time,
            "Prediction": "High Risk" if prediction is not None and prediction[0] == 1 else "Low Risk"
        })

    def format_record(record, idx):
        prediction = record.get("Prediction", "N/A")
        return f"""
        <div style='
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        font-size: 14px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        '>
        <h4 style='color: #2c3e50;'>Record {idx+1}</h4>
        <table style='width: 100%; border-collapse: collapse;'>
            <tr>
                <td><strong>Age:</strong> {record["age"]} years</td>
                <td><strong>Creatinine Phosphokinase:</strong> {record["creatinine_phosphokinase"]} mcg/L</td>
                <td><strong>Ejection Fraction:</strong> {record["ejection_fraction"]}%</td>
            </tr>
            <tr>
                <td><strong>Platelets:</strong> {record["platelets"]} kiloplatelets/mL</td>
                <td><strong>Serum Creatinine:</strong> {record["serum_creatinine"]} mg/dL</td>
                <td><strong>Serum Sodium:</strong> {record["serum_sodium"]} mEq/L</td>
            </tr>
            <tr>
                <td><strong>Anaemia:</strong> {record["anaemia"]}</td>
                <td><strong>Diabetes:</strong> {record["diabetes"]}</td>
                <td><strong>High Blood Pressure:</strong> {record["high_blood_pressure"]}</td>
            </tr>
            <tr>  
                <td><strong>Sex:</strong> {record["sex"]}</td>
                <td><strong>Smoking:</strong> {record["smoking"]}</td>
                <td colspan='3'><strong>Follow-up Period:</strong> {record["time"]} days</td>
            </tr>
            <tr>
                <td><strong>Prediction:</strong> {prediction}</td>
            </tr>
        </table>
        </div>
        """

    st.markdown("")
    # Reset option to clear inputs only (keeping the model)
    if st.button("Reset"):
        for key in ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", 
                    "serum_creatinine", "serum_sodium", "anaemia", "diabetes", 
                    "high_blood_pressure", "sex", "smoking", "time"]:
            if key in st.session_state:
                del st.session_state[key]

        st.success("All inputs have been reset. You can continue entering data.")
    
    st.markdown("---") 
      # Display input history in styled format
    st.subheader("Input History")
    if "input_history" in st.session_state:
        for idx, record in enumerate(st.session_state["input_history"]):
            st.markdown(format_record(record, idx), unsafe_allow_html=True)
    else:
        st.info("No input history available.")
    st.markdown("")
    st.markdown("")

import streamlit as st
import pandas as pd

def show_data_overview(df):
    st.title("Descriptive analytics")
    
    # Dataset basic info
    total_records = len(df)
    positive_cases = df['DEATH_EVENT'].value_counts().get(1, 0)
    negative_cases = df['DEATH_EVENT'].value_counts().get(0, 0)
    missing_values = df.isnull().sum().sum()

    col1, col2, col3, col4 = st.columns(4)

    card_style = """
        <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center; color: white; height: 300px; margin: 10px; display: flex; flex-direction: column; justify-content: center;">
            <h3>{title}</h3>
            <p style="font-size: 24px; font-weight: bold;">{value}</p>
            <p>{description}</p>
        </div>
    """

    with col1:
        st.markdown(card_style.format(bg_color="#1abc9c", title="Total Records", value=total_records,
                                    description="This represents the total number of patient records in the dataset."), unsafe_allow_html=True)

    with col2:
        st.markdown(card_style.format(bg_color="#e74c3c", title="Positive Cases", value=positive_cases,
                                    description="Number of patients who experienced a death event (1)."), unsafe_allow_html=True)

    with col3:
        st.markdown(card_style.format(bg_color="#3498db", title="Negative Cases", value=negative_cases,
                                    description="Number of patients who did not experience a death event (0)."), unsafe_allow_html=True)

    with col4:
        st.markdown(card_style.format(bg_color="#9b59b6", title="Missing Values", value=missing_values,
                                    description="This indicates how many data entries are missing across all features."), unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("")
    st.write("Here, numerical features are defined if the the attribute has more than 5 unique elements else it is a categorical feature. By the way, all categorical features here are boolean features.")
    left_column, right_column = st.columns(2)
    
    col = list(df)
    categorical_features = []
    numerical_features = []
    for i in col:
        if len(df[i].unique()) > 5:
            numerical_features.append(i)
        else:
            categorical_features.append(i)
        
    with left_column:

        st.write("### Distribution of Categorical Features")
    
        st.write("Select a Categorical feature from the dataset to visualize its distribution. The histogram will color-code the data based on the DEATH_EVENT status.")
        selected_column = st.selectbox("Select a feature to visualize", categorical_features)
        if 'DEATH_EVENT' in categorical_features:
            categorical_features.remove('DEATH_EVENT')
    
        count_data = df[selected_column].value_counts().reset_index()
        count_data.columns = [selected_column, 'count']

        fig_feature_1 = px.pie(
            count_data, 
            names=selected_column, 
            values='count', 
            color_discrete_sequence=['#2ca02c', '#ff7f0e'], 
            title=f'Distribution of {selected_column}'
        )
        fig_feature_1.update_traces(textinfo='percent+label',
                                    marker=dict(line=dict(color='white', width=4))
        )           
        st.plotly_chart(fig_feature_1, use_container_width=True)

        fig_feature_2 = px.histogram(df, x=selected_column, color='DEATH_EVENT', barmode='group',
                    color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                       title=f'Distribution of {selected_column} vs Death Event' )

        st.plotly_chart(fig_feature_2, use_container_width=True)
        st.write('''all the graphs near about share the same pattern. Nothing special here.''')
   
    with right_column:

        st.write("### Distribution of Numerical Features")

        st.write("Select a numerical feature from the dataset to visualize its distribution. The histogram will color-code the data based on the DEATH_EVENT status.")
        selected_column = st.selectbox("Select a feature to visualize", numerical_features)
        if 'DEATH_EVENT' in numerical_features:
            numerical_features.remove('DEATH_EVENT')

        fig_feature_1 = px.histogram(df, x=selected_column, barmode='group',
                                   color_discrete_sequence=['#FF69B4'],
                                    title=f'Distribution of {selected_column}')
        fig_feature_1.update_layout(bargap=0.2)
        
        fig_feature_2 = px.histogram(df, x=selected_column, color='DEATH_EVENT', barmode='group',
                                    color_discrete_sequence=['#8A2BE2', '#FFD700'],
                                    title=f'Distribution of {selected_column} vs Death Event')
        
        st.plotly_chart(fig_feature_1, use_container_width=True)
        st.plotly_chart(fig_feature_2, use_container_width=True)
        st.write('''Cases of "DEATH_EVENT" initiate from the "age" of 42. Some specific peaks of high cases of "DEATH_EVENT" can be observed at 45, 50, 60, 65, 70, 75 and 80.
                   High cases of "DEATH_EVENT" can be observed for "ejaction_fraction" values from 20 - 60.
                       "serum_creatinine" values from 0.6 - 3.0 have higher probability to lead to DEATH_EVENT.
                        "serum_sodium" values 127 - 145 indicate towards a "DEATH_EVENT" due to "heart failure".''')
    
    st.markdown("<br>"*3, unsafe_allow_html=True)

def show_eda(df):
    st.title("Diagnostic Analysis")
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

            left_column, right_column =st.columns(2)
            with left_column:
                x_axis = st.selectbox('Select X-axis feature:', df.columns.tolist())
            with right_column:
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
            left_column, middle_column, right_column =st.columns(3)
            with left_column:
                x_axis = st.selectbox('Select X-axis feature:', df.columns.tolist())
            with middle_column:
                y_axis = st.selectbox('Select Y-axis feature:', df.columns.tolist())
            with right_column:
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


                st.markdown("<br>"*1, unsafe_allow_html=True)

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
            left_column, middle_column, right_column =st.columns(3)
            with left_column:
                x_axis = st.selectbox('Select X-axis feature:', df.columns.tolist())
            with middle_column:
                y_axis = st.selectbox('Select Y-axis feature:', df.columns.tolist())
            with right_column:
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


                st.markdown("<br>"*1, unsafe_allow_html=True)

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

    st.markdown("<br>"*3, unsafe_allow_html=True)

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
    st.title("Clustering Analysis")
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

    left_column, right_column =st.columns(2)
    with left_column:
        feature1 = st.selectbox('Select First Feature for Clustering:', df.columns.tolist())
    with right_column:
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

        st.markdown("<br>"*3, unsafe_allow_html=True)


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

def show_correlation(df):
    st.subheader("Correlation Matrix")

    selected_features = st.multiselect("Select features for correlation analysis:", df.columns.tolist(), default=df.columns.tolist())
    st.markdown("")

    if len(selected_features) > 0:

        corr_df = df[selected_features].corr()


        fig = ff.create_annotated_heatmap(
            z=corr_df.values,
            x=list(corr_df.columns),
            y=list(corr_df.index),
            annotation_text=corr_df.round(2).values,
            colorscale='Viridis', 
            showscale=True,
            colorbar=dict(title="Correlation Coefficient")
        )

        fig.update_layout(
            title='Correlation Matrix',
            xaxis_title='Features',
            yaxis_title='Features',
            width=800,
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

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

    st.markdown("<br>"*3, unsafe_allow_html=True)

import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import plotly.express as px
import io
import pickle
import joblib
import shap
import matplotlib.pyplot as plt

def show_model_performance(df):
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

    # Load your pre-trained model from a specified path
    model_path = 'assets/logistic_regression_model.pkl'  # Update the path as needed
    model = joblib.load(model_path)  # Load the model

    # Assuming X_scaled is already defined elsewhere in your code
    # Make predictions
    predictions = model.predict(X)
    predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
        
    df = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)

    # User selection for evaluation or SHAP
    analysis_option = st.sidebar.selectbox(
        "Select Analysis Type:",
        ("View Predictions", "Model Performance", "SHAP Analysis")
    )

    if analysis_option == "Model Performance":
        st.subheader("Model Performance")
        
        # Calculate accuracy
        accuracy = accuracy_score(y, predictions)
        st.write(f"**Accuracy:** {accuracy:.4f}")
    
        left_column,right_column = st.columns(2)
        with right_column:
            # Classification Report
            
            report = classification_report(y, predictions, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.write("### Classification Report")
            st.markdown("")
            st.markdown("")
            st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
            st.write("""
            The classification report provides a comprehensive overview of the model's performance across different classes. Here are the key metrics included:

            - **Precision**: This metric indicates the accuracy of positive predictions. It is calculated as the ratio of true positive predictions to the total predicted positives. High precision means that most positive predictions are correct.

            - **Recall**: Also known as sensitivity or true positive rate, recall measures the ability of the model to identify all relevant instances. It is calculated as the ratio of true positive predictions to the total actual positives. High recall indicates that the model is good at capturing positive cases.

            - **F1-Score**: This is the harmonic mean of precision and recall, providing a single score that balances both metrics. It is particularly useful when you want to find an optimal balance between precision and recall. A high F1-score indicates a good balance between precision and recall.

            - **Support**: This indicates the number of actual occurrences of each class in the specified dataset. It helps in understanding the distribution of classes and the relevance of other metrics.

            Overall, the classification report is crucial for evaluating the model's effectiveness, especially in cases where class imbalance exists. By analyzing these metrics, we can gain insights into areas for improvement and make informed decisions about model optimization.
            """)

        with left_column:
            # Confusion Matrix
            
            conf_matrix = confusion_matrix(y, predictions)
            fig = px.imshow(
                conf_matrix, 
                text_auto=True, 
                color_continuous_scale="Blues",
                labels={'color': 'Count'},
                aspect="auto"
            )
            fig.update_layout(
                title = "Confusion Matrix",
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
            - **Accuracy**: \\((TP + TN) / (TP + TN + FP + FN)\\) - Overall correctness of the model.
        - **Precision**: \\(TP / (TP + FP)\\) - How many of the predicted positives are actually positive.
        - **Recall**: \\(TP / (TP + FN)\\) - How many of the actual positives were correctly identified.
        - **F1-Score**: The harmonic mean of precision and recall.

            The confusion matrix is a powerful tool for understanding the strengths and weaknesses of the classification model, helping to identify specific areas where the model may need improvement. By analyzing the counts, we can gain insights into class-specific performance and potential biases in the model.
            """)
            st.markdown("<br>"*3, unsafe_allow_html=True)

    elif analysis_option == "SHAP Analysis":
        st.header("SHAP Analysis")
        
        X_scaled_np = np.array(X)
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
        
        left_column, right_column = st.columns(2)
        with left_column:
            # SHAP Summary Plot
            st.subheader("Summary Plot")
            st.markdown("")
            st.markdown("")
            shap.summary_plot(shap_values, X_scaled_np, feature_names=sorted_feature_names)
            st.pyplot(plt)
        
        with right_column:
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
        st.markdown("<br>"*3, unsafe_allow_html=True)

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
                max-width: 900px;
            ">   
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                ">
                    <h2 style="
                        margin: 0;
                        font-size: 22px;
                        color: #2c3e50;
                        border-bottom: 2px solid #3498db;
                        padding-bottom: 15px;
                        flex: 1;
                    ">
                        Patient Index:{patient_index}
                    </h2>
                    <h2 style="
                        margin: 0;
                        font-size: 22px;
                        color: #2c3e50;
                        border-bottom: 2px solid #3498db;
                        padding-bottom: 15px;
                        flex: 1;
                        text-align: right;
                    ">
                        Model: {model}
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
        st.download_button(label="Download Patient Data and Predictions as Excel",
                            data=output.getvalue(),
                            file_name=f"{model}_patient_data_and_predictions.xlsx",
                            mime="application/vnd.ms-excel")
        st.markdown("<br>"*3, unsafe_allow_html=True)


if __name__ == "__main__":
    show_dashboard()