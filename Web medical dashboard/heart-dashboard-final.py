## "pip install -r requirements.txt" in terminal first##

import streamlit as st
import io
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go
import warnings
from sklearn.cluster import KMeans
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
    
    **Email**: heart_to_say_team@dsv.su.se
                
    **Phone**: +46 123456789
                
    **Group Members**:  
    - Ifani Pinto Nada  
    - Mahmoud Elachi  
    - Nan Jiang  
    - Sahid Hasan Rahim  
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
    3. Joseph P, Roy A, Lonn E, Störk S, Floras J, Mielniczuk L, et al. Global Variations in Heart Failure Etiology, Management, and Outcomes. JAMA. 2023 May 16;329(19):1650-1661.
    4. Regitz-Zagrosek V. Sex and Gender Differences in Heart Failure. Int J Heart Fail. 2020 Apr 13;2(3):157-81.
    5. Donzé JD, Beeler PE, Bates DW. Impact of Hyponatremia Correction on the Risk for 30-Day Readmission and Death in Patients with Congestive Heart Failure. Am J Med. 2016 Aug;129(8):836-42.
    6. Stewart S, Playford D, Scalia GM, Currie P, Celermajer DS, Prior D, Codde J, Strange G; NEDA Investigators. Ejection fraction and mortality: a nationwide register-based cohort study of 499 153 women and men. Eur J Heart Fail. 2021 Mar;23(3):406-416.
    7. Zhong J, Gao J, Luo JC, Zheng JL, Tu GW, Xue Y. Serum creatinine as a predictor of mortality in patients readmitted to the intensive care unit after cardiac surgery: a retrospective cohort study in China. J Thorac Dis. 2021 Mar;13(3):1728-1736.
    8. Metra M, Cotter G, Gheorghiade M, Dei Cas L, Voors AA. The role of the kidney in heart failure. Eur Heart J. 2012 Sep;33(17):2135-42.
    9. Mayo Clinic. Creatinine test [Internet]. Rochester, MN: Mayo Foundation for Medical Education and Research; 2022 [cited 2024 Oct 19]. Available from: https://www.mayoclinic.org/tests-procedures/creatinine-test/about/pac-20384646
    10. British Heart Foundation. Heart failure [Internet]. London: British Heart Foundation; 2024 [cited 2024 Oct 19]. Available from: https://www.bhf.org.uk/informationsupport/conditions/heart-failure            

    **Got some thoughts or suggestions? Don't hesitate to reach out to us. We'd love to hear from you!**
    """)

    st.markdown("---") 

def show_dashboard():
    if st.sidebar.button("Logout"):
        logout()
    
    with st.sidebar:
        image_path = r"assets/heart_to_say.png"
        image = Image.open(image_path)
        resized_image = resize_image(image, width=300)
        st.image(resized_image, use_column_width=True)

        st.subheader(" Home ")
    show_input_data()
    
def show_input_data():

    with st.sidebar:
        st.subheader(":guide_dog: Navigation")
        option = st.radio("Select an option:", ["Home","Descriptive analytics", "Diagnostic analytics","Predictive analytics", "Contact Us"])
    
    df = pd.read_csv("Web medical dashboard/heart_failure_clinical_records_dataset.csv")
    df.rename(columns={'time': 'follow-up days'}, inplace=True)

    if option == "Descriptive analytics":
        show_data_overview(df)
    elif option == "Diagnostic analytics":
        show_eda(df)
    elif option == "Contact Us":
        show_contact_us()
    elif option == "Predictive analytics":
        with st.sidebar:
            sub_option = st.radio("Choose an action:", ["Input your data", "Model performance (SHAP)"])
        if sub_option == "Input your data":
            upload_pre_model()
        elif sub_option == "Model performance (SHAP)":
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
        model = joblib.load('Web medical dashboard/xgb3_model.pkl')
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
                    Model: {"XGBClassifier"}
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

    col1, col2, col3 = st.columns(3)

    card_style = """
        <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center; color: white; height: 300px; margin: 10px; display: flex; flex-direction: column; justify-content: center;">
            <h3>{title}</h3>
            <p style="font-size: 24px; font-weight: bold;">{value}</p>
            <p>{description}</p>
        </div>
    """

    with col1:
        st.markdown(card_style.format(bg_color="#2ca02c", title="Total Records", value=total_records,
                                  description="This represents the total number of patient records in the dataset."), unsafe_allow_html=True)

    with col2:
        st.markdown(card_style.format(bg_color="#ff0000", title="Death Cases", value=positive_cases,
                                    description="Number of patients who experienced a death event."), unsafe_allow_html=True)

    with col3:
        st.markdown(card_style.format(bg_color="#808080", title="Survival Cases", value=negative_cases,
                                    description="Number of patients who did not experience a death event."), unsafe_allow_html=True)

    
    st.markdown("")
    st.markdown("")

    st.subheader("Overview of Patient Data")
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
    
        selected_column = st.selectbox("Select a binary feature to visualize (e.g., diabetes, anaemia, etc)", categorical_features,
                                       index=categorical_features.index('diabetes'))
        if 'DEATH_EVENT' in categorical_features:
            categorical_features.remove('DEATH_EVENT')
    
        count_data = df[selected_column].value_counts().reset_index()
        count_data.columns = [selected_column, 'count']
        if selected_column == 'sex':
                label_map = {
                    0: 'Female',
                    1: 'Male'
                }
        else:
            label_map = {
                0: f"No {selected_column}",
                1: f"{selected_column} Occurred"
            }
        count_data[selected_column] = count_data[selected_column].map(label_map)

        fig_feature_1 = px.pie(
            count_data, 
            names=selected_column, 
            values='count', 
            color_discrete_sequence=['#808080', '#ff0000'], 
            title=f'Distribution of {selected_column}'
        )
        fig_feature_1.update_traces(textinfo='percent+label',
                                    textfont=dict(color='white'),
                                    marker=dict(line=dict(color='white', width=4))
        )           
        st.plotly_chart(fig_feature_1, use_container_width=True)

        selected_counts = df[selected_column].value_counts()
        selected_column_percentage = round((selected_counts.get(1, 0) / len(df)) * 100)  # dead
        opposite_percentage = round((selected_counts.get(0, 0) / len(df)) * 100) 

        conclusion_text = f"Based on the dataset, {selected_column_percentage}% are {selected_column}, while {opposite_percentage}% belong to the opposite category. " 
        st.write(conclusion_text)

        fig_feature_2 = px.histogram(df, x=selected_column, color='DEATH_EVENT', barmode='group',
                    color_discrete_map={0: '#808080', 1: '#ff0000'},
                    title=f'Distribution of {selected_column} vs Death Event' )
        fig_feature_2.for_each_trace(lambda t: t.update(name='Survived' if t.name == '0' else 'Death occured'))
        st.plotly_chart(fig_feature_2, use_container_width=True)

        with right_column:

            selected_column = st.selectbox("Select a continual feature to visualize (e.g., age, platelets, etc)", numerical_features)
            if 'DEATH_EVENT' in numerical_features:
                numerical_features.remove('DEATH_EVENT')

            fig_feature_1 = px.histogram(df, x=selected_column, barmode='group',
                                    color_discrete_sequence=['#2ca02c'],
                                        title=f'Distribution of {selected_column}')
            fig_feature_1.update_layout(bargap=0.2)
            
            fig_feature_2 = px.histogram(df, x=selected_column, color='DEATH_EVENT', barmode='group',
                                        color_discrete_sequence=['#ff0000','#808080'],
                                        title=f'Distribution of {selected_column} vs Death Event')
            
            st.plotly_chart(fig_feature_1, use_container_width=True)
            min_value = round(df[selected_column].min())
            max_value = round(df[selected_column].max())
            average_value = round(df[selected_column].mean())

            conclusion_text = (
                f"Based on the results, heart failure patients range from {min_value} to {max_value} for the feature '{selected_column}', "
                f"with an average value of {average_value}."
            )
            st.write(conclusion_text)
            fig_feature_2.for_each_trace(lambda t: t.update(name='Survived' if t.name == '0' else 'Death occured'))
            st.plotly_chart(fig_feature_2, use_container_width=True)

    st.markdown("<br>"*3, unsafe_allow_html=True)

def show_eda(df):
    st.title("Diagnostic Analysis")
    st.write("This section provides a brief overview of the dataset, including the total number of records and the features available.")
    st.markdown("---")

    # EDA 
    eda_option = st.selectbox("Choose analysis option:", [ "Correlation Coefficient","Basic Feature Relationships"])

    if eda_option == "Correlation Coefficient":
        show_correlation(df)
    elif eda_option == "Basic Feature Relationships":
        basic_feature_relationships(df)

def basic_feature_relationships(df):

    col1, col2 = st.columns([4, 1])

    with col1:

        chart_type = st.radio("Select Chart Type:", ["Line Plot", "Box Plot","Scatter Plot"])
        
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
            left_column, right_column =st.columns(2)
            with left_column:
                x_axis = st.selectbox('Select X-axis feature:', df.columns.tolist())
            with right_column:
                y_axis = st.selectbox('Select Y-axis feature:', df.columns.tolist())
            
            st.write("### Box Plot Visualization")
            st.write("A box plot is useful for displaying the distribution of a dataset based on a continuous variable, grouped by another variable.")
            if x_axis and y_axis:
                st.write(f"Selected Features: **X-axis:** {x_axis}, **Y-axis:** {y_axis}")

                # Feature Stats
                st.subheader("Feature Stats")
                stats_dict = {
                    f"{x_axis}": df[x_axis].describe(),
                    f"{y_axis}": df[y_axis].describe(),
                }
                stats_df = pd.concat(stats_dict, axis=1).T
                st.dataframe(stats_df)

                st.markdown("<br>"*1, unsafe_allow_html=True)

            st.subheader(f"Box Plot: {x_axis} vs {y_axis}")
            fig = px.box(df, x=x_axis, y=y_axis,
                             labels={x_axis: x_axis, y_axis: y_axis},
                             )
            fig.update_layout(
                    width=1000,  # Adjust width to match selectbox
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=50)  # Adjust top margin here
                )
            st.plotly_chart(fig)

        elif chart_type == "Scatter Plot":
            left_column, right_column = st.columns(2)
            with left_column:
                x_axis = st.selectbox('Select X-axis feature:', df.columns.tolist())
                
            with right_column:
                y_axis = st.selectbox('Select Y-axis feature:', df.columns.tolist())

            st.write("### Scatter Plot Visualization")
            st.write("Scatter plots are great for visualizing relationships between two continuous variables, with the option to color-code based on a third feature.")

            if x_axis and y_axis:
                st.write(f"Selected Features: **X-axis:** {x_axis}, **Y-axis:** {y_axis}")

                # Feature Stats
                st.subheader("Feature Stats")
                stats_dict = {
                    f"{x_axis}": df[x_axis].describe(),
                    f"{y_axis}": df[y_axis].describe(),
                }
                stats_df = pd.concat(stats_dict, axis=1).T
                st.dataframe(stats_df)

                st.markdown("<br>" * 1, unsafe_allow_html=True)

                # Scatter Plot
                st.subheader(f"Scatter Plot: {x_axis} vs {y_axis}")
                custom_colors = px.colors.qualitative.Set1

                fig = px.scatter(
                    df, x=x_axis, y=y_axis,
                    color_discrete_sequence=custom_colors,
                    title=f"Scatter Plot of {y_axis} vs {x_axis}",
                    labels={x_axis: x_axis.capitalize(), y_axis: y_axis.capitalize()},
                    width=1000,
                    height=400
                )

                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=50)  # Adjust margins if necessary
                )

                st.plotly_chart(fig)


    with col2:

        st.subheader("Data Overview")
        st.write("**Total Records:** ", len(df))
        st.write("**Features:**", len(df.columns))
        st.write("**Columns:**", ', '.join(df.columns))


    # Define the questions and their corresponding answers
    qa_data = {
        "Q2": {
            "question": "Why Can Patients with High Serum Creatinine Levels Have Worse Outcomes Even If Their Ejection Fraction Is Normal?",
            "answer": """
            High serum creatinine is a significant biomarker for heart failure, as kidney dysfunction is common among these patients [8]. It is associated with an increased risk of mortality. However, a pertinent question arises: why are patients with normal ejection fractions but high serum creatinine levels at risk of death? 

            To explore this, we first need to establish the normal ranges for serum creatinine and ejection fraction:
            
            - **Normal Serum Creatinine Levels:**
                - Men: **0.74 to 1.35 mg/dL** [9]
                - Women: **0.59 to 1.04 mg/dL**
            
            - **Normal Ejection Fraction:** 
                - An ejection fraction of over **50%** is considered normal for both sexes [10].

            Based on the analysis of this subpopulation, the main findings are as follows:

            - **Anemia**: Patients with anemia are more likely to experience mortality, as they show a higher count in the "Dead" category compared to those who survived.
            
            - **Gender**: Being male is a significant factor in the risk of mortality.

            - **Serum Sodium Levels**: Those who passed away tended to have higher serum sodium levels on average, concentrated between **137-142 mmol/L**.

            - **Platelet Counts**: Survivors generally had higher platelet counts, while those who died had relatively lower counts. This suggests that lower platelet counts might be associated with worse outcomes.

            - **Diabetes**: The role of diabetes in mortality could not be clearly distinguished from the dataset.

            In summary, while high serum creatinine levels indicate kidney dysfunction and a risk factor for mortality, other factors such as anemia, gender, serum sodium levels, and platelet counts also play critical roles in determining patient outcomes.
            """
        },
        "Q3": {
            "question": "Why Do Some Patients Have Higher Serum Creatinine Levels Than Others?",
            "answer": """
            - **Sex (Boxplot)**: The boxplot of serum creatinine by sex shows that males tend to have higher serum creatinine levels than females. This could be related to differences in muscle mass, as creatinine is a byproduct of muscle metabolism.

            - **High Blood Pressure (Boxplot)**: The boxplot for high blood pressure reveals that patients with high blood pressure tend to have slightly higher serum creatinine levels. This is likely due to the long-term effects of high blood pressure on kidney function, leading to reduced filtration efficiency and thus higher serum creatinine.

            - **Diabetes (Boxplot)**: The boxplot for diabetes shows that diabetic patients tend to have higher serum creatinine levels. This could be due to diabetic nephropathy, a condition where high blood sugar levels damage the kidneys, reducing their ability to filter creatinine from the blood.

            - **Anaemia (Boxplot)**: Patients with anaemia seem to show slightly higher serum creatinine levels. Anaemia can be associated with kidney problems, as the kidneys help regulate red blood cell production. Impaired kidney function, which can increase creatinine levels, may lead to anaemia in some patients.

            - **Smoking (Boxplot)**: The boxplot for smoking status shows that smokers tend to have higher serum creatinine levels than non-smokers. Smoking is known to impair kidney function, likely due to the harmful effects of tobacco on the cardiovascular system, which in turn affects kidney filtration.

            - **Age (Scatter Plot)**: The scatter plot of serum creatinine vs. age suggests that older patients generally have higher serum creatinine levels. This could be due to the natural decline in kidney function with age, as the kidneys become less efficient at filtering waste from the blood over time.

            - **Ejection Fraction (Scatter Plot)**: The scatter plot of serum creatinine vs. ejection fraction indicates that patients with lower ejection fractions tend to have higher creatinine levels. Reduced heart function (low ejection fraction) can impair kidney function due to poor circulation and reduced kidney perfusion, leading to higher creatinine.

            - **Creatinine Phosphokinase (CPK) (Scatter Plot)**: The scatter plot of serum creatinine vs. CPK levels shows no clear linear trend, but patients with higher CPK levels (indicating muscle injury or stress) may have elevated creatinine due to increased muscle breakdown.

            - **Platelet Count (Scatter Plot)**: The scatter plot of serum creatinine vs. platelet count suggests no strong correlation, indicating that platelet levels may not significantly influence serum creatinine levels.

            - **Serum Sodium (Scatter Plot)**: The scatter plot of serum creatinine vs. serum sodium levels shows some trend where lower sodium levels could be associated with higher creatinine levels. Low sodium could be an indicator of kidney dysfunction, which correlates with higher creatinine.

            In summary, factors such as sex, high blood pressure, diabetes, anaemia, smoking, and age appear to be significant contributors to variations in serum creatinine levels, as visualized in the plots. This supports the idea that differences in kidney function and muscle metabolism, influenced by these variables, are key reasons for the higher creatinine levels in some patients.
            """
        }
    }

    selected_question = st.selectbox("Select a question to display:", options=list(qa_data.keys()))

    # Display the selected question and answer
    st.write(f"#### {qa_data[selected_question]['question']}")
    st.write(qa_data[selected_question]['answer'])

    st.markdown("<br>"*3, unsafe_allow_html=True)

import matplotlib.pyplot as plt
import plotly.figure_factory as ff

def show_correlation(df):
    st.subheader("Correlation Matrix")

    selected_features = df.columns.tolist()
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
        st.write(
                """
                #### Q1: Why Are Certain Heart Failure Patients at Risk of Death?

                **A1:** To answer the question "Why Are Certain Heart Failure Patients at Risk of Death?", we need to first observe the correlations between features and the mortality outcome. Correlation coefficients range from **-1 to 1**:
                - A coefficient close to **1** indicates a positive correlation, meaning as one variable increases, so does the other.
                - A coefficient close to **-1** indicates a negative correlation, meaning as one variable increases, the other decreases.
                - A coefficient around **0** suggests no correlation.

                From the heatmap analysis, we can understand that the following features have relatively stronger correlations with death:

                - **Serum Creatinine:** Positive correlation with DEATH_EVENT (**0.29**)
                -- Higher serum creatinine levels are associated with increased likelihood of mortality.
                
                - **Ejection Fraction:** Negative correlation with DEATH_EVENT (**-0.27**)
                -- A higher ejection fraction indicates a lower risk of mortality, highlighting the negative correlation.
                
                - **Time:** Negative correlation with DEATH_EVENT (**-0.53**)
                -- Longer time (days) a patient lives correlates with lower mortality risk, emphasizing the importance of intervention and care plans for heart failure patients.

                - **Age:** Positive correlation with DEATH_EVENT (**0.25**)
                -- Older patients are more likely to experience mortality.

                - **Serum Sodium:** Negative correlation with DEATH_EVENT (**-0.20**)
                -- Higher serum sodium levels indicate a lower likelihood of death.

                **Conclusion:** Certain heart failure patients are more likely to die due to:
                - Higher age
                - Higher serum creatinine levels
                - Lower ejection fraction
                - Lower serum sodium levels

                Time is also a significant factor, but its interpretation depends on the context of the patient's care.

                In scientific literature, higher serum creatinine, lower ejection fraction, and lower serum sodium (hyponatremia) are linked to an increased risk of mortality, validating the correlations found in this dataset [5-7].

                It's noteworthy that smoking, diabetes, anemia, and high blood pressure did not show strong correlations to mortality in this dataset. This suggests that these features alone are not sufficient to determine the mortality risk for patients.
                """
            )

        
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
from sklearn.model_selection import train_test_split

def show_model_performance(df):
    
    all_features = df.columns.tolist()
    all_features.remove("DEATH_EVENT")

    # Preparing data
    X = df[all_features] 
    y = df["DEATH_EVENT"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Apply standardization

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify = y, test_size=0.30, random_state=25)
    
    model = joblib.load("Web medical dashboard/xgb3_model.pkl")  # Load the model
    model.fit(X_train, y_train)

    # Assuming X_scaled is already defined elsewhere in your code
    # Make predictions
    predictions = model.predict(X)
    predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
    df = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)

    st.header("SHAP Analysis")
    
    y_np = y.values
    class_labels = np.unique(y_np)

    # Find the index of the positive class
    positive_class_index = np.where(class_labels == 1)[0]
    if len(positive_class_index) == 0:
        st.error("There are no positive labels.")
    else:
        positive_class_index = positive_class_index[0]

    # Initialize SHAP TreeExplainer for XGBoost model
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)

    # Access the shap values
    shap_values_array = shap_values.values  

    # Check the ndim attribute of the SHAP values array
    if shap_values_array.ndim == 3:
        shap_values_array = shap_values_array[:, :, positive_class_index]
    elif shap_values_array.ndim == 1:
        shap_values_array = shap_values_array.reshape(-1, 1)

    # Ensure SHAP values are 2D
    if shap_values_array.ndim == 2:
        feature_importances = np.abs(shap_values_array).mean(axis=0)
    else:
        st.error("SHAP values should be 2D.")

    # Sort the feature importances
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_feature_names = [all_features[i] for i in sorted_indices]

    # Display SHAP analysis in Streamlit
    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Summary Plot")
        X_scaled_df = pd.DataFrame(X_scaled, columns=all_features)
        shap.summary_plot(shap_values_array, X_scaled_df, feature_names=all_features, show=False)
        st.pyplot(plt)

    with right_column:
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


if __name__ == "__main__":
    show_dashboard()