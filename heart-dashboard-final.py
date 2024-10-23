## "pip install -r requirements.txt" in terminal first##
import os
os.system('pip install streamlit')
os.system('pip install pandas')
os.system('pip install numpy')
os.system('pip install plotly')
os.system('pip install Pillow')
os.system('pip install scikit-learn')
os.system('pip install joblib')
os.system('pip install shap')
os.system('pip install matplotlib')

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
    image_path_2 = r"/assets/heart_to_say.png"
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
        option = st.radio("Select an option:", ["Home","Overview of Patients", "Factors Correlation","Mortality Risk Prediction", "Contact Us"])
    
    df = pd.read_csv("Web medical dashboard/heart_failure_clinical_records_dataset.csv")
    df.rename(columns={ 'time': 'follow-up days',
                        "DEATH_EVENT": "mortality",
                        "creatinine_phosphokinase": "creatinine phosphokinase",
                        "ejection_fraction": "ejection fraction",
                        "serum_creatinine": "serum creatinine",
                        "serum_sodium": "serum sodium",
                        "high_blood_pressure": "hypertension"
                        }, inplace=True)
                        
    if option == "Overview of Patients":
        show_data_overview(df)
    elif option == "Factors Correlation":
        show_eda(df)
    elif option == "Contact Us":
        show_contact_us()
    elif option == "Mortality Risk Prediction":
        with st.sidebar:
            sub_option = st.radio("Choose an action:", ["Input your data", "Model explanation (SHAP)"])
        if sub_option == "Input your data":
            upload_pre_model()
        elif sub_option == "Model explanation (SHAP)":
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
            sex = st.selectbox("**Sex**", options=["Male", "Female"], index=0 if st.session_state.get("sex") != "Female" else 1)
            smoking = st.selectbox("**Smoking (Yes/No)**", options=["Yes", "No"], index=0 if st.session_state.get("smoking") != "Yes" else 1)
            time = st.number_input("**Follow-up Period (days)**", min_value=0, value=st.session_state.get("time", 0))
            
        with col2:
            high_blood_pressure = st.selectbox("**Hypertension (Yes/No)**", options=["Yes", "No"], index=0 if st.session_state.get("high_blood_pressure") != "Yes" else 1)
            anaemia = st.selectbox("**Anaemia (Yes/No)**", options=["Yes", "No"], index=0 if st.session_state.get("anaemia") != "Yes" else 1)
            diabetes = st.selectbox("**Diabetes (Yes/No)**", options=["Yes", "No"], index=0 if st.session_state.get("diabetes") != "Yes" else 1)
            ejection_fraction = st.number_input("**Ejection Fraction (%)**", min_value=0.0, max_value=100.0, format="%.2f", value=st.session_state.get("ejection_fraction", 0.0))
            
        with col3:
            serum_creatinine = st.number_input("**Serum Creatinine (mg/dL)**", min_value=0.0, format="%.2f", value=st.session_state.get("serum_creatinine", 0.0))
            serum_sodium = st.number_input("**Serum Sodium (mEq/L)**", min_value=0.0, format="%.2f", value=st.session_state.get("serum_sodium", 0.0))
            creatinine_phosphokinase = st.number_input("**Creatinine Phosphokinase (mcg/L)**", min_value=0.0, format="%.2f", value=st.session_state.get("creatinine_phosphokinase", 0.0))
            platelets = st.number_input("**Platelets (kiloplatelets/mL)**", min_value=0, value=st.session_state.get("platelets", 0))
            
    
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
                    Prediction Outcome
                </h2>
                <hr style="
                    border: 0;
                    height: 2px;
                    background: #3498db;
                    margin: 20px 0;
                ">
                <p style="
                font-size: 24px;  
                margin: 10px 0;
                font-weight: bold;
            ">
                Risk Level: 
                <span style="
                    font-weight: bold;
                    font-size: 24px;  
                    color: {'#e74c3c' if prediction[0] == 1 else '#27ae60'};
                ">
                    {'High Risk' if prediction[0] == 1 else 'Low Risk'}
                </span>
                </p>
            </div>
            """, unsafe_allow_html=True)


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
    show_history = st.checkbox("Show Input History")
    if show_history:
        st.subheader("Input History")
        for idx, record in enumerate(st.session_state["input_history"]):
            st.markdown(format_record(record, idx), unsafe_allow_html=True)
    st.markdown("<br><br><br>", unsafe_allow_html=True) 

import streamlit as st
import pandas as pd

def show_data_overview(df):
    st.title("Overview of Patients")
    
    # Dataset basic info
    total_records = len(df)
    positive_cases = df['mortality'].value_counts().get(1, 0)
    negative_cases = df['mortality'].value_counts().get(0, 0)

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
                                  description="This represents the total number of patient records."), unsafe_allow_html=True)

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
        if 'mortality' in categorical_features:
            categorical_features.remove('mortality')
    
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
                1: f"{selected_column}"
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

        fig_feature_2 = px.histogram(df, x=selected_column, color='mortality', barmode='group',
                    color_discrete_map={0: '#808080', 1: '#ff0000'},
                    title=f'Distribution of {selected_column} vs Death Event' )
        fig_feature_2.for_each_trace(lambda t: t.update(name='Survived' if t.name == '0' else 'Death occured'))
        st.plotly_chart(fig_feature_2, use_container_width=True)

        with right_column:

            selected_column = st.selectbox("Select a continual feature to visualize (e.g., age, platelets, etc)", numerical_features)
            if 'mortality' in numerical_features:
                numerical_features.remove('mortality')

            fig_feature_1 = px.histogram(df, x=selected_column, barmode='group',
                                    color_discrete_sequence=['#2ca02c'],
                                        title=f'Distribution of {selected_column}')
            fig_feature_1.update_layout(bargap=0.2)
            
            fig_feature_2 = px.histogram(df, x=selected_column, color='mortality', barmode='group',
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
    st.title("Heart Failures Factors Correlation")
    st.markdown("---")
    
    show_correlation(df)

import matplotlib.pyplot as plt
import plotly.figure_factory as ff

def show_correlation(df):
    st.subheader("Correlation Matrix")
    st.write("A correlation matrix shows how heart failure factors are related to each other in a simple table.")
    
    st.markdown("")
    df.rename(columns={"mortality": "mortality risk"}, inplace=True)
    selected_features = df.columns.tolist()
    selected_features = [feature for feature in df.columns if feature != 'mortality risk']
    target_variable = 'mortality risk'  
    
    if target_variable not in selected_features:
        selected_features.append(target_variable)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <div style="margin: 10px;">
                    <br><br><br>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        selected_feature = st.selectbox("Select a factor to view its correlation with mortality risk:", selected_features,
                                    index=selected_features.index("diabetes"))
    with col2:
        if len(selected_features) > 0:
            
            corr_df = df[selected_features].corr()

            target_correlation = corr_df[target_variable][selected_feature]

            st.write(f"The correlation Coefficient between {selected_feature} and {target_variable} is: {target_correlation:.2f}")
            color_map = {
                'Very high positive correlation': '#e74c3c',  
                'High positive correlation': '#d45d27',       
                'Moderate positive correlation': '#e6a900',   
                'Low positive correlation': '#f0e68c',       
                'Negligible correlation': '#bdc3c7',          
                'Very high negative correlation': '#c0392b', 
                'High negative correlation': '#8e44ad',       
                'Moderate negative correlation': '#2980b9',   
                'Low negative correlation': '#3498db',       
            }

            if target_correlation >= 0.9:
                interpretation = 'Very high positive correlation'
                color = color_map[interpretation]
            elif target_correlation >= 0.7:
                interpretation = 'High positive correlation'
                color = color_map[interpretation]
            elif target_correlation >= 0.5:
                interpretation = 'Moderate positive correlation'
                color = color_map[interpretation]
            elif target_correlation >= 0.3:
                interpretation = 'Low positive correlation'
                color = color_map[interpretation]
            elif target_correlation >= 0.0:
                interpretation = 'Negligible correlation'
                color = color_map[interpretation]
            elif target_correlation <= -0.9:
                interpretation = 'Very high negative correlation'
                color = color_map[interpretation]
            elif target_correlation <= -0.7:
                interpretation = 'High negative correlation'
                color = color_map[interpretation]
            elif target_correlation <= -0.5:
                interpretation = 'Moderate negative correlation'
                color = color_map[interpretation]
            elif target_correlation <= -0.3:
                interpretation = 'Low negative correlation'
                color = color_map[interpretation]
            else:
                interpretation = 'Negligible correlation'
                color = color_map[interpretation]

            st.markdown(
                f"""
                <div style="background-color: {color}; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); width: 700px;">
                    <h3 style="color: white; font-family: 'Arial', sans-serif; font-size: 22px; margin-bottom: 10px;">Correlation Interpretation</h3>
                    <p style="color: white; font-family: 'Arial', sans-serif; font-size: 18px; margin-bottom: 5px;">
                        The correlation between 
                        <strong style="color: yellow; font-size: 24px;">{selected_feature}</strong> 
                        and 
                        <strong style="color: white; text-shadow: 1px 1px 2px black; font-size: 24px;">{target_variable}</strong> 
                        is:
                    </p>
                    <h2 style="color: #FFD700; font-family: 'Arial', sans-serif; font-size: 26px; font-weight: bold; text-shadow: 1px 1px 2px black;">
                        {interpretation}
                    </h2>
                </div>
                """, unsafe_allow_html=True
            )
    st.markdown("<br>" *2, unsafe_allow_html=True)
        
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
       
    else:
        st.warning("Please select at least one feature for correlation analysis.")
    
    st.write("The heat map illustrates the correlation between cardiovascular disease-related data features, revealing a positive correlation between serum creatinine and mortality, as well as between age and mortality, suggesting that an increase in one factor tends to lead to an increase in the other. Conversely, the number of follow-up days and ejection fraction exhibit a negative correlation with mortality, implying that an increase in one factor is associated with a decrease in the other. These observations offer valuable insights into the risk factors for heart disease.")
    st.markdown("<br>"*3, unsafe_allow_html=True)

import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import io
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def show_model_performance(df):
    
    all_features = df.columns.tolist()
    all_features.remove("mortality")

    # Preparing data
    X = df[all_features] 
    y = df["mortality"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Apply standardization

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify = y, test_size=0.30, random_state=25)

    model = joblib.load("Web medical dashboard/xgb3_model.pkl")
    model.fit(X_train, y_train)
    # Make predictions
    predictions = model.predict(X_test)
    
    # Create a DataFrame for SHAP analysis
    X_test_df = pd.DataFrame(X_test, columns=all_features)
    X_test_df['Predictions'] = predictions
    X_test_reduced = X_test_df.iloc[:, :12] 

    # SHAP analysis
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_test_reduced)

    st.header("SHAP Analysis")

    y_np = y_test.values
    class_labels = np.unique(y_np)

    # Find the index of the positive class
    positive_class_index = np.where(class_labels == 1)[0]
    if len(positive_class_index) == 0:
        st.error("There are no positive labels.")
    else:
        positive_class_index = positive_class_index[0]

    # Check the shape of SHAP values
    if isinstance(shap_values, np.ndarray):
        shap_values_array = shap_values  
    else:
        shap_values_array = shap_values.values  

    if shap_values_array.ndim == 3:  # If it's a 3D array
        shap_values_array = shap_values_array[:, :, 1]  # Focus on positive class if applicable
    elif shap_values_array.ndim == 1:
        shap_values_array = shap_values_array.reshape(-1, 1)
    # Access the SHAP values
    shap_values_array = shap_values.values

    # Feature importances
    feature_importances = np.abs(shap_values_array).mean(axis=0)
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_feature_names = [all_features[i] for i in sorted_indices]

    left_column, right_column = st.columns(2)

    # SHAP Summary Plot
    with left_column:
        st.subheader("Summary Plot")
        shap.summary_plot(shap_values_array, X_test_reduced, feature_names=all_features, show=False)
        st.pyplot(plt)
        
    # Feature Importance Bar Plot
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

    # Create a summary card to display SHAP analysis results 
    top_contributions = shap_importance_df.nlargest(3, 'Importance')

    shap_summary = f"""
    <div style="background-color: #ffffff; padding: 30px; border-radius: 15px; box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2); transition: transform 0.2s;">
        <h2 style="color: #2c3e50; font-family: 'Arial', sans-serif; margin-bottom: 15px;">SHAP Analysis Results Summary</h2>
        <p style="font-family: 'Arial', sans-serif; font-size: 16px; color: #34495e; margin-bottom: 20px;">
            The SHAP analysis provides crucial insights into how each factor influence the model's predictions regarding mortality risk.
        </p>
        <h3 style="color: #2980b9; font-family: 'Arial', sans-serif; font-size: 22px; margin-bottom: 10px;">Key Important Factors:</h3>
        <ul style="font-family: 'Arial', sans-serif; color: #444; list-style-type: circle; padding-left: 20px;">
            {"".join(f"<li style='margin-bottom: 5px;'><strong>{row['Feature']}</strong>: Importance Score = {row['Importance']:.4f}</li>" for _, row in top_contributions.iterrows())}
        </ul>
        <h3 style="color: #2980b9; font-family: 'Arial', sans-serif; font-size: 22px; margin-top: 20px; margin-bottom: 10px;">Insights:</h3>
        <p style="font-family: 'Arial', sans-serif; font-size: 16px; color: #555;">
            The plots suggest that <span style="color: red;"><strong>{top_contributions.iloc[0]['Feature']}</strong></span> had the <strong>greatest impact on the model output</strong>, followed by <span style="color: red;"><strong>{top_contributions.iloc[1]['Feature']}</strong></span> and <span style="color: red;"><strong>{top_contributions.iloc[2]['Feature']}</strong></span>.
        </p>
        <p style="font-family: 'Arial', sans-serif; font-size: 16px; color: #555; margin-top: 20px;">
            Understanding these contributions is vital for interpreting the model's behavior and making informed decisions based on the predictions.
            For more about SHAP score, go to the following link: 
        <a href="https://selfexplainml.github.io/PiML-Toolbox/_build/html/guides/explain/shap.html" style="color: #2980b9; text-decoration: underline;">
            SHAP Score Guide
        </p>
    </div>
    """

    # Hover effect to scale up the card
    style = """
    <style>
        div:hover {
            transform: scale(1.02);
        }
    </style>
    """

    st.markdown(shap_summary, unsafe_allow_html=True)

    st.markdown("<br>"*3, unsafe_allow_html=True)
    

if __name__ == "__main__":
    show_dashboard()