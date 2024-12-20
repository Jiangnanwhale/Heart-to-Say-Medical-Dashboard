## "pip install -r requirements.txt" in terminal first##
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import pointbiserialr

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
    **For any inquiries or support, please reach out to us at:**
    
    **Email**: heart_to_say_team@dsv.su.se
                
    **Phone**: +46 123456789
                
    **Group Members**:  
    - Ifani Pinto Nada  
    - Mahmoud Elachi  
    - Nan Jiang  
    - Sahid Hasan Rahim  
    - Zhao Chen  
    
      
    **Data Resource[1,2]**:  
    [Heart Failure Clinical Data](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)
                  
    **Project Information**:  
    [Heart to Say Medical Dashboard GitHub](https://github.com/Jiangnanwhale/Heart-Health-Caring-Team)
                
    **Problem Description**: 

    Cardiovascular disease (CVD) is the leading cause of death worldwide, with an estimated 20.5 million deaths reported in 2021 [3, 4]. Among CVD conditions, heart failure presents a survival prognosis comparable to that of severe cancers [3, 5]. Key risk factors include smoking [6], anemia [7–10], diabetes [11–13], ejection fraction, and hypertension [6, 14, 15]. Elderly patients [6, 16] and men [6] are often more represented among those who die from heart failure. Accurate forecasting in heart failure patients is crucial for preventing mortality [6, 17]. 
    
    **Project Scope and Purpose**:
    
    The Heart to Say project aims to develop a prediction model to estimate the likelihood of mortality in heart failure patients in Pakistan, using a dataset adopted from Chicco D et al. that focuses on the aforementioned subgroup [1,2]. Focusing on Pakistan is important, as heart failure patients in lower-income countries are 3-5 times more likely to die within 30 days of the first hospital admission compared to high-income countries, even after accounting for patient differences and long-term treatments [18]. Pakistan is categorized as a low-middle income country according to the Organisation for Economic Co-operation and Development [19]. Hence, the goal is to deliver a web dashboard for general practitioners, cardiologists and cardiac nurses in Pakistan to:
    1. Predict the likelihood of mortality due to heart failure.
    2. Display risk factors and severity levels.
    3. Aid in clinical decision-making to address mortality risk factors effectively.
                
    **Design Process**: 
    1. Team Rules: Document, Paper prototype.        
    2. Project Charter: Document, Digital prototype and Preprocessing dataset.         
    3. Project Delivery: Web medical dashboard, Video showcase and Final project report.      

    **Project Stakeholders**:
    - Karolinska Institutet, Sweden (in collaboration with Aga Khan University, Pakistan)
    - Ministry of National Health Services, Regulations and Coordination, Pakistan
    - Pakistan Medical Commission, Pakistan
    - Pakistan Cardiac Society, Pakistan
    - The Swedish International Development Cooperation Agency, Sweden

    **References**:  
    1. D C, G J. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Med Inform Decis Mak. 2020 Mar 2;20(1):16.
    2. Kaggle. Heart Failure Prediction [Internet]. Kaggle. [cited 2024 Sep 11]. Available from: https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data
    3. Spitaleri G, Zamora E, Cediel G, Codina P, Santiago-Vacas E, Domingo M, et al. Cause of Death in Heart Failure Based on Etiology: Long-Term Cohort Study of All-Cause and Cardiovascular Mortality. J Clin Med. 2022 Jan 31;11(3):784.
    4. Savarese G, Becher PM, Lund LH, Seferovic P, Rosano GMC, Coats AJS. Global burden of heart failure: a comprehensive and updated review of epidemiology. Cardiovasc Res. 2023 Jan 18;118(17):3272–87.
    5. Mamas MA, Sperrin M, Watson MC, Coutts A, Wilde K, Burton C, et al. Do patients have worse outcomes in heart failure than in cancer? A primary care-based cohort study with 10-year follow-up in Scotland. Eur J Heart Fail. 2017 Sep;19(9):1095–104.
    6. Bozkurt B, Ahmad T, Alexander KM, Baker WL, Bosak K, Breathett K, et al. Heart Failure Epidemiology and Outcomes Statistics: A Report of the Heart Failure Society of America. J Card Fail. 2023 Oct;29(10):1412–51.
    7. Köseoğlu FD, Özlek B. Anemia and Iron Deficiency Predict All-Cause Mortality in Patients with Heart Failure and Preserved Ejection Fraction: 6-Year Follow-Up Study. Diagnostics. 2024 Jan;14(2):209.
    8. Siddiqui SW, Ashok T, Patni N, Fatima M, Lamis A, Anne KK. Anemia and Heart Failure: A Narrative Review. Cureus. 2022;14(7):e27167.
    9. Groenveld HF, Januzzi JL, Damman K, van Wijngaarden J, Hillege HL, van Veldhuisen DJ, et al. Anemia and Mortality in Heart Failure Patients: A Systematic Review and Meta-Analysis. Journal of the American College of Cardiology. 2008 Sep 2;52(10):818–27.
    10. Xia H, Shen H, Cha W, Lu Q. The Prognostic Significance of Anemia in Patients With Heart Failure: A Meta-Analysis of Studies From the Last Decade. Front Cardiovasc Med. 2021 May 13;8:632318.
    11. Dunlay SM, Givertz MM, Aguilar D, Allen LA, Chan M, Desai AS, et al. Type 2 Diabetes Mellitus and Heart Failure: A Scientific Statement From the American Heart Association and the Heart Failure Society of America. Circulation. 2019 Aug 13;140(7):e294–324.
    12. Jr P, Tj G, Rm T. Diabetes, Hypertension, and Cardiovascular Disease: Clinical Insights and Vascular Mechanisms. The Canadian journal of cardiology. 2018 May;34(5). Available from: https://pubmed.ncbi.nlm.nih.gov/29459239/
    13. Siao WZ, Chen YH, Tsai CF, Lee CM, Jong GP. Diabetes Mellitus and Heart Failure. J Pers Med. 2022 Oct 11;12(10):1698.
    14. Oh GC, Cho HJ. Blood pressure and heart failure. Clin Hypertens. 2020 Jan 2;26:1.
    15. Triposkiadis F, Sarafidis P, Briasoulis A, Magouliotis DE, Athanasiou T, Skoularigis J, et al. Hypertensive Heart Failure. Journal of Clinical Medicine. 2023 Jan;12(15):5090.
    16. Krittayaphong R, Karaketklang K, Yindeengam A, Janwanishstaporn S. Heart failure mortality compared between elderly and non-elderly Thai patients. J Geriatr Cardiol. 2018 Dec;15(12):718–24.
    17. Goff DC, Brass L, Braun LT, Croft JB, Flesch JD, Fowkes FGR, et al. Essential Features of a Surveillance System to Support the Prevention and Management of Heart Disease and Stroke. Circulation. 2007 Jan 2;115(1):127–55.
    18. G-CHF Investigators, Joseph P, Roy A, Lonn E, Störk S, Floras J, et al. Global Variations in Heart Failure Etiology, Management, and Outcomes. JAMA. 2023 May 16;329(19):1650–61.
    19. Organisation for Economic Co-operation and Development. DAC List of ODA Recipients | Effective for reporting on 2024 and 2025 flows [Internet]. OECD. 2023 [cited 2024 Oct 17]. Available from: https://www.oecd.org/content/dam/oecd/en/topics/policy-sub-issues/oda-eligibility-and-conditions/DAC-List-of-ODA-Recipients-for-reporting-2024-25-flows.pdf

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
        option = st.radio("Select an option:", ["Home","Patients Data Overview", "Data Analysis","Mortality Risk Prediction", "Contact Us"])
    
    df = pd.read_csv("Web medical dashboard/heart_failure_clinical_records_dataset.csv")
    df.rename(columns={ 'time': 'follow-up days',
                        "DEATH_EVENT": "mortality",
                        "creatinine_phosphokinase": "creatinine phosphokinase",
                        "ejection_fraction": "ejection fraction",
                        "serum_creatinine": "serum creatinine",
                        "serum_sodium": "serum sodium",
                        "high_blood_pressure": "hypertension"
                        }, inplace=True)
                        
    if option == "Patients Data Overview":
        show_data_overview(df)
    elif option == "Contact Us":
        show_contact_us()
    elif option == "Data Analysis":
        st.title("Data Analysis")
        sub_option = st.radio("Choose an option:", ["Factors Correlation", "Group Identification"])
        if sub_option == "Factors Correlation":
            show_eda(df)  
        elif sub_option == "Group Identification":
            show_clustering_analysis(df) 
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
    st.title("💖 Welcome to the Heart to Say Web Dashboard")
    st.markdown("---")
    
    st.markdown(
    """
    **This dashboard supports General Practitioners, Cardiologists and Cardiac Nurses in predicting the risk of mortality due to heart failure.** 
    """
    )
    
    st.markdown("### Dashboard Features")
    st.markdown(
        """We aim to provide insights for better clinical decision-making by utilising patient data and advanced analytics."""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            - **🏠 Home**: 
            Provides an overview of the dashboard's functionality.
            
            - **📊 Patients Data Overview**: 
            Explore heart failure patient's data, enabling you to view trends and prevalence based on:
                - Age and gender
                - Smoking status
                - Comorbidities
                - Laboratory test results
            """
        )
    with col2:
        st.markdown(
            """
            - **🔍 Data Analysis**: 
            Analyze correlations and patterns between heart failure risk factors and mortality to provide a comprehensive overview. Explore specific patient characteristics to identify groups for adverse health outcomes based on our clustering analysis.
            
            - **🤖 Mortality Risk Prediction**: 
            Input patient data on heart failure risk factors to predict the risk level of mortality.
            
            - **📞 Contact Us**: 
            Get in touch for more information about the project, our team, and how to reach us.
            """
        )
    st.markdown("<br><br><br>", unsafe_allow_html=True) 
    
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
        model_path = os.path.join(os.getcwd(), "xgb3_model.pkl")
        model = joblib.load(model_path)
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

def show_data_overview(df):
    st.title("Patients Data Overview")
    
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
        st.markdown(card_style.format(bg_color="#808080", title="Total Records", value=total_records,
                                  description="This represents the total number of patient records."), unsafe_allow_html=True)

    with col2:
        st.markdown(card_style.format(bg_color="#ff0000", title="Death Cases", value=positive_cases,
                                    description="Number of patients who experienced a death event."), unsafe_allow_html=True)

    with col3:
        st.markdown(card_style.format(bg_color="#2ca02c", title="Survival Cases", value=negative_cases,
                                    description="Number of patients who did not experience a death event."), unsafe_allow_html=True)

    
    st.markdown("")
    st.markdown("")

    st.subheader("Patients Data Overview")
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
    
        selected_column = st.selectbox("Select a factor to visualize from the following options: diabetes, anaemia, hypertension, sex, smoking, mortality.", categorical_features,
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
        
        if selected_column == 'sex':
            conclusion_text = f"Based on the dataset, {selected_column_percentage}% are Male, while {opposite_percentage}% are Female."
        else:
            conclusion_text = f"Based on the dataset, {selected_column_percentage}% are {selected_column}, while {opposite_percentage}% belong to the opposite category."
        st.write(conclusion_text)

        df_copy = df.copy()   
        df_copy[selected_column] = df_copy[selected_column].map(label_map)
        fig_feature_2 = px.histogram(
            df,
            x=selected_column,
            color='mortality',
            barmode='group',
            color_discrete_map={0: '#808080', 1: '#ff0000'},
            title=f'Distribution of {selected_column} vs Death Event'
        )

        def update_trace_names(trace):
            if trace.name == '0':
                trace.update(name='Survived')
            else:
                trace.update(name='Death occurred')

        fig_feature_2.for_each_trace(update_trace_names)

        if selected_column == 'sex':
            x_label_map = {0: 'Female', 1: 'Male'}
        else:
            x_label_map = {
                0: f'No {selected_column}', 
                1: f'{selected_column}'
            }

        fig_feature_2.update_xaxes(
            title=selected_column,
            tickvals=[0, 1],  
            ticktext=[x_label_map[0], x_label_map[1]],  
            tickmode='array' 
        )

        st.plotly_chart(fig_feature_2, use_container_width=True)



        with right_column:

            selected_column = st.selectbox("Select a factor to visualize from the following options: age, creatinine phosphookianse, ejection fraction, platelets, serum creatinine, serum sodium, follow-up days.", numerical_features)
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

def show_correlation(df):

    st.subheader("Correlation Matrix")
    st.write("A correlation matrix shows how heart failure factors are related to each other in a simple table.")

    # Prepare your DataFrame
    selected_features = [feature for feature in df.columns if feature != 'mortality risk']
    target_variable = 'mortality'

    if target_variable not in df.columns:
        st.error(f"Target variable '{target_variable}' not found in the DataFrame.")
        return
    # Convert categorical variables to numerical
    for column in df.columns:
        if df[column].dtype == object and set(df[column].dropna().unique()).issubset({'Yes', 'No'}):
            df[column] = df[column].map({'Yes': 1, 'No': 0})
        elif column == 'sex' and df['sex'].dtype == object:
            df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

    # Extract binary and numerical features
    binary_features = [col for col in df.columns if set(df[col].dropna().unique()).issubset({0, 1})]
    numerical_features = [col for col in df.select_dtypes(include=np.number).columns if col not in binary_features]

    correlation_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

    for feature1 in binary_features:
        for feature2 in binary_features:
            if feature1 != feature2:
                correlation_matrix.loc[feature1, feature2] = np.corrcoef(df[feature1], df[feature2])[0, 1]
            else:
                correlation_matrix.loc[feature1, feature2] = 1  

    for binary in binary_features:
        for numerical in numerical_features:
            correlation_matrix.loc[binary, numerical] = pointbiserialr(df[binary], df[numerical])[0]
            correlation_matrix.loc[numerical, binary] = pointbiserialr(df[binary], df[numerical])[0]

    for feature1 in numerical_features:
        for feature2 in numerical_features:
            correlation_matrix.loc[feature1, feature2] = df[[feature1, feature2]].corr().iloc[0, 1]

    correlation_matrix = correlation_matrix.astype(float)

    # Plotly Heatmap
    fig = px.imshow(correlation_matrix, 
                    text_auto=True, 
                    aspect="auto", 
                    color_continuous_scale='RdBu', 
                    range_color=[-1, 1], 
                    title='Correlation Matrix',
                    labels=dict(x="Features", y="Features"))

    # Feature Selection and Correlation Interpretation
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
            if target_variable in corr_df.columns:
                target_correlation = corr_df[target_variable][selected_feature]
                st.write(f"The correlation coefficient between {selected_feature} and {target_variable} is: {target_correlation:.2f}")
            else:
                st.warning(f"Target variable '{target_variable}' not found in correlation DataFrame.")

            # Interpret correlation
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

    st.plotly_chart(fig)

    st.write("The heat map illustrates the correlation between heart failure-related data features. It reveals a positive correlation between serum creatinine and mortality and between age and mortality, suggesting that an increase in one factor tends to lead to an increase in the other. Conversely, the number of follow-up days and ejection fraction exhibits a negative correlation with mortality, implying that an increase in one factor is associated with a decrease in the other. These observations offer valuable insights into the risk factors for heart failure.")
    st.markdown("<br>" * 3, unsafe_allow_html=True)


def show_clustering_analysis(df):
    st.title("Group Identification")
    st.markdown("In this section, you can choose specific factors to help identify patient characteristics using our clustering analysis.")

    allowed_features = [
                    "age",
                    "creatinine phosphokinase",
                    "ejection fraction",
                    "platelets",
                    "serum creatinine",
                    "serum sodium",
                    "follow-up days"
                        ]
    
    left_column, right_column = st.columns(2)
    with left_column:
        feature1 = st.selectbox('Select First Factor for Clustering:', allowed_features, index=allowed_features.index("age"))
    with right_column:
        feature2 = st.selectbox('Select Second Factor for Clustering:', allowed_features, index=allowed_features.index("follow-up days"))

    if feature1 == feature2:
        st.warning("Please select different factors for clustering.")
    else:
        selected_features = [feature1, feature2]

        if len(selected_features) > 0:
            selected_df = df[selected_features]

            n_clusters = 2
            kmeans = KMeans(n_clusters=n_clusters)
            df['Cluster'] = kmeans.fit_predict(selected_df)

            if selected_df.shape[1] >= 2:
                # Standardize the selected features before PCA
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                selected_df_scaled = scaler.fit_transform(selected_df)

                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(selected_df_scaled)

                pca_df = pd.DataFrame(data=pca_result, columns=['PCA Component 1', 'PCA Component 2'])
                pca_df['Cluster'] = df['Cluster'].astype(str)
                
                pca_df[feature1] = selected_df[feature1].values
                pca_df[feature2] = selected_df[feature2].values
                color_map = {
                        '0': 'green',   
                        '1': 'red',     
                    }
                fig = px.scatter(
                    pca_df, 
                    x=feature2, 
                    y=feature1, 
                    color='Cluster', 
                    title="K-Means Clustering",
                    color_discrete_map=color_map,
                    labels={"color": "Cluster"}
                )
                fig.update_layout(title_x=0.5)
                st.plotly_chart(fig)
            
                cluster_risks = df.groupby('Cluster')['mortality'].mean()
                
                risk_df = cluster_risks.reset_index()
                risk_df.columns = ['Cluster', 'Average Mortality Risk']
                
                # Determine high-risk and low-risk clusters
                risk_threshold = risk_df['Average Mortality Risk'].median()
                risk_df['Risk Level'] = risk_df['Average Mortality Risk'].apply(lambda x: 'High Risk' if x > risk_threshold else 'Low Risk')
                
                st.subheader("Cluster Overview")
                for index, row in risk_df.iterrows():
                    with st.expander(f"**Cluster {row['Cluster']} Overview**"):
                        # Show more details about the cluster
                        cluster_data = df[df['Cluster'] == row['Cluster']]
                    
                        st.write(f"<span style='color: #007bff;'><strong>Total Members in Cluster:</strong> {len(cluster_data)}</span>", unsafe_allow_html=True)
                    
                        st.write("### Key Insights:")
                        st.write("The average values of the top five factors in this cluster provide insights into the typical patient profile.")
                        
                        # Get the top 5 features by mean value
                        important_features = cluster_data.mean().nlargest(5)
                    
                        for feature, value in important_features.items():
                            st.write(f"<span style='color: #28a745;'><strong>{feature}:</strong> {value:.2f}</span>", unsafe_allow_html=True)


        st.markdown("<br>"*3, unsafe_allow_html=True)


def show_model_performance(df):
    
    all_features = df.columns.tolist()
    all_features.remove("mortality")

    # Preparing data
    X = df[all_features] 
    y = df["mortality"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Apply standardization

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify = y, test_size=0.30, random_state=25)

    model_path = os.path.join(os.getcwd(), "xgb3_model.pkl")
    model = joblib.load(model_path)
    model.fit(X_train, y_train)
    # Make predictions
    predictions = model.predict(X_test)
   
    # Create a DataFrame for SHAP analysis
    X_test_df = pd.DataFrame(X_test, columns=all_features)
    X_test_df['Predictions'] = predictions
    X_test_reduced = X_test_df.iloc[:, :-1] 

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
        st.markdown("")
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
            The SHAP analysis provides crucial insights into how each factor influences the model's predictions regarding mortality risk.
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
