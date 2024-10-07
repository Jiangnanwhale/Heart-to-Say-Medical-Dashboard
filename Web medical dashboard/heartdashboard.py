import streamlit as st
import pandas as pd
import plotly.express as px

# Function to simulate login
def login(username, password):
    # Always allow login regardless of the credentials
    st.session_state["logged_in"] = True

def login_page():
    st.title(":key: Login to Your Account")

    with st.container():
        st.subheader("Please enter your credentials")

        # Create a form layout
        with st.form(key='login_form'):
            col1, col2 = st.columns([2, 1])

            with col1:
                username = st.text_input("Username", key="username_input")
                password = st.text_input("Password", type="password", key="password_input")
            with col2:
                # This column can be used for spacing or additional content
                st.write("")  # Empty content or other UI elements can be added here
                # Add an image to the right column
                st.image("https://my.clevelandclinic.org/-/scassets/images/org/health/articles/21704-heart-overview-outside", use_column_width=True)

            
            # Button placement within the form
            login_button = st.form_submit_button("Login")

            if login_button:
                login(username, password)
                if st.session_state.get("logged_in", False):
                    st.success("Login successful")

                    # Display "Proceed to Dashboard" button
                    st.session_state["show_dashboard_button"] = True
                else:
                    st.error("Invalid credentials")

    if st.session_state.get("show_dashboard_button", False):
        if st.button("Proceed to Dashboard"):
            # Handle redirection to dashboard
            st.session_state["page"] = "dashboard"

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
    st.write("Welcome to the dashboard!")

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
                
            st.sidebar.subheader("Dataset Summary")
            st.sidebar.write(f"**Total Records:** {len(df)}")
            st.sidebar.write(f"**Positive Cases:** {df['DEATH_EVENT'].value_counts().get(1, 0)}")
            st.sidebar.write(f"**Negative Cases:** {df['DEATH_EVENT'].value_counts().get(0, 0)}")

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
        option = st.sidebar.radio("Select an option:", [ "Show Data", "Show Histogram","Show Correlation Matrix"])

    # First section: Show the dataset
    st.subheader("Dataset Overview")
    st.write(df.head())

    # Columns layout for multiple data analysis sections
    col1, col2 = st.columns(2)

    # First column: Age Distribution of Death Events
    death = df[df['DEATH_EVENT'] == 1]
    fig_age = px.histogram(death, x='age', color='sex', nbins=10, 
                           labels={'age': 'Age', 'sex': 'Sex'}, 
                           title='Age Distribution of Death Events by Sex')
    col1.subheader("Age Distribution of Death Events by Sex")
    col1.plotly_chart(fig_age, use_container_width=True)

    # Second column: Correlation Matrix
    fig_corr = px.imshow(df.corr(), title='Correlation Matrix')
    col2.subheader("Correlation Matrix")
    col2.plotly_chart(fig_corr, use_container_width=True)

    # Additional section for more analysis
    st.subheader("Additional Analysis")
    fig_bp = px.box(df, x='DEATH_EVENT', y='serum_creatinine', color='sex',
                    labels={'serum_creatinine': 'Serum Creatinine', 'DEATH_EVENT': 'Death Event'},
                    title='Serum Creatinine Levels by Death Event and Sex')
    st.plotly_chart(fig_bp, use_container_width=True)

# Main function to handle the navigation between steps
def main():
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        login_page()  # Show login page
    else:
        show_dashboard()  # Show dashboard after login

# Streamlit page configuration
st.set_page_config(page_title="Data Analysis App", page_icon=":bar_chart:")

if __name__ == "__main__":
    main()
