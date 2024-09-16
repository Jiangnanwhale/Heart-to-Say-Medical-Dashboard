import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Heart Failure Mortality Prediction", page_icon=":heartpulse:", layout="wide")

st.title(":broken_heart: Heart Failure Mortality Prediction")
st.markdown("<style>div.block-container{padding-top:2.5rem;}</,style>",unsafe_allow_html=True)

st.write("Here is a web medical dashboard that supports physicians to predict the risk of mortality due to heart failure.")

df = pd.read_csv("d:/KI/project management_SU/PROHI-dashboard-class-exercise/heart_failure_clinical_records_dataset.csv")

st.sidebar.title(":guide_dog: Navigation")

option = st.sidebar.radio("Select an option:", [ "Show Data","Show Histogram", "Show Correlation Matrix"])

if option == "Show Data":
    st.subheader("Data Description")
    st.write(df.head(15))
elif option == "Show Histogram":
    death = df[df["DEATH_EVENT"] == 1]
    st.subheader("Age Distribution of Death Events by Sex")
    fig = px.histogram(death, x="age", color="sex", nbins=10, 
                labels={"age": "Age", "sex": "Sex"})
    st.plotly_chart(fig)
elif option == "Show Correlation Matrix":
    st.subheader("Correlation Matrix")
    styled_corr = df.corr().style.background_gradient(cmap="coolwarm")
    st.dataframe(styled_corr)


