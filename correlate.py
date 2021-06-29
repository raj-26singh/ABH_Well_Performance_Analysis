import streamlit as st
import pandas as pd
import numpy as np
import machine_learning
import preprocess_data
import base64
import warnings
warnings.filterwarnings("ignore")

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="Processed_Data.csv">Download Processed File</a>'


def correlate(df):
    
    st.write("## Correlate the parameters here ###")
    st.write("***Select the parameters to correlate :***")
        
    if df is not None:
        
        droplist = ['S. No.','Remarks']#'Max Surface SL','Max Downhole SL']
        if st.checkbox("Pump Submergence (mTVD)")==False:
            droplist.append('Pump Submergence (mTVD)')
        if st.checkbox("Pump depth (mTVD)")==False:
            droplist.append("Pump depth (mTVD)")
        if st.checkbox("SPM")==False:
            droplist.append("SPM")
        if st.checkbox("Pump Bore Size (in)")==False:
            droplist.append("Pump Bore Size (in)")
        if st.checkbox("Oil rate (bpd)")==False:
            droplist.append("Oil rate (bpd)")
        if st.checkbox("GOR (scf/bbl)")==False:
            droplist.append("GOR (scf/bbl)")
        if st.checkbox("CHP (psi)")==False:
            droplist.append("CHP (psi)")
        if st.checkbox("THP (psi)")==False:
            droplist.append("THP (psi)")
        if st.checkbox("Dyna Eff SL (surface) in")==False:
            droplist.append("Dyna Eff SL (surface) in")
        if st.checkbox("Dyna Eff SL (pump) in")==False:
            droplist.append("Dyna Eff SL (Pump) in")
        if st.checkbox("Dyna load (pump) lb")==False:
            droplist.append("Dyna load (pump) lb")
        if st.checkbox("PIP dyna (psi)")==False:
            droplist.append("PIP dyna (psi)")
        if st.checkbox("PIP echoshot (psi)")==False:
            droplist.append("PIP echoshot (psi)")
        if st.checkbox("Theoritical Liquid rate with 80% eff (bpd)")==False:
            droplist.append("Theoritical Liquid rate with 80% eff (bpd)")
        if st.checkbox("Liq Rate Slippage (bpd)")==False:
            droplist.append("Liq Rate Slippage (bpd)")
        if st.checkbox("PI (dyna)")==False:
            droplist.append("PI (dyna)")
        if st.checkbox("PI (echoshot)")==False:
            droplist.append("PI (echoshot)")
        if st.checkbox("Max Surface SL")==False:
            droplist.append("Max Surface SL")
        if st.checkbox("Max Downhole SL")==False:
            droplist.append("Max Downhole SL")
            

        #df = df.reset_index(drop=True,inplace=True)
        df = preprocess_data.preprocess_data(df)

        st.markdown(get_table_download_link(df), unsafe_allow_html=True)
        #st.write(df.columns)
        df = df.drop(columns=droplist)
        #df = df.dropna()
        st.dataframe(df)
        if st.button("Correlate"):
            st.write(df.corr())
        count = len(df.columns)
        choice = df.columns
        st.subheader("Please select the target variable (Y):")
        y = st.selectbox("Target Variable (Y)", choice)
        for i in range(count):
            if choice[i] == y:
                indexy = i
        #x = df.drop(columns=y).columns
        #st.write(x," ",y)
        #df = df.dropna()
        st.subheader("Please select the regression model to fit :")
        model_name = st.selectbox("Regression Model Name (for Single Prediction only)",['Linear Regression','Polynomial Regression','Ridge Regression','Lasso Regression','Support Vector Regression','GAM','RandomForest Regression','XGBoost Regression','CatBoost Regression','Light GBM Regression','Double-Output XGBoost Regression','Triple-Output XGBoost Regression'])
        machine_learning.machine_learning(model_name,df,y,indexy)

    else:
        st.error('File Not uploaded yet. Please upload the well data first in order to correlate them')