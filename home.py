import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


def home():
 
    st.info(""" Analyse your ABH wells here with simplicity """)

    st.write('### Introdction to the interface')
    
    st.write('''The webapp is an interface to analyse the Aishwariya Barmer Hill (ABH) wells. 
             The ABH field contains 37 horizontal(2200 mMD, average lateral length of 1000 m) and 2 deviated wells(1300 mMD).
             The key problem with an unoptimized well is the low oil production.''')
    st.write('To begin using the app, load your ABH well excel data file using the file upload option on the sidebar. Once you have done this, you can navigate to the relevant tools using the Navigation menu.')
    st.write('\n')
    st.header('Sections')
    st.subheader('Individual Well Analysis')
    st.write('**Home:** A brief description of the web app.')
    st.write('**Visualizer:** For data visualization')
    st.write('**Correlate:** For data correlation')
    st.write('**Data Entry:** For creating a new data entry into the well ABH dataset')
    st.subheader('Field Analysis')
    st.write('**Home:** A brief description of the web app.')
    st.write('**Well Pad Analysis:** Analysis of each well, well-pad wise.')
    st.write('**Overall Field Analysis:** Overall Analysis of the field by selection of multiple wells')
    st.write('**GOR-Based Field Analysis:** GOR-Based Analysis of the field by High, Moderate, Low and Very Low GOR Wells')
    st.write('**Workover/RRU Analysis:** Analysis of Workovers/RRUs of the wells of the ABH field')
    st.write('**Well Wise Visualisation:** Visualise each well of the field')
    
    
    st.write('\n')
    st.header('Well Parameters')
    st.write("1. Porosity = 24 %")
    st.write("2. Permeability = 1 mD")
    st.write("3. Initial Reservoir Pressure = 1510 psi")
    st.write("4. Initial Reservoir Temperature = 64 deg")
    st.write('\n')
    st.header('Fluid Parameters')
    st.write("1. Oil API = 30 deg API")
    st.write("2. Viscosity = 3 cp")
    st.write("3. Bubble Point = 1450 psi")    
    st.write('\n')
    st.subheader('           - Created by: Raj Kumar Singh')