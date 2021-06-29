import pandas as pd
#from openpyxl import load_workbook
import base64
import numpy as np
import streamlit as st
#import math
#import plotly.express as px
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

import warnings
warnings.filterwarnings("ignore")

def data_entry(well_data):
    st.write('''## New Data Entry''')
    #st.write('Uses any of the 4 Algorithms to converge to optimal x and y coordinates')
    if well_data is not None:

        
        # Turbine Specifications.
        st.write('''## Please provide the following inputs''') 
        
        global sno, date, pump_depth, spm, q0, gor, chp, thp, sl_surface, sl_pump, fluid_load, pip
        
        sno_last = well_data.iloc[-1,0]
        sno = st.number_input("S. No.", value=sno_last+1, step=1)
        date = st.date_input("Date")
        pump_depth = st.number_input("Pump depth (mTVD)",min_value= 0.0, max_value=5000.0, value=100.0, step=0.01,format="%f")
        spm = st.number_input("SPM",min_value=0.0,max_value=4.0,value=2.5, step=0.01,format="%f")
        q0 = st.slider("Oil rate (bpd)",min_value=0.0,max_value=1000.0,value=100.0, step=0.5,format="%f")
        gor = st.slider("GOR (scf/bbl)",min_value=0.0,max_value=20000.0,value=100.0, step=0.5,format="%f")
        chp = st.number_input("CHP (psi)",min_value=0.0,max_value=5000.0,value=100.0, step=0.5,format="%f")
        thp = st.number_input("THP (psi)",min_value=0.0,max_value=5000.0,value=100.0, step=0.5,format="%f")
        sl_surface = st.number_input("Dyna Eff SL (surface) in",min_value=0.0,max_value=216.0,value=150.0, step=0.01,format="%f")
        sl_pump = st.number_input("Dyna Eff SL (Pump) in",min_value=0.0,max_value=216.0,value=150.0, step=0.01,format="%f")
        fluid_load = st.slider("Dyna load (pump) lb",min_value=0.0,max_value=50000.0,value=5000.0, step=0.1,format="%f")
        pip = thp + (0.379*3.28*pump_depth) + (4*fluid_load/(3.142*3.25*3.25))
        st.write('''Pump-Intake Pressure (psi) :''',pip)
        values = [sno, date, pump_depth, spm, q0, gor, chp, thp, sl_surface, sl_pump, fluid_load, pip]

        if st.button("Add"):
            well_data.loc[len(well_data.index)] = values
            st.write(" Entry added successfully!")
            st.dataframe(well_data)
            def csv_downloader(data):
                csvfile = data.to_csv(index=False)
                b64 = base64.b64encode(csvfile.encode()).decode()
                new_filename = "Output_{}_.csv".format(timestr)
                st.markdown("### Download File ###")
                href = f'<a href="data:file/csv;base,{b64}" download="{new_filename}">Click Here!</a>'
                st.markdown(href,unsafe_allow_html=True)
            csv_downloader(well_data)
            
            '''def xlsx_downloader(data):
                xlsxfile = data.to_excel(index=False)
                b64 = base64.b64encode(xlsxfile.encode()).decode()
                new_filename = "Output_{}_.xlsx".format(timestr)
                st.markdown("### Download File ###")
                href = f'<a href="data:file/xlsx;base,{b64}" download="{new_filename}">Click Here!</a>'
                st.markdown(href,unsafe_allow_html=True)
            xlsx_downloader(well_data)'''
            
                
    else:
        st.error('File Not uploaded yet. Please upload the well data in order to add values to the file')
    
    