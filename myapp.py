import pandas as pd
import numpy as np
import Data
import home
import correlate
#import data_entry
import streamlit as st
import workover_RRU
import Visual_well_wise
import wellpad
import Overall_Analysis
import gor_analysis
import PI_forecast
import MPFM_Analysis

import warnings
warnings.filterwarnings("ignore")

st.image('https://media.giphy.com/media/ryMsLQnimPWeCeLcTq/giphy.gif', use_column_width=True)

#st.video('https://thumbs.gfycat.com/ConcreteChiefCollardlizard-mobile.mp4', format='mp4', start_time=0)

st.title('Hydraulic Rod Pumps in ABH Wells')



#st.sidebar.write('upload power data') 
#powercurve_data = st.sidebar.file_uploader(label =" ", type = ['csv'])                 


# Visualizing Uploaded File
global welldatadf, fielddatadf    

# Sidebar Navigation
#st.sidebar.title('Navigation')

if st.sidebar.checkbox("Individual Well Analysis"):

    st.sidebar.write('## Upload Individual ABH Well Data') 
    well_data = st.sidebar.file_uploader(label ="For Individual Well Analysis", type = ['csv', 'xlsx'])  

    if well_data is not None:
        st.sidebar.success("Files Uploaded successfully!")
        try:
            welldatadf = pd.read_csv(well_data)
            #powercurvedf = pd.read_csv(powercurve_data)
        except Exception as e:
            print(e)
            welldatadf = pd.read_excel(well_data)
            #powercurvedf = pd.read_excel(powercurve_data)
        
    else:
        st.sidebar.write("No File Uploaded") 

    #st.sidebar.subheader('For Individual Wells')
    options = st.sidebar.radio('Select a page:',  ['Home', 'Visualizer', 'Correlate'])

    if options == 'Home':
        home.home()
    elif options == 'Visualizer':
        if well_data is not None:
            Data.data(welldatadf)
        else:
            Data.data(None)
    elif options == 'Correlate':
        if well_data is not None:
            correlate.correlate(welldatadf)
        else:
            correlate.correlate(None)
    '''else:
        if well_data is not None:
            data_entry.data_entry(welldatadf)
        else:
            data_entry.data_entry(None)'''

if st.sidebar.checkbox("Field Analysis"):

    st.sidebar.write('## Upload ABH Field Data') 
    field_data = st.sidebar.file_uploader(label ="For Field Analysis (upload GOR file)", type = ['csv', 'xlsx']) 

    if field_data is not None:
        st.sidebar.success("Files Uploaded successfully!")
        try:
            fielddatadf = pd.read_csv(field_data)
            #powercurvedf = pd.read_csv(powercurve_data)
        except Exception as e:
            print(e)
            fielddatadf = pd.read_excel(field_data,sheet_name="Data_PI")
            #powercurvedf = pd.read_excel(powercurve_data)
        
    else:
        st.sidebar.write("No File Uploaded")

    options2 = st.sidebar.radio('Select a page:',  ['Home','Well Pad Analysis','Overall MPFM Analysis','Overall Pump Analysis', 'GOR-Based Pump Analysis','Well Wise Visualisation'])  # 'Workover/RRU Analysis',

    if options2 == 'Home':
        home.home()
    elif options2 == 'Well Wise Visualisation':
        if fielddatadf is not None:
            Visual_well_wise.Visual_well_wise(fielddatadf)
        else:
            Visual_well_wise.Visual_well_wise(None)
    elif options2 == 'Well Pad Analysis':
        if fielddatadf is not None:
            wellpad.wellpad(fielddatadf)
        else:
            wellpad.wellpad(None)        
    elif options2 == 'PI Forecasting':
        if fielddatadf is not None:
            PI_forecast.PI_forecast(fielddatadf)
        else:
            PI_forecast.PI_forecast(None)
    elif options2 == 'Overall Pump Analysis':
        if fielddatadf is not None:
            Overall_Analysis.Overall_Analysis(fielddatadf)
        else:
            Overall_Analysis.Overall_Analysis(None)
    elif options2 == 'GOR-Based Pump Analysis':
        if fielddatadf is not None:
            gor_analysis.gor_analysis(fielddatadf)
        else:
            gor_analysis.gor_analysis(None)
            
    elif options2 == 'Overall MPFM Analysis':
        if fielddatadf is not None:
            MPFM_Analysis.MPFM_Analysis(fielddatadf)
        else:
            MPFM_Analysis.MPFM_Analysis(None)




