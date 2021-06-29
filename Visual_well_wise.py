import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import warnings
warnings.filterwarnings("ignore")

def Visual_well_wise(df):
    if df is not None:
        #well_no = ['ABH-007','ABH-008','ABH-012','ABH-013','ABH-014','ABH-016','ABH-017','ABH-018','ABH-020','ABH-021','ABH-022','ABH-023','ABH-024','ABH-025','ABH-026','ABH-027','ABH-028','ABH-029','ABH-030','ABH-031','ABH-032','ABH-033','ABH-034','ABH-035','ABH-036','ABH-037','ABH-038','ABH-039','ABH-040','ABH-041','ABH-042','ABH-043','ABH-044','ABH-045','ABH-046','ABH-047']
        well_no1 = df.Well
        well_no = []
        for i in range(len(well_no1)):
            if well_no1[i] is not math.nan:
                well_no.append(well_no1[i])
        well = st.selectbox("Please select the well no.",well_no)
        st.write("Please upload ",well," data file for analysis")
        well_datafile = st.file_uploader("Well Data", type=['csv','xlsx'])
        if well_datafile is not None:
            st.sidebar.success("Files Uploaded successfully!")
        try:
            welldatafiledf = pd.read_csv(well_datafile)
        except Exception as e:
            print(e)
            welldatafiledf = pd.read_excel(well_datafile)
        
        else:
            st.sidebar.write("No File Uploaded") 
        if well_datafile is not None:
            cols = st.beta_columns(2)
            cols[0].write('''### Well-Data dataframe''')
            cols[0].write(welldatafiledf)

            st.write('''### Data plots''')
            x_options = welldatafiledf.columns
            y_options = welldatafiledf.columns
            cols = st.beta_columns(2)
            x_axis = cols[0].selectbox('Choose Value for X axis', x_options)
            y_axis = cols[1].selectbox('Choose Value for primary Y axis', y_options)
            y2_axis = cols[1].selectbox('Choose Value for secondary Y axis',y_options)
            st.write(x_axis,'  vs  ', y_axis,'&', y2_axis)
            for i in range(len(x_options)):
                if x_options[i]==x_axis:
                    x_ind = i
            for i in range(len(y_options)):
                if x_options[i]==y_axis:
                    y_ind = i
            for i in range(len(y_options)):
                if x_options[i]==y2_axis:
                    y2_ind = i

            x_val = np.array(welldatafiledf.iloc[:,x_ind])
            y_val = np.array(welldatafiledf.iloc[:,y_ind])
            y2_val = np.array(welldatafiledf.iloc[:,y2_ind])

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=x_val, y=y_val,name="Primary Y-axis"),secondary_y=False,)
            fig.add_trace(go.Scatter(x=x_val, y=y2_val,name="Secondary Y-axis"),secondary_y=True,)
            fig.update_yaxes(title_text="<b>Primary</b> Y-axis ", secondary_y=False)
            fig.update_yaxes(title_text="<b>Secondary</b> Y-axis ", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Well details")
            for i in range(39):
                if well_no[i]==well:
                    ind = i
            st.write("Well Name :",well)
            st.write("Well Pad No. :",df.iloc[38-i,4])
            st.write("Zone :",df.iloc[38-i,1])
            st.write("Current CHP :",df.iloc[38-i,19])
            st.write("Current FTHP :",df.iloc[38-i,18])
            st.write("Total Proppants :",round(df.iloc[38-i,6],0))
            st.write("Total Fracs :",round(df.iloc[38-i,7],0))
            st.write("Fracture Spacing :",round(df.iloc[38-i,34],3))
            st.write("Average Proppant per effective frac :",round(df.iloc[38-i,36],3))
            st.write("Slugging :",df.iloc[38-i,26])
            st.write("Reservoir Pressure :",df.iloc[38-i,27])
            st.write("Liquid Productivity Index (PI) :",df.iloc[38-i,28])
            st.write("Total Online Days :",df.iloc[38-i,29])
            st.write("60 days average oil rate :",round(df.iloc[38-i,30],4))
            st.write("30 days average oil rate :",round(df.iloc[38-i,31],4))
            st.write("Topmost Perforation (mTVD):",df.iloc[38-i,32])
            st.write("Trajectory :",df.iloc[38-i,3])
            st.write("Lateral Length :",df.iloc[38-i,5])
            

    else:
        st.error('File Not uploaded yet. Please upload the field data first in order to analyse')