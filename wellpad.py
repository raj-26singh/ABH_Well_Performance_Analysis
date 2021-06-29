import streamlit as st
import pandas as pd
import numpy as np
import correlate
import warnings
warnings.filterwarnings("ignore")

def wellpad(df):
    if df is not None:
        '''well_runlife = st.file_uploader("Well Data", type=['csv','xlsx'])
        if well_runlife is not None:
            st.sidebar.success("Files Uploaded successfully!")
        try:
            wellrunlifedf = pd.read_csv(well_runlife)
        except Exception as e:
            print(e)
            wellrunlifedf = pd.read_excel(well_runlife,sheet_name="ABH Wells run lifes")
        
        else:
            st.sidebar.write("No File Uploaded") 
        '''
        st.dataframe(df)
        st.subheader("Please enter the Well Pad Number")
        #well_pad = st.number_input("Well Pad No. :",min_value=1,max_value=9,value=4,step=1)
        #for i in range(39):
        #    if df.iloc[38-i,4]==well_pad:
        #       well.append(df.iloc[38-i,4])
        pads = ['Well Pad No. 1','Well Pad No. 2','Well Pad No. 3','Well Pad No. 4','Well Pad No. 5','Well Pad No. 6','Well Pad No. 7','Well Pad No. 8','Well Pad No. 9']
        well_pad = st.selectbox("Well Pad No.",pads)
        if well_pad==pads[0]:
            wells = ['ABH-020','ABH-021','ABH-022','ABH-023','ABH-024','ABH-025']
            st.subheader("Details of wells in Well Pad No. 1 are :")
            for i in [12,13,14,15,16,17]:
                st.write('### Well :',df.iloc[i-1,0])
                st.write("** Zone :**",df.iloc[i-1,1])
                st.write("** Trajectory :**",df.iloc[i-1,3])
                st.write("** Lateral Length :**",df.iloc[i-1,5])
                st.write("** Total Proppant :**",df.iloc[i-1,6])
                st.write("** No. of fracs :**",df.iloc[i-1,7])
                st.write("** Average Proppant per effective frac :**",df.iloc[i-1,36])
                st.write("** Fracture Spacing :**",df.iloc[i-1,34])
                st.write("** Liquid PI :**",df.iloc[i-1,28])
                st.write("\n")
            st.header("Analysis of wells")
            st.subheader("Please select the well you want to analyse")
            selected_well = st.selectbox("Well No.",wells)
            for i in range(len(wells)):
                if wells[i]==selected_well:
                    st.write("** Please upload the data file for well :**",selected_well)
                    well_data = st.file_uploader(label = selected_well, type = ['csv', 'xlsx'])  
                    if well_data is not None:
                        st.sidebar.success("Files Uploaded successfully!")
                        try:
                            welldatadf = pd.read_csv(well_data)
                        except Exception as e:
                            print(e)
                            welldatadf = pd.read_excel(well_data)
                    else:
                        st.sidebar.write("No File Uploaded")
                    correlate.correlate(welldatadf)

        if well_pad==pads[1]:
            wells = ['ABH-041','ABH-044','ABH-045','ABH-046']
            st.subheader("Details of wells in Well Pad No. 1 are :")
            for i in [33,36,37,38]:
                st.write('### Well :',df.iloc[i-1,0])
                st.write("** Zone :**",df.iloc[i-1,1])
                st.write("** Trajectory :**",df.iloc[i-1,3])
                st.write("** Lateral Length :**",df.iloc[i-1,5])
                st.write("** Total Proppant :**",df.iloc[i-1,6])
                st.write("** No. of fracs :**",df.iloc[i-1,7])
                st.write("** Average Proppant per effective frac :**",df.iloc[i-1,36])
                st.write("** Fracture Spacing :**",df.iloc[i-1,34])
                st.write("** Liquid PI :**",df.iloc[i-1,28])
                st.write("\n")
            st.header("Analysis of wells")
            st.subheader("Please select the well you want to analyse")
            selected_well = st.selectbox("Well No.",wells)
            for i in range(len(wells)):
                if wells[i]==selected_well:
                    st.write("** Please upload the data file for well :**",selected_well)
                    well_data = st.file_uploader(label = selected_well, type = ['csv', 'xlsx'])  
                    if well_data is not None:
                        st.sidebar.success("Files Uploaded successfully!")
                        try:
                            welldatadf = pd.read_csv(well_data)
                        except Exception as e:
                            print(e)
                            welldatadf = pd.read_excel(well_data)
                    else:
                        st.sidebar.write("No File Uploaded")
                    correlate.correlate(welldatadf)

        if well_pad==pads[2]:
            wells = ['ABH-012','ABH-013','ABH-014','ABH-017']
            st.subheader("Details of wells in Well Pad No. 1 are :")
            for i in [4,5,6,9]:
                st.write('### Well :',df.iloc[i-1,0])
                st.write("** Zone :**",df.iloc[i-1,1])
                st.write("** Trajectory :**",df.iloc[i-1,3])
                st.write("** Lateral Length :**",df.iloc[i-1,5])
                st.write("** Total Proppant :**",df.iloc[i-1,6])
                st.write("** No. of fracs :**",df.iloc[i-1,7])
                st.write("** Average Proppant per effective frac :**",df.iloc[i-1,36])
                st.write("** Fracture Spacing :**",df.iloc[i-1,34])
                st.write("** Liquid PI :**",df.iloc[i-1,28])
                st.write("\n")
            st.header("Analysis of wells")
            st.subheader("Please select the well you want to analyse")
            selected_well = st.selectbox("Well No.",wells)
            for i in range(len(wells)):
                if wells[i]==selected_well:
                    st.write("** Please upload the data file for well :**",selected_well)
                    well_data = st.file_uploader(label = selected_well, type = ['csv', 'xlsx'])  
                    if well_data is not None:
                        st.sidebar.success("Files Uploaded successfully!")
                        try:
                            welldatadf = pd.read_csv(well_data)
                        except Exception as e:
                            print(e)
                            welldatadf = pd.read_excel(well_data)
                    else:
                        st.sidebar.write("No File Uploaded")
                    correlate.correlate(welldatadf) 

        if well_pad==pads[3]:
            wells = ['ABH-032','ABH-033','ABH-034','ABH-035','ABH-037','ABH-038']
            st.subheader("Details of wells in Well Pad No. 1 are :")
            for i in [24,25,26,27,29,30]:
                st.write('### Well :',df.iloc[i-1,0])
                st.write("** Zone :**",df.iloc[i-1,1])
                st.write("** Trajectory :**",df.iloc[i-1,3])
                st.write("** Lateral Length :**",df.iloc[i-1,5])
                st.write("** Total Proppant :**",df.iloc[i-1,6])
                st.write("** No. of fracs :**",df.iloc[i-1,7])
                st.write("** Average Proppant per effective frac :**",df.iloc[i-1,36])
                st.write("** Fracture Spacing :**",df.iloc[i-1,34])
                st.write("** Liquid PI :**",df.iloc[i-1,28])
                st.write("\n")
            st.header("Analysis of wells")
            st.subheader("Please select the well you want to analyse")
            selected_well = st.selectbox("Well No.",wells)
            for i in range(len(wells)):
                if wells[i]==selected_well:
                    st.write("** Please upload the data file for well :**",selected_well)
                    well_data = st.file_uploader(label = selected_well, type = ['csv', 'xlsx'])  
                    if well_data is not None:
                        st.sidebar.success("Files Uploaded successfully!")
                        try:
                            welldatadf = pd.read_csv(well_data)
                        except Exception as e:
                            print(e)
                            welldatadf = pd.read_excel(well_data)
                    else:
                        st.sidebar.write("No File Uploaded")
                    correlate.correlate(welldatadf)

        if well_pad==pads[4]:
            wells = ['ABH-039','ABH-040','ABH-042','ABH-043']
            st.subheader("Details of wells in Well Pad No. 1 are :")
            for i in [31,32,34,35]:
                st.write('### Well :',df.iloc[i-1,0])
                st.write("** Zone :**",df.iloc[i-1,1])
                st.write("** Trajectory :**",df.iloc[i-1,3])
                st.write("** Lateral Length :**",df.iloc[i-1,5])
                st.write("** Total Proppant :**",df.iloc[i-1,6])
                st.write("** No. of fracs :**",df.iloc[i-1,7])
                st.write("** Average Proppant per effective frac :**",df.iloc[i-1,36])
                st.write("** Fracture Spacing :**",df.iloc[i-1,34])
                st.write("** Liquid PI :**",df.iloc[i-1,28])
                st.write("\n")
            st.header("Analysis of wells")
            st.subheader("Please select the well you want to analyse")
            selected_well = st.selectbox("Well No.",wells)
            for i in range(len(wells)):
                if wells[i]==selected_well:
                    st.write("** Please upload the data file for well :**",selected_well)
                    well_data = st.file_uploader(label = selected_well, type = ['csv', 'xlsx'])  
                    if well_data is not None:
                        st.sidebar.success("Files Uploaded successfully!")
                        try:
                            welldatadf = pd.read_csv(well_data)
                        except Exception as e:
                            print(e)
                            welldatadf = pd.read_excel(well_data)
                    else:
                        st.sidebar.write("No File Uploaded")
                    correlate.correlate(welldatadf)

        if well_pad==pads[5]:
            wells = ['ABH-026','ABH-027','ABH-028']
            st.subheader("Details of wells in Well Pad No. 1 are :")
            for i in [18,19,20]:
                st.write('### Well :',df.iloc[i-1,0])
                st.write("** Zone :**",df.iloc[i-1,1])
                st.write("** Trajectory :**",df.iloc[i-1,3])
                st.write("** Lateral Length :**",df.iloc[i-1,5])
                st.write("** Total Proppant :**",df.iloc[i-1,6])
                st.write("** No. of fracs :**",df.iloc[i-1,7])
                st.write("** Average Proppant per effective frac :**",df.iloc[i-1,36])
                st.write("** Fracture Spacing :**",df.iloc[i-1,34])
                st.write("** Liquid PI :**",df.iloc[i-1,28])
                st.write("\n")
            st.header("Analysis of wells")
            st.subheader("Please select the well you want to analyse")
            selected_well = st.selectbox("Well No.",wells)
            for i in range(len(wells)):
                if wells[i]==selected_well:
                    st.write("** Please upload the data file for well :**",selected_well)
                    well_data = st.file_uploader(label = selected_well, type = ['csv', 'xlsx'])  
                    if well_data is not None:
                        st.sidebar.success("Files Uploaded successfully!")
                        try:
                            welldatadf = pd.read_csv(well_data)
                        except Exception as e:
                            print(e)
                            welldatadf = pd.read_excel(well_data)
                    else:
                        st.sidebar.write("No File Uploaded")
                    correlate.correlate(welldatadf)

        if well_pad==pads[6]:
            wells = ['ABH-008','ABH-016','ABH-018','ABH-047']
            st.subheader("Details of wells in Well Pad No. 1 are :")
            for i in [2,8,10,39]:
                st.write('### Well :',df.iloc[i-1,0])
                st.write("** Zone :**",df.iloc[i-1,1])
                st.write("** Trajectory :**",df.iloc[i-1,3])
                st.write("** Lateral Length :**",df.iloc[i-1,5])
                st.write("** Total Proppant :**",df.iloc[i-1,6])
                st.write("** No. of fracs :**",df.iloc[i-1,7])
                st.write("** Average Proppant per effective frac :**",df.iloc[i-1,36])
                st.write("** Fracture Spacing :**",df.iloc[i-1,34])
                st.write("** Liquid PI :**",df.iloc[i-1,28])
                st.write("\n")
            st.header("Analysis of wells")
            st.subheader("Please select the well you want to analyse")
            selected_well = st.selectbox("Well No.",wells)
            for i in range(len(wells)):
                if wells[i]==selected_well:
                    st.write("** Please upload the data file for well :**",selected_well)
                    well_data = st.file_uploader(label = selected_well, type = ['csv', 'xlsx'])  
                    if well_data is not None:
                        st.sidebar.success("Files Uploaded successfully!")
                        try:
                            welldatadf = pd.read_csv(well_data)
                        except Exception as e:
                            print(e)
                            welldatadf = pd.read_excel(well_data)
                    else:
                        st.sidebar.write("No File Uploaded")
                    correlate.correlate(welldatadf)

        if well_pad==pads[7]:
            wells = ['ABH-029','ABH-030','ABH-031','ABH-036']
            st.subheader("Details of wells in Well Pad No. 1 are :")
            for i in [21,22,23,28]:
                st.write('### Well :',df.iloc[i-1,0])
                st.write("** Zone :**",df.iloc[i-1,1])
                st.write("** Trajectory :**",df.iloc[i-1,3])
                st.write("** Lateral Length :**",df.iloc[i-1,5])
                st.write("** Total Proppant :**",df.iloc[i-1,6])
                st.write("** No. of fracs :**",df.iloc[i-1,7])
                st.write("** Average Proppant per effective frac :**",df.iloc[i-1,36])
                st.write("** Fracture Spacing :**",df.iloc[i-1,34])
                st.write("** Liquid PI :**",df.iloc[i-1,28])
                st.write("\n")
            st.header("Analysis of wells")
            st.subheader("Please select the well you want to analyse")
            selected_well = st.selectbox("Well No.",wells)
            for i in range(len(wells)):
                if wells[i]==selected_well:
                    st.write("** Please upload the data file for well :**",selected_well)
                    well_data = st.file_uploader(label = selected_well, type = ['csv', 'xlsx'])  
                    if well_data is not None:
                        st.sidebar.success("Files Uploaded successfully!")
                        try:
                            welldatadf = pd.read_csv(well_data)
                        except Exception as e:
                            print(e)
                            welldatadf = pd.read_excel(well_data)
                    else:
                        st.sidebar.write("No File Uploaded")
                    correlate.correlate(welldatadf)

        if well_pad==pads[8]:
            wells = ['ABH-007','ABH-011','ABH-015','ABH-019']
            st.subheader("Details of wells in Well Pad No. 1 are :")
            for i in [1,3,7,11]:
                st.write('### Well :',df.iloc[i-1,0])
                st.write("** Zone :**",df.iloc[i-1,1])
                st.write("** Trajectory :**",df.iloc[i-1,3])
                st.write("** Lateral Length :**",df.iloc[i-1,5])
                st.write("** Total Proppant :**",df.iloc[i-1,6])
                st.write("** No. of fracs :**",df.iloc[i-1,7])
                st.write("** Average Proppant per effective frac :**",df.iloc[i-1,36])
                st.write("** Fracture Spacing :**",df.iloc[i-1,34])
                st.write("** Liquid PI :**",df.iloc[i-1,28])
                st.write("\n")
            st.header("Analysis of wells")
            st.subheader("Please select the well you want to analyse")
            selected_well = st.selectbox("Well No.",wells)
            for i in range(len(wells)):
                if wells[i]==selected_well:
                    st.write("** Please upload the data file for well :**",selected_well)
                    well_data = st.file_uploader(label = selected_well, type = ['csv', 'xlsx'])  
                    if well_data is not None:
                        st.sidebar.success("Files Uploaded successfully!")
                        try:
                            welldatadf = pd.read_csv(well_data)
                        except Exception as e:
                            print(e)
                            welldatadf = pd.read_excel(well_data)
                    else:
                        st.sidebar.write("No File Uploaded")
                    correlate.correlate(welldatadf)
                


    else:
        st.error('File Not uploaded yet. Please upload the field data first in order to analyse')