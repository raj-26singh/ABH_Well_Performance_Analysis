import streamlit as st
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def PI_forecast(df):
    if df is not None:
        st.header("Please Upload the ABH Field Master file")
        all_wells = st.file_uploader(label ="For Field Analysis", type = ['csv', 'xlsx']) 

        if all_wells is not None:
            st.sidebar.success("Files Uploaded successfully!")
            try:
                all_wells_df = pd.read_csv(all_wells)
            except Exception as e:
                print(e)
                all_wells_df = pd.read_excel(all_wells,sheet_name="Sheet1")
        else:
            st.sidebar.write("No File Uploaded")

        
        well = []

        cols = st.beta_columns(4)

        if cols[0].checkbox("ABH-007"):
            well.append("ABH-007")
        if cols[1].checkbox("ABH-008"):
            well.append("ABH-008")
        if cols[2].checkbox("ABH-012"):
            well.append("ABH-012")
        if cols[3].checkbox("ABH-013"):
            well.append("ABH-013")
            
        if cols[0].checkbox("ABH-014"):
            well.append("ABH-014")
        if cols[1].checkbox("ABH-018"):
            well.append("ABH-018")
        if cols[2].checkbox("ABH-020"):
            well.append("ABH-020")
        if cols[3].checkbox("ABH-021"):
            well.append("ABH-021")
            
        if cols[0].checkbox("ABH-022"):
            well.append("ABH-022")
        if cols[1].checkbox("ABH-023"):
            well.append("ABH-023")
        if cols[2].checkbox("ABH-024"):
            well.append("ABH-024")
        if cols[3].checkbox("ABH-025"):
            well.append("ABH-025")

        if cols[0].checkbox("ABH-026"):
            well.append("ABH-026")
        if cols[1].checkbox("ABH-027"):
            well.append("ABH-027")
        if cols[2].checkbox("ABH-028"):
            well.append("ABH-028")
        if cols[3].checkbox("ABH-029"):
            well.append("ABH-029")
        
        if cols[0].checkbox("ABH-030"):
            well.append("ABH-030")
        if cols[1].checkbox("ABH-031"):
            well.append("ABH-031")
        if cols[2].checkbox("ABH-032"):
            well.append("ABH-032")
        if cols[3].checkbox("ABH-033"):
            well.append("ABH-033")

        if cols[0].checkbox("ABH-034"):
            well.append("ABH-034")
        if cols[1].checkbox("ABH-035"):
            well.append("ABH-035")
        if cols[2].checkbox("ABH-036"):
            well.append("ABH-036")
        if cols[3].checkbox("ABH-037"):
            well.append("ABH-037")

        if cols[0].checkbox("ABH-038"):
            well.append("ABH-038")
        if cols[1].checkbox("ABH-040"):
            well.append("ABH-040")
        if cols[2].checkbox("ABH-041"):
            well.append("ABH-041")
        if cols[3].checkbox("ABH-044"):
            well.append("ABH-044")
        
        if cols[0].checkbox("ABH-045"):
            well.append("ABH-045")
        if cols[1].checkbox("ABH-046"):
            well.append("ABH-046")
        if cols[2].checkbox("ABH-047"):
            well.append("ABH-047")

        well = np.array(well)
        ind = []
        for i in range(all_wells_df.shape[0]):
            key = 0
            for j in range(len(well)):
                if all_wells_df.iloc[i,0]==well[j]:
                    key = key + 1
            if key == 0:
                ind.append(i)
        
        all_wells_df = all_wells_df.drop(ind,axis=0)
        #all_wells_df = all_wells_df.reset_index(drop=True,inplace=True)       
        st.dataframe(all_wells_df)    
        all_wells_df = all_wells_df.drop('Source_Name',axis=1)


    else:
        st.error('File Not uploaded yet. Please upload the field data first in order to analyse workover/RRU')