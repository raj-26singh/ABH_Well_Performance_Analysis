import streamlit as st
import pandas as pd
import numpy as np
import correlate
import warnings
warnings.filterwarnings("ignore")

def gor_analysis(df):
    
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

    if st.checkbox("High GOR Wells"):
        well = ['ABH-007','ABH-011','ABH-012','ABH-027','ABH-043']
        
    if st.checkbox("Moderate GOR Wells"):
        well = ['ABH-008','ABH-013','ABH-026','ABH-028','ABH-040','ABH-042','ABH-045','ABH-046','ABH-047','ABH-014','ABH-039']
            
    if st.checkbox("Low GOR Wells"):
        well = ['ABH-023','ABH-024','ABH-029','ABH-030','ABH-036','ABH-041','ABH-044','ABH-018']
            
    if st.checkbox("Very Low GOR Wells"):
        well = ['ABH-020','ABH-021','ABH-022','ABH-25','ABH-031','ABH-032','ABH-033','ABH-034','ABH-035','ABH-037','ABH-038']
    
    well = np.array(well)
        #st.write(len(all_wells_df))
    ind = []
    for i in range(all_wells_df.shape[0]):
         key = 0
         for j in range(len(well)):
            if all_wells_df.iloc[i,0]==well[j]:
                key = key + 1
         if key == 0:
            ind.append(i)
        
    all_wells_df = all_wells_df.drop(ind,axis=0)
               
    all_wells_df = all_wells_df.drop('Source_Name',axis=1)
    
    st.dataframe(all_wells_df)
    correlate.correlate(all_wells_df)