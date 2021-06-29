import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import MPFM_ML
import sensitivity_analysis

def MPFM_Analysis(df):
    if df is not None:
        st.header("Please Upload the ABH Well Dataset")
        well0 = st.file_uploader(label ="For MPFM Analysis", type = ['csv', 'xlsx'])
        if well0 is not None:
            st.sidebar.success("Files Uploaded successfully!")
            try:
                well = pd.read_csv(well0)
            except Exception as e:
                print(e)
                well = pd.read_excel(well0,sheet_name="values")
        else:
            st.sidebar.write("No File Uploaded")

        st.write(well.shape)
        #well = well.dropna()
        '''well = well[subset=[n for n in well if (n!='Bad' and n!=0 and n!='Type mismatch')]
        well = well[well['Oil rate']!='Bad' and well['Oil rate']!=0 and well['Oil rate']!='Type mismatch']
        well = well[well['Water rate']!='Bad' and well['Water rate']!=0 and well['Water rate']!='Type mismatch']
        well = well[well['Gas rate']!='Bad' and well['Gas rate']!=0 and well['Gas rate']!='Type mismatch']
        well = well[well['CHP']!='Bad' and well['CHP']!=0 and well['CHP']!='Type mismatch']
        well = well[well['THP']!='Bad' and well['THP']!=0 and well['THP']!='Type mismatch']
        well = well[well['SPM']!='Bad' and well['SPM']!=0 and well['SPM']!='Type mismatch']'''
        
        
        #well.drop(well.loc[well['DPV']==0].index, inplace=True)
        well.drop(well.loc[well['DPV']=='Bad'].index, inplace=True)
        well.drop(well.loc[well['DPV']=='Type mismatch'].index, inplace=True)
        convert = {'DPV':float}
        well = well.astype(convert)
        well.drop(well.loc[well['DPV']==0].index, inplace=True)
        #well = well[well.DPV!=0]
        #well = well[well.DPV!='Bad']
        well.drop(well.loc[well['Oil rate']=='Bad'].index, inplace=True)
        well.drop(well.loc[well['Oil rate']=='Type mismatch'].index, inplace=True)
        convert = {'DPV':float,'Oil rate':float}
        well = well.astype(convert)
        well.drop(well.loc[well['Oil rate']<=0].index, inplace=True)
        
        well.drop(well.loc[well['Water rate']=='Bad'].index, inplace=True)
        well.drop(well.loc[well['Water rate']=='Type mismatch'].index, inplace=True)
        convert = {'DPV':float,'Oil rate':float,'Water rate':float}
        well = well.astype(convert)
        well.drop(well.loc[well['Water rate']<=0].index, inplace=True)
        
        well.drop(well.loc[well['Gas rate']=='Bad'].index, inplace=True)
        well.drop(well.loc[well['Gas rate']=='Type mismatch'].index, inplace=True)
        convert = {'DPV':float,'Oil rate':float,'Water rate':float,'Gas rate':float}
        well = well.astype(convert)
        well.drop(well.loc[well['Gas rate']==0].index, inplace=True)
        
        well.drop(well.loc[well['CHP']=='Bad'].index, inplace=True)
        well.drop(well.loc[well['CHP']=='Type mismatch'].index, inplace=True)
        convert = {'DPV':float,'Oil rate':float,'Water rate':float,'Gas rate':float,'CHP':float}
        well = well.astype(convert)
        
        well.drop(well.loc[well['THP']=='Bad'].index, inplace=True)
        well.drop(well.loc[well['THP']=='Type mismatch'].index, inplace=True)
        convert = {'DPV':float,'Oil rate':float,'Water rate':float,'Gas rate':float,'CHP':float,'THP':float}
        well = well.astype(convert)
        
        well.drop(well.loc[well['SPM']=='Bad'].index, inplace=True)
        well.drop(well.loc[well['SPM']=='Type mismatch'].index, inplace=True)
        convert = {'DPV':float,'Oil rate':float,'Water rate':float,'Gas rate':float,'CHP':float,'THP':float,'SPM':float}
        well = well.astype(convert)

        well = well.dropna()
        well['CHP'] = well['CHP']*0.145
        well['THP'] = well['THP']*0.145
        st.write(well.shape)

        if st.button("Show Dataframe"):
            st.dataframe(well)
        
        st.write("## Correlate the parameters here ###")
        st.write("***Select the parameters to correlate :***")

        params = []
        if st.button("Select All")==True:
            params = np.array(['DPV','Oil rate','Water rate','Gas rate','CHP','THP','SPM'])
        else:
            if st.checkbox("DPV"):
                params.append("DPV")
            if st.checkbox("Oil Rate"):
                params.append("Oil rate")
            if st.checkbox("Water Rate"):
                params.append("Water rate")
            if st.checkbox("Gas Rate"):
                params.append("Gas rate")
            if st.checkbox("CHP (psi)"):
                params.append("CHP")
            if st.checkbox("THP (psi)"):
                params.append("THP")
            if st.checkbox("Strokes Per Minute (SPM)"):
                params.append("SPM")
            params = np.array(params)

        df = pd.DataFrame(well[params])

        if st.button('Correlate')==True: 
            st.write(df.corr())
        
        st.subheader("Please select the target variable (Y):")
        y = st.selectbox("Target Variable (Y)", params)
        for i in range(len(params)):
            if params[i] == y:
                indexy = i
        
        X = df.drop(columns=y)
        Y = df.iloc[:,indexy]

        if st.checkbox("Use Train-Test Split"):
            tst_sz = st.number_input("Please enter the test split of the dataset",min_value=0.00,max_value=1.00,step=0.01,value=0.25)
            x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=tst_sz)
        else:
            x_train,x_test,y_train,y_test = X,X,Y,Y

        if st.checkbox("Single Prediction"):
            st.subheader("Please select the regression model to fit :")
            model_name = st.selectbox("Regression Model Name",['Linear Regression','Polynomial Regression','Ridge Regression','Lasso Regression','Support Vector Regression','GAM','RandomForest Regression','XGBoost Regression','CatBoost Regression','Light GBM Regression'])
            MPFM_ML.ML(x_train,x_test,y_train,y_test,model_name,y)
        
        if st.checkbox("Sensitivity Analysis"):
            st.subheader("Please select the regression model for Sensitivity :")
            model_name = st.selectbox("Regression Model Name",['Linear Regression','Ridge Regression','Lasso Regression','RandomForest Regression','XGBoost Regression','Light GBM Regression'])
            
            sensitivity_analysis.sensitivity_analysis(X,Y,y,model_name)

    else:
        st.warning("Empty File uploaded")
