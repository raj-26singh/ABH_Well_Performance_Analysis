import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBRegressor
import base64
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoLarsCV
from sklearn.ensemble import RandomForestRegressor
import catboost as cb
from lightgbm import LGBMRegressor
from stqdm import stqdm
#from lightgbm import LGBMRegressor
#from pygam import GAM, s, te
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
    return f'<a href="data:file/csv;base64,{b64}" download="Predicted_Data.csv">Download Predictions</a>'

def sensitivity_analysis(x1,y1,result,name):
    st.subheader("Sensitivity Analysis")
    st.write("***Select the sensitive parameters (max 3):***")
    #st.write(x1.columns)
    count = 0
    sensitive_index = np.array([-1,-1,-1])
    for i in range(len(x1.columns)):
        if count>3:
            st.warning("You can select a maximum of 3 sensitive variables only")
        if st.checkbox(x1.columns[i],key=i):
            sensitive_index[count] = i
            count = count + 1
            if count==1:
                start = st.number_input("Start from :",value=0.0,format="%.5f",key=i+10)
                end = st.number_input("End at :",value=0.0,format="%.5f",key=i+11)
                steps = st.number_input("No. of steps :",value=0,key=i+12)
                a = np.linspace(start,end,num=steps)
            if count==2:
                start = st.number_input("Start from :",value=0.0,format="%.5f",key=i+10)
                end = st.number_input("End at :",value=0.0,format="%.5f",key=i+11)
                steps = st.number_input("No. of steps :",value=0,key=i+12)
                b = np.linspace(start,end,num=steps)
            if count==3:
                start = st.number_input("Start from :",value=0.0,format="%.5f",key=i+10)
                end = st.number_input("End at :",value=0.0,format="%.5f",key=i+11)
                steps = st.number_input("No. of steps :",value=0,key=i+12)
                c = np.linspace(start,end,num=steps)
    #st.write(a,b,c)
    st.write("***Enter values of other (non-sensitive) parameters :***")
    #length = len(x1.columns)-count
    val = np.zeros(len(x1.columns))
    for i in range(len(x1.columns)):
        flag = 0
        for j in sensitive_index:
            if i==j:
                flag = 1
        if flag==0:
            val[i] = st.number_input(x1.columns[i],value=0.0,format="%.5f",key=i+100)

    if st.button("RUN"):
        st.write("Running...")    
        if name=='XGBoost Regression':
            regr = XGBRegressor()
            regr.fit(x1,y1)
        if name=='Light GBM Regression':
            parameters1 = [{'n_estimators':[200,300],'learning_rate':[0.1,0.5,0.8],'random_state':[None,42,61,66]}]
            RR = LGBMRegressor()
            Grid1 = GridSearchCV(RR,parameters1,cv=10)
            Grid1.fit(x1,y1)
            params = Grid1.best_params_
            regr = LGBMRegressor(n_estimators=params['n_estimators'],learning_rate=params['learning_rate'],random_state=params['random_state'])
            regr.fit(x1,y1)
        if name=='RandomForest Regression':
            parameters1 = [{'n_estimators':[10,50,100,150,200,250,300],'max_features':['auto','sqrt','log2']}]
            RR = RandomForestRegressor()
            Grid1 = GridSearchCV(RR,parameters1,cv=4)
            Grid1.fit(x1,y1)
            params = Grid1.best_params_
            regr = RandomForestRegressor(n_estimators=params['n_estimators'],max_features=params['max_features'])
            regr.fit(x1,y1)
        if name=='Lasso Regression':
            regr = LassoLarsCV(cv=5)
            regr.fit(x1,y1)
        if name=='Ridge Regression':
            RR = Ridge()
            parameters1 = [{'alpha':[0.0001,0.001,0.01,0.1,0.5,1,10,100,1000,10000],'normalize':[True,False]}]
            Grid1 = GridSearchCV(RR,parameters1,cv=5)
            Grid1.fit(x1,y1)
            params = Grid1.best_params_
            #st.write(params['alpha'])
            regr = Ridge(alpha=params['alpha'],normalize=params['normalize'])
            #regr = XGBRegressor()
            regr.fit(x1,y1)
        if name=='Linear Regression':
            regr = LinearRegression()
            regr.fit(x1,y1)
        
        if count==1:
            x0 = np.zeros((len(a),len(x1.columns)))
            for i in range(len(a)):
                for j in range(len(x1.columns)):
                    if j==sensitive_index[0] or j==sensitive_index[1] or j==sensitive_index[2]:
                        x0[i,j] = a[i]
                    else:
                        x0[i,j] = val[j]
            y0 = regr.predict(x0)
            df = pd.DataFrame(x0,columns=x1.columns)
            df[result] = y0
            st.dataframe(df)
            st.markdown(get_table_download_link(df), unsafe_allow_html=True)
        
        if count==2:
            x0 = np.zeros(((len(a)*len(b)),len(x1.columns)))
            index_x = 0
            for i in range(len(a)):
                for k in range(len(b)):
                    for j in range(len(x1.columns)):
                        if j==sensitive_index[0]:
                            x0[index_x,j] = a[i]
                        elif j==sensitive_index[1]:
                            x0[index_x,j] = b[k]
                        else:
                            x0[index_x,j] = val[j]
                            #x0[i+len(a),j] = val[j]
                    index_x = index_x + 1
            y0 = regr.predict(x0)
            df = pd.DataFrame(x0,columns=x1.columns)
            df[result] = y0
            st.dataframe(df)
            #st.write(a,b)
            st.markdown(get_table_download_link(df), unsafe_allow_html=True)

        if count==3:
            x0 = np.zeros(((len(a)*len(b)*len(c)),len(x1.columns)))
            index_x = 0
            for i in range(len(a)):
                for k in range(len(b)):
                    for l in range(len(c)):
                        for j in range(len(x1.columns)):
                            #st.write(index_x)
                            if j==sensitive_index[0]:
                                x0[index_x,j] = a[i]
                            elif j==sensitive_index[1]:
                                x0[index_x,j] = b[k]
                            elif j==sensitive_index[2]:
                                x0[index_x,j] = c[l]
                            else:
                                x0[index_x,j] = val[j]
                        index_x = index_x + 1 
                                
            y0 = regr.predict(x0)
            df = pd.DataFrame(x0,columns=x1.columns)
            df[result] = y0
            st.dataframe(df)
            st.markdown(get_table_download_link(df), unsafe_allow_html=True)
            






    