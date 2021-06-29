import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoLarsCV
from sklearn.ensemble import RandomForestRegressor
from pygam import GAM, s, te
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import catboost as cb
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
import sensitivity_analysis
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from stqdm import stqdm

def ML(x1,x2,y1,y2,name,y):
    
    if name=='Linear Regression':
    
        st.write("Linear Regression Model for your data is :")
        lm = LinearRegression()
        lm.fit(x1,y1)
        y_hat = lm.predict(x2)
        
        coef = lm.coef_
        intercept = lm.intercept_
        #st.write(intercept," ",coef)
        
        if len(coef)==1:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0])
        if len(coef)==2:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1])
        if len(coef)==3:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2])
        if len(coef)==4:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2],"+ (",coef[3],")",x1.columns[3])
        if len(coef)==5:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2],"+ (",coef[3],")",x1.columns[3],"+ (",coef[4],")",x1.columns[4])
        if len(coef)==6:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2],"+ (",coef[3],")",x1.columns[3],"+ (",coef[4],")",x1.columns[4],"+ (",coef[5],")",x1.columns[5])
        if len(coef)==7:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2],"+ (",coef[3],")",x1.columns[3],"+ (",coef[4],")",x1.columns[4],"+ (",coef[5],")",x1.columns[5],"+ (",coef[6],")",x1.columns[6])
        if len(coef)==8:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2],"+ (",coef[3],")",x1.columns[3],"+ (",coef[4],")",x1.columns[4],"+ (",coef[5],")",x1.columns[5],"+ (",coef[6],")",x1.columns[6],"+ (",coef[7],")",x1.columns[7])
        
        st.write("Training set Results")
        st.write("Mean Squared Error :",mean_squared_error(lm.predict(x1),y1))
        st.write("R squared :",lm.score(x1,y1))

        st.write("Testing set Results")
        st.write("Mean Squared Error :",mean_squared_error(y_hat,y2))
        st.write("R squared :",lm.score(x2,y2))

        st.header("Predictions")
        st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
        val = np.zeros(len(x1.columns))
        for i in range(len(x1.columns)):
            val[i] = st.number_input(x1.columns[i],value=0.0,format="%.5f")
        ans = intercept
        if st.button("RUN"):
            for i in range(len(x1.columns)):
                ans = ans + val[i]*coef[i]
            st.write(y,"=",ans)

    elif name=='Polynomial Regression':

        lm = LinearRegression()
        deg = st.number_input("Please select the degree of polynomial to fit :",min_value=2,max_value=10,step=1,value=2)
        st.write("Polynomial Regression Model of degree ",deg,"for your data is :")
        pr = PolynomialFeatures(degree=deg)
        x1_pr = pr.fit_transform(x1)
        x2_pr = pr.fit_transform(x2)
        lm.fit(x1_pr,y1)
        y_hat = lm.predict(x2_pr)
        coef = lm.coef_
        intercept = lm.intercept_
        #st.write(intercept," ",coef)
        st.write("intercept :",intercept)
        st.write("Co-efficients :",coef[0:])
        
        st.write("Training set Results")
        st.write("Mean Squared Error :",mean_squared_error(lm.predict(x1_pr),y1))
        st.write("R squared :",lm.score(x1_pr,y1))

        st.write("Testing set Results")
        st.write("Mean Squared Error :",mean_squared_error(y_hat,y2))
        st.write("R squared :",lm.score(x2_pr,y2))

        st.header("Predictions")
        st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
        val = np.zeros((1,len(x1.columns)))
        for i in range(len(x1.columns)):
            val[0,i] = st.number_input(x1.columns[i],value=0.0,format="%.5f")
        val_pr = pr.fit_transform(val)
        if st.button("RUN"):
            ans = lm.predict(val_pr)
            st.write(y,"=",ans[0])


    elif name=='Ridge Regression':
    
        st.write("Ridge Regression Model for your data is :")
        parameters1 = [{'alpha':[0.0001,0.001,0.01,0.1,0.5,1,10,100,1000,10000],'normalize':[True,False]}]
        RR = Ridge()
        Grid1 = GridSearchCV(RR,parameters1,cv=5)
        Grid1.fit(x1,y1)
        params = Grid1.best_params_
        #st.write(params['alpha'])
        RidgeModel = Ridge(alpha=params['alpha'],normalize=params['normalize'])
        RidgeModel.fit(x1,y1)
        y_hat = RidgeModel.predict(x2)

        coef = RidgeModel.coef_
        intercept = RidgeModel.intercept_
        if len(coef)==1:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0])
        if len(coef)==2:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1])
        if len(coef)==3:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2])
        if len(coef)==4:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2],"+ (",coef[3],")",x1.columns[3])
        if len(coef)==5:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2],"+ (",coef[3],")",x1.columns[3],"+ (",coef[4],")",x1.columns[4])
        if len(coef)==6:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2],"+ (",coef[3],")",x1.columns[3],"+ (",coef[4],")",x1.columns[4],"+ (",coef[5],")",x1.columns[5])
        if len(coef)==7:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2],"+ (",coef[3],")",x1.columns[3],"+ (",coef[4],")",x1.columns[4],"+ (",coef[5],")",x1.columns[5],"+ (",coef[6],")",x1.columns[6])
        if len(coef)==8:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2],"+ (",coef[3],")",x1.columns[3],"+ (",coef[4],")",x1.columns[4],"+ (",coef[5],")",x1.columns[5],"+ (",coef[6],")",x1.columns[6],"+ (",coef[7],")",x1.columns[7])
        
        st.write("Training set Results")
        st.write("Mean Squared Error :",mean_squared_error(RidgeModel.predict(x1),y1))
        st.write("R squared :",RidgeModel.score(x1,y1))

        st.write("Testing set Results")
        st.write("Mean Squared Error :",mean_squared_error(y_hat,y2))
        st.write("R squared :",RidgeModel.score(x2,y2))


        st.header("Predictions")
        st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
        val = np.zeros((1,len(x1.columns)))
        for i in range(len(x1.columns)):
            val[0,i] = st.number_input(x1.columns[i],value=0.0,format="%.5f")
        if st.button("RUN"):
            ans = RidgeModel.predict(val)
            st.write(y,"=",ans[0])
    
    elif name=='Lasso Regression':
        
        st.write("Lasso Regression Model for your data is :")
        model = LassoLarsCV(cv=5).fit(x1,y1)
        y_hat = model.predict(x2)

        coef = model.coef_
        intercept = model.intercept_

        if len(coef)==1:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0])
        if len(coef)==2:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1])
        if len(coef)==3:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2])
        if len(coef)==4:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2],"+ (",coef[3],")",x1.columns[3])
        if len(coef)==5:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2],"+ (",coef[3],")",x1.columns[3],"+ (",coef[4],")",x1.columns[4])
        if len(coef)==6:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2],"+ (",coef[3],")",x1.columns[3],"+ (",coef[4],")",x1.columns[4],"+ (",coef[5],")",x1.columns[5])
        if len(coef)==7:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2],"+ (",coef[3],")",x1.columns[3],"+ (",coef[4],")",x1.columns[4],"+ (",coef[5],")",x1.columns[5],"+ (",coef[6],")",x1.columns[6])
        if len(coef)==8:
            st.write(y,"=",intercept,"+ (",coef[0],")",x1.columns[0],"+ (",coef[1],")",x1.columns[1],"+ (",coef[2],")",x1.columns[2],"+ (",coef[3],")",x1.columns[3],"+ (",coef[4],")",x1.columns[4],"+ (",coef[5],")",x1.columns[5],"+ (",coef[6],")",x1.columns[6],"+ (",coef[7],")",x1.columns[7])
        
        st.write("Training set Results")
        st.write("Mean Squared Error :",mean_squared_error(model.predict(x1),y1))
        st.write("R squared :",model.score(x1,y1))

        st.write("Testing set Results")
        st.write("Mean Squared Error :",mean_squared_error(y_hat,y2))
        st.write("R squared :",model.score(x2,y2))

        st.header("Predictions")
        st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
        val = np.zeros((1,len(x1.columns)))
        for i in range(len(x1.columns)):
            val[0,i] = st.number_input(x1.columns[i],value=0.0,format="%.5f")
        if st.button("RUN"):
            ans = model.predict(val)
            st.write(y,"=",ans[0])

    elif name=='RandomForest Regression':
        st.write("RandomForest Regression Model for your data")
        parameters1 = [{'n_estimators':[10,50,100,150,200,250,300],'max_features':['auto','sqrt','log2']}]
        RR = RandomForestRegressor()
        Grid1 = GridSearchCV(RR,parameters1,cv=4)
        Grid1.fit(x1,y1)
        params = Grid1.best_params_
        model0 = RandomForestRegressor(n_estimators=params['n_estimators'],max_features=params['max_features'])
        stqdm(model0.fit(x1,y1))
        y_hat = model0.predict(x2)
        
        st.write("Training set Results")
        st.write("Mean Squared Error :",mean_squared_error(model0.predict(x1),y1))
        st.write("R squared :",model0.score(x1,y1))

        st.write("Testing set Results")
        st.write("Mean Squared Error :",mean_squared_error(y_hat,y2))
        st.write("R squared :",model0.score(x2,y2))

        st.header("Predictions")
        st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
        val = np.zeros((1,len(x1.columns)))
        for i in range(len(x1.columns)):
            val[0,i] = st.number_input(x1.columns[i],value=0.0,format="%.5f")
        if st.button("RUN"):
            ans = model0.predict(val)
            st.write(y,"=",ans[0])

    elif name=='XGBoost Regression':
        st.write("XGBoost Regression Model for your data")
        model = XGBRegressor(enable_categorical=True)#max_depth=3,gamma=5,eta=0.9,reg_alpha=0.5,reg_lambda=0.5)
        model.fit(x1,y1)
        y_hat = model.predict(x2)

        st.write("Training set Results")
        st.write("Mean Squared Error :",mean_squared_error(model.predict(x1),y1))
        st.write("R squared :",model.score(x1,y1))

        st.write("Testing set Results")
        st.write("Mean Squared Error :",mean_squared_error(y_hat,y2))
        st.write("R squared :",model.score(x2,y2))

        st.header("Predictions")
        st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
        val = np.zeros((1,len(x1.columns)))
        for i in range(len(x1.columns)):
            val[0,i] = st.number_input(x1.columns[i],value=0.0,format="%.5f")
        if st.button("RUN"):
            ans = model.predict(val)
            st.write(y,"=",ans[0])

    elif name=='CatBoost Regression':
        st.write("CatBoost Regression Model for your data")
        train_dataset = cb.Pool(x1,y1)
        test_dataset = cb.Pool(x2,y2)
        model = cb.CatBoostRegressor(loss_function='RMSE')
        grid = {'iterations': [75, 150, 200],
        'learning_rate': [0.03, 0.07, 0.1],
        #'depth': [2, 4, 6, 8],
        'l2_leaf_reg': [0.2, 0.5, 1, 3]}
        model.grid_search(grid, train_dataset)
        y_hat = model.predict(x2)
        
        st.write("Training set Results")
        st.write("Mean Squared Error :",mean_squared_error(model.predict(x1),y1))
        st.write("R squared :",model.score(x1,y1))

        st.write("Testing set Results")
        st.write("Mean Squared Error :",mean_squared_error(y_hat,y2))
        st.write("R squared :",model.score(x2,y2))

        
        st.header("Predictions")
        st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
        val = np.zeros((1,len(x1.columns)))
        for i in range(len(x1.columns)):
            val[0,i] = st.number_input(x1.columns[i],value=0.0,format="%.5f")
        if st.button("RUN"):
            ans = model.predict(val)
            st.write(y,"=",ans[0])
    
    elif name=='Light GBM Regression':
        st.write("Light GBM Regression Model for your data")
        parameters1 = [{'n_estimators':[200,300],'learning_rate':[0.1,0.5,0.8],'random_state':[None,42,61,66]}]
        RR = LGBMRegressor()
        Grid1 = GridSearchCV(RR,parameters1,cv=10)
        Grid1.fit(x1,y1)
        params = Grid1.best_params_
        #model0 = RandomForestRegressor(n_estimators=params['n_estimators'],max_features=params['max_features'])
        model = LGBMRegressor(n_estimators=params['n_estimators'],learning_rate=params['learning_rate'],random_state=params['random_state'])
        model.fit(x1,y1)
        y_hat = model.predict(x2)
        
        st.write("Training set Results")
        st.write("Mean Squared Error :",mean_squared_error(model.predict(x1),y1))
        st.write("R squared :",model.score(x1,y1))

        st.write("Testing set Results")
        st.write("Mean Squared Error :",mean_squared_error(y_hat,y2))
        st.write("R squared :",model.score(x2,y2))


        st.header("Predictions")
        st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
        val = np.zeros((1,len(x1.columns)))
        for i in range(len(x1.columns)):
            val[0,i] = st.number_input(x1.columns[i],value=0.0,format="%.5f")
        if st.button("RUN"):
            ans = model.predict(val)
            st.write(y,"=",ans[0])

    elif name=='Support Vector Regression':
        st.write("Support Vector Regression for your data")
        
        regr = make_pipeline(StandardScaler(), SVR(C=1,epsilon=0.2))
        regr.fit(x1,y1)
        y_hat = regr.predict(x2)

        st.write("Training set Results")
        st.write("Mean Squared Error :",mean_squared_error(regr.predict(x1),y1))
        st.write("R squared :",regr.score(x1,y1))

        st.write("Testing set Results")
        st.write("Mean Squared Error :",mean_squared_error(y_hat,y2))
        st.write("R squared :",regr.score(x2,y2))


        st.header("Predictions")
        st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
        val = np.zeros((1,len(x1.columns)))
        for i in range(len(x1.columns)):
            val[0,i] = st.number_input(x1.columns[i],value=0.0,format="%.5f")
        if st.button("RUN"):
            ans = regr.predict(val)
            st.write(y,"=",ans[0])

    elif name=='GAM':
        st.write("GAM Regression for your data")
        
        regr = GAM(s(0, n_splines=200) + te(3,1) + s(2), distribution='poisson', link='log')
        regr.fit(x1,y1)
        y_hat = regr.predict(x2)

        st.write("Training set Results")
        st.write("Mean Squared Error :",mean_squared_error(regr.predict(x1),y1))
        #st.write("R squared :",model.score(x1,y1))

        st.write("Testing set Results")
        st.write("Mean Squared Error :",mean_squared_error(y_hat,y2))
        #st.write("R squared :",model.score(x2,y2))

        st.header("Predictions")
        st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
        val = np.zeros((1,len(x1.columns)))
        for i in range(len(x1.columns)):
            val[0,i] = st.number_input(x1.columns[i],value=0.0,format="%.5f")
        if st.button("RUN"):
            ans = regr.predict(val)
            st.write(y,"=",ans[0])