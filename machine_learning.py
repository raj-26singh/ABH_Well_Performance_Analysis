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

def machine_learning(name,df,y,indexy):
    
    
    Y = df.iloc[:,indexy]
    #st.write(Y)
    d = [y,"Date"]
    x = df.drop(columns=d)
    #st.write(x)
    
    if st.checkbox("Use Train-Test Split"):
        tst_sz = st.number_input("Please enter the test split of the dataset",min_value=0.00,max_value=1.00,step=0.01,value=0.25)
        x_train,x_test,y_train,y_test = train_test_split(x,Y,test_size=tst_sz)
    else:
        x_train,x_test,y_train,y_test = x,x,Y,Y
    st.header("Analysis")   
    if st.checkbox("Single Prediction"):

        if name=='Linear Regression':
        
            st.write("Linear Regression Model for your data is :")
            lm = LinearRegression()
            lm.fit(x_train,y_train)
            y_hat = lm.predict(x_test)
            
            coef = lm.coef_
            intercept = lm.intercept_
            #st.write(intercept," ",coef)
            
            if len(coef)==1:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0])
            if len(coef)==2:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1])
            if len(coef)==3:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2])
            if len(coef)==4:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3])
            if len(coef)==5:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4])
            if len(coef)==6:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5])
            if len(coef)==7:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6])
            if len(coef)==8:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6],"+ (",coef[7],")",x.columns[7])
            if len(coef)==9:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6],"+ (",coef[7],")",x.columns[7],"+ (",coef[8],")",x.columns[8])
            if len(coef)==10:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6],"+ (",coef[7],")",x.columns[7],"+ (",coef[8],")",x.columns[8],"+ (",coef[9],")",x.columns[9])
            if len(coef)==11:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6],"+ (",coef[7],")",x.columns[7],"+ (",coef[8],")",x.columns[8],"+ (",coef[9],")",x.columns[9],"+ (",coef[10],")",x.columns[10])
            if len(coef)==12:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6],"+ (",coef[7],")",x.columns[7],"+ (",coef[8],")",x.columns[8],"+ (",coef[9],")",x.columns[9],"+ (",coef[10],")",x.columns[10],"+ (",coef[11],")",x.columns[11])
            
            st.write("Mean Squared Error :",mean_squared_error(y_hat,y_test))
            st.write("R squared :",lm.score(x,Y))

            st.header("Predictions")
            st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
            val = np.zeros(len(x.columns))
            for i in range(len(x.columns)):
                val[i] = st.number_input(x.columns[i],value=0.0,format="%.5f")
            ans = intercept
            for i in range(len(x.columns)):
                ans = ans + val[i]*coef[i]
            st.write(y,"=",ans)

        elif name=='Polynomial Regression':

            lm = LinearRegression()
            deg = st.number_input("Please select the degree of polynomial to fit :",min_value=2,max_value=10,step=1,value=2)
            st.write("Polynomial Regression Model of degree ",deg,"for your data is :")
            pr = PolynomialFeatures(degree=deg)
            x_train_pr = pr.fit_transform(x_train)
            x_test_pr = pr.fit_transform(x_test)
            lm.fit(x_train_pr,y_train)
            y_hat = lm.predict(x_test_pr)
            coef = lm.coef_
            intercept = lm.intercept_
            #st.write(intercept," ",coef)
            st.write("intercept :",intercept)
            st.write("Co-efficients :",coef[0:])
            
            st.write("Mean Squared Error :",mean_squared_error(y_hat,y_test))
            st.write("R squared :",lm.score(x_test_pr,y_test))
            st.header("Predictions")
            st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
            val = np.zeros((1,len(x.columns)))
            for i in range(len(x.columns)):
                val[0,i] = st.number_input(x.columns[i],value=0.0,format="%.5f")
            val_pr = pr.fit_transform(val)
            ans = lm.predict(val_pr)
            st.write(y,"=",ans[0])


        elif name=='Ridge Regression':
        
            st.write("Ridge Regression Model for your data is :")
            parameters1 = [{'alpha':[0.0001,0.001,0.01,0.1,0.5,1,10,100,1000,10000],'normalize':[True,False]}]
            RR = Ridge()
            Grid1 = GridSearchCV(RR,parameters1,cv=5)
            Grid1.fit(x,Y)
            params = Grid1.best_params_
            #st.write(params['alpha'])
            RidgeModel = Ridge(alpha=params['alpha'],normalize=params['normalize'])
            RidgeModel.fit(x_train,y_train)
            y_hat = RidgeModel.predict(x_test)

            coef = RidgeModel.coef_
            intercept = RidgeModel.intercept_
            if len(coef)==1:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0])
            if len(coef)==2:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1])
            if len(coef)==3:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2])
            if len(coef)==4:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3])
            if len(coef)==5:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4])
            if len(coef)==6:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5])
            if len(coef)==7:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6])
            if len(coef)==8:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6],"+ (",coef[7],")",x.columns[7])
            if len(coef)==9:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6],"+ (",coef[7],")",x.columns[7],"+ (",coef[8],")",x.columns[8])
            if len(coef)==10:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6],"+ (",coef[7],")",x.columns[7],"+ (",coef[8],")",x.columns[8],"+ (",coef[9],")",x.columns[9])
            if len(coef)==11:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6],"+ (",coef[7],")",x.columns[7],"+ (",coef[8],")",x.columns[8],"+ (",coef[9],")",x.columns[9],"+ (",coef[10],")",x.columns[10])
            if len(coef)==12:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6],"+ (",coef[7],")",x.columns[7],"+ (",coef[8],")",x.columns[8],"+ (",coef[9],")",x.columns[9],"+ (",coef[10],")",x.columns[10],"+ (",coef[11],")",x.columns[11])
            
            st.write("Mean Squared Error :",mean_squared_error(y_hat,y_test))
            st.write("R squared :",RidgeModel.score(x,Y))

            st.header("Predictions")
            st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
            val = np.zeros((1,len(x.columns)))
            for i in range(len(x.columns)):
                val[0,i] = st.number_input(x.columns[i],value=0.0,format="%.5f")
            ans = RidgeModel.predict(val)
            st.write(y,"=",ans[0])
        
        elif name=='Lasso Regression':
            
            st.write("Lasso Regression Model for your data is :")
            model = LassoLarsCV(cv=5).fit(x_train,y_train)
            y_hat = model.predict(x_test)

            coef = model.coef_
            intercept = model.intercept_

            if len(coef)==1:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0])
            if len(coef)==2:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1])
            if len(coef)==3:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2])
            if len(coef)==4:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3])
            if len(coef)==5:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4])
            if len(coef)==6:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5])
            if len(coef)==7:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6])
            if len(coef)==8:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6],"+ (",coef[7],")",x.columns[7])
            if len(coef)==9:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6],"+ (",coef[7],")",x.columns[7],"+ (",coef[8],")",x.columns[8])
            if len(coef)==10:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6],"+ (",coef[7],")",x.columns[7],"+ (",coef[8],")",x.columns[8],"+ (",coef[9],")",x.columns[9])
            if len(coef)==11:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6],"+ (",coef[7],")",x.columns[7],"+ (",coef[8],")",x.columns[8],"+ (",coef[9],")",x.columns[9],"+ (",coef[10],")",x.columns[10])
            if len(coef)==12:
                st.write(y,"=",intercept,"+ (",coef[0],")",x.columns[0],"+ (",coef[1],")",x.columns[1],"+ (",coef[2],")",x.columns[2],"+ (",coef[3],")",x.columns[3],"+ (",coef[4],")",x.columns[4],"+ (",coef[5],")",x.columns[5],"+ (",coef[6],")",x.columns[6],"+ (",coef[7],")",x.columns[7],"+ (",coef[8],")",x.columns[8],"+ (",coef[9],")",x.columns[9],"+ (",coef[10],")",x.columns[10],"+ (",coef[11],")",x.columns[11])
            
            st.write("Mean Squared Error :",mean_squared_error(y_hat,y_test))
            st.write("R squared :",model.score(x,Y))

            st.header("Predictions")
            st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
            val = np.zeros((1,len(x.columns)))
            for i in range(len(x.columns)):
                val[0,i] = st.number_input(x.columns[i],value=0.0,format="%.5f")
            ans = model.predict(val)
            st.write(y,"=",ans[0])

        elif name=='RandomForest Regression':
            st.write("RandomForest Regression Model for your data")
            parameters1 = [{'n_estimators':[10,50,100,150,200,250,300],'max_features':['auto','sqrt','log2']}]
            RR = RandomForestRegressor()
            Grid1 = GridSearchCV(RR,parameters1,cv=4)
            Grid1.fit(x,Y)
            params = Grid1.best_params_
            model0 = RandomForestRegressor(n_estimators=params['n_estimators'],max_features=params['max_features'])
            model0.fit(x_train,y_train)
            y_hat = model0.predict(x_test)

            st.write("Mean Squared Error :",mean_squared_error(y_hat,y_test))
            st.write("R squared :",model0.score(x,Y))

            st.header("Predictions")
            st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
            val = np.zeros((1,len(x.columns)))
            for i in range(len(x.columns)):
                val[0,i] = st.number_input(x.columns[i],value=0.0,format="%.5f")
            ans = model0.predict(val)
            st.write(y,"=",ans[0])

        elif name=='XGBoost Regression':
            st.write("XGBoost Regression Model for your data")
            model = XGBRegressor(enable_categorical=True,max_depth=3,gamma=5,eta=0.9,reg_alpha=0.5,reg_lambda=0.5)
            model.fit(x_train,y_train)
            y_hat = model.predict(x_test)

            st.write("Mean Squared Error :",mean_squared_error(y_hat,y_test))
            st.write("R squared :",model.score(x,Y))

            st.header("Predictions")
            st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
            val = np.zeros((1,len(x.columns)))
            for i in range(len(x.columns)):
                val[0,i] = st.number_input(x.columns[i],value=0.0,format="%.5f")
            ans = model.predict(val)
            st.write(y,"=",ans[0])

        elif name=='CatBoost Regression':
            st.write("CatBoost Regression Model for your data")
            train_dataset = cb.Pool(x_train,y_train)
            test_dataset = cb.Pool(x_test,y_test)
            model = cb.CatBoostRegressor(loss_function='RMSE')
            grid = {'iterations': [75, 150, 200],
            'learning_rate': [0.03, 0.07, 0.1],
            #'depth': [2, 4, 6, 8],
            'l2_leaf_reg': [0.2, 0.5, 1, 3]}
            model.grid_search(grid, train_dataset)
            y_hat = model.predict(x_test)
            st.write("Mean Squared Error :",mean_squared_error(y_hat,y_test))
            st.write("R squared :",model.score(x,Y))
            
            st.header("Predictions")
            st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
            val = np.zeros((1,len(x.columns)))
            for i in range(len(x.columns)):
                val[0,i] = st.number_input(x.columns[i],value=0.0,format="%.5f")
            ans = model.predict(val)
            st.write(y,"=",ans[0])
        
        elif name=='Light GBM Regression':
            st.write("Light GBM Regression Model for your data")
            parameters1 = [{'n_estimators':[200,300],'learning_rate':[0.1,0.5,0.8],'random_state':[None,42,61,66]}]
            RR = LGBMRegressor()
            Grid1 = GridSearchCV(RR,parameters1,cv=10)
            Grid1.fit(x,Y)
            params = Grid1.best_params_
            #model0 = RandomForestRegressor(n_estimators=params['n_estimators'],max_features=params['max_features'])
            model = LGBMRegressor(n_estimators=params['n_estimators'],learning_rate=params['learning_rate'],random_state=params['random_state'])
            model.fit(x_train,y_train)
            y_hat = model.predict(x_test)
            st.write("Mean Squared Error :",mean_squared_error(y_hat,y_test))
            st.write("R squared :",model.score(x,Y))

            st.header("Predictions")
            st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
            val = np.zeros((1,len(x.columns)))
            for i in range(len(x.columns)):
                val[0,i] = st.number_input(x.columns[i],value=0.0,format="%.5f")
            ans = model.predict(val)
            st.write(y,"=",ans[0])

        elif name=='Double-Output XGBoost Regression':
            st.write("Multi-Output XGBoost Regression. \nPlease do not use train test split here")
            st.subheader("Please select the second target variable (Y2):")
            y2 = st.selectbox("Target Variable (Y2)", df.columns)
            x_train = x_train.drop(columns=y2)
            x_test = x_test.drop(columns=y2)
            for i in range(len(df.columns)):
                if df.columns[i] == y2:
                    indexy2 = i
            y22 = df.iloc[:,indexy2]
            y_train2 = pd.DataFrame({'y1':y_train,'y2':y22})
            
            #xgb = XGBRegressor()
            # fitting
            multioutputregressor = MultiOutputRegressor(XGBRegressor())
            multioutputregressor.fit(x_train,y_train2)
            y_hat2 = multioutputregressor.predict(x_test)
            #st.write(y_hat2.shape)
            st.write("Mean Squared Error :",np.mean(np.mean((y_hat2-y_train2)**2)))
            #r2 = 1 - (np.mean((y_hat2-y_train2)**2)/np.mean(((np.mean(y_train)*y_train.shape[0])-y_train2)**2))
            #st.write("R squared :",np.mean([XGBRegressor.score(x_test,y_hat2[0]),XGBRegressor.score(x_test,y_hat2[1])]))

            st.header("Predictions")
            st.write("### To predict the suitable value of",y," and ",y2,",please enter suitable values of other parameters :")
            val = np.zeros((1,len(x_train.columns)))
            for i in range(len(x_train.columns)):
                val[0,i] = st.number_input(x_train.columns[i],value=0.0,format="%.5f")
            ans = multioutputregressor.predict(val)
            
            st.write(y,"=",ans[0,0]," and ",y2,"=",ans[0,1])

        elif name=='Triple-Output XGBoost Regression':
            st.write("Multi-Output XGBoost Regression. \nPlease do not use train test split here")
            st.subheader("Please select the second target variable (Y2):")
            y2 = st.selectbox("Target Variable (Y2)", df.columns)
            st.subheader("Please select the third target variable (Y3):")
            y3 = st.selectbox("Target Variable (Y3)", df.columns)
            x_train = x_train.drop(columns=[y2,y3])
            x_test = x_test.drop(columns=[y2,y3])
            for i in range(len(df.columns)):
                if df.columns[i] == y2:
                    indexy2 = i
                if df.columns[i] == y3:
                    indexy3 = i
                
            y22 = df.iloc[:,indexy2]
            y33 = df.iloc[:,indexy3]
            y_train3 = pd.DataFrame({'y1':y_train,'y2':y22,'y3':y33})
            
            #xgb = XGBRegressor()
            # fitting
            multioutputregressor = MultiOutputRegressor(XGBRegressor())
            multioutputregressor.fit(x_train,y_train3)
            y_hat2 = multioutputregressor.predict(x_test)
            #st.write(y_hat2.shape)
            st.write("Mean Squared Error :",np.mean(np.mean((y_hat2-y_train3)**2)))
            #r2 = 1 - (np.mean((y_hat2-y_train2)**2)/np.mean(((np.mean(y_train)*y_train.shape[0])-y_train2)**2))
            #st.write("R squared :",np.mean([XGBRegressor.score(x_test,y_hat2[0]),XGBRegressor.score(x_test,y_hat2[1])]))

            st.header("Predictions")
            st.write("### To predict the suitable value of",y,",",y2," and ",y3,",please enter suitable values of other parameters :")
            val = np.zeros((1,len(x_train.columns)))
            for i in range(len(x_train.columns)):
                val[0,i] = st.number_input(x_train.columns[i],value=0.0,format="%.5f")
            ans = multioutputregressor.predict(val)
            
            st.write(y,"=",ans[0,0],", ",y2,"=",ans[0,1]," and ",y3,"=",ans[0,2])
        
        
        elif name=='Support Vector Regression':
            st.write("Support Vector Regression for your data")
            
            regr = make_pipeline(StandardScaler(), SVR(C=1,epsilon=0.2))
            regr.fit(x_train,y_train)
            y_hat = regr.predict(x_test)

            st.write("Mean Squared Error :",mean_squared_error(y_hat,y_test))
            st.write("R squared :",regr.score(x,Y))

            st.header("Predictions")
            st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
            val = np.zeros((1,len(x.columns)))
            for i in range(len(x.columns)):
                val[0,i] = st.number_input(x.columns[i],value=0.0,format="%.5f")
            ans = regr.predict(val)
            st.write(y,"=",ans[0])

        elif name=='GAM':
            st.write("GAM Regression for your data")
            
            regr = GAM(s(0, n_splines=200) + te(3,1) + s(2), distribution='poisson', link='log')
            regr.fit(x_train,y_train)
            y_hat = regr.predict(x_test)

            st.write("Mean Squared Error :",mean_squared_error(y_hat,y_test))
            #st.write("R squared :",regr.score(x,Y))

            st.header("Predictions")
            st.write("### To predict the suitable value of",y,", please enter suitable values of other parameters :")
            val = np.zeros((1,len(x.columns)))
            for i in range(len(x.columns)):
                val[0,i] = st.number_input(x.columns[i],value=0.0,format="%.5f")
            ans = regr.predict(val)
            st.write(y,"=",ans[0])

        



    if st.checkbox("Sensitivity Analysis"):
        sensitivity_analysis.sensitivity_analysis(x,Y,y)
