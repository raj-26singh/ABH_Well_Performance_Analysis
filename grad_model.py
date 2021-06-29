import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

pr = PolynomialFeatures(degree=2)

df = pd.read_csv(r"C:\Users\LENOVO\OneDrive - Vedanta Limited\Desktop\Cairn\Internship\Coding\Data Files\Gas_Grad_Dataset.csv")

X = df[['CHP','Sp. Gr.']]
y = df['Pressure Gradient']

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.15,random_state=42)

x_train_pr = pr.fit_transform(x_train[['CHP','Sp. Gr.']])
x_test_pr = pr.fit_transform(x_test[['CHP','Sp. Gr.']])

lr = LinearRegression()

lr.fit(x_train_pr,y_train)
y_hat = lr.predict(x_test_pr)

'''print("Degree = 4")
print("MSE = ",mean_squared_error(y_test,y_hat))
print("Coeff are :", lr.coef_)
print("Intercept :",lr.intercept_)
print("\n")'''

#print(lr.coef_)

xgb = XGBRegressor()
xgb.fit(x_train,y_train)
y_hat = xgb.predict(x_test)

print("Train Data")
print("MSE :",mean_squared_error(y_train,xgb.predict(x_train)))
print("R squared :",xgb.score(x_train,xgb.predict(x_train)))

print("Test Data")
print("MSE :",mean_squared_error(y_test,y_hat))
print("R squared :",xgb.score(x_test,y_hat))

print(xgb.predict(np.array([[700,1.25]])))

