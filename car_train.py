import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
import pickle

car_data=pd.read_csv(r"C:\Users\shashikumar\Downloads\car data.csv")
car_data.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
car_data.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
car_data.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
X=car_data.drop(columns=['Car_Name','Selling_Price'],axis=1)
Y=car_data['Selling_Price']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=2)
lin_reg=LinearRegression()
lin_reg.fit(X_train,Y_train)
train_data_pred=lin_reg.predict(X_train)
train_data_error=metrics.r2_score(train_data_pred,Y_train)
print(train_data_error)
with open(r"C:\Users\shashikumar\OneDrive\machinelearning models\carprediction_model_flask\car_model.pkl",'wb') as file:
    pickle.dump(lin_reg,file)