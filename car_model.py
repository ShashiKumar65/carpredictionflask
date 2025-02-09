from flask import Flask,render_template,request
import pickle
import pandas as pd
app = Flask(__name__)
data=pd.read_csv("car data.csv")

model=pickle.load(open("car_model.pkl",'rb'))
@app.route('/')
def index():
    Fuel_Type=sorted(data['Fuel_Type'].unique())
    Seller_Type=sorted(data['Seller_Type'].unique())
    Transmission=sorted(data['Transmission'].unique())
    return render_template('index.html',fuel=Fuel_Type,seller=Seller_Type,trans=Transmission)
@app.route('/predict',methods=["POST"])
def predict():
    if request.method=='POST':
        Year=float(request.form.get('year'))
        Present_Price=float(request.form.get('p_price'))
        Kms_Driven=float(request.form.get('kms'))
        Fuel_Type=request.form.get('f_type')
        Seller_Type=request.form.get('s_type')
        Transmission=request.form.get('trans')
        Owner=float(request.form.get('own'))
        input_data=pd.DataFrame([[Year,Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission,Owner]],
                                columns=['Year','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner'])
        input_data.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
        input_data.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
        input_data['Transmission'].replace(['Manual','Automatic'],[0,1],inplace=True)
        pred=model.predict(input_data)
        return render_template('index.html',prediction_text='Result : {}'.format(pred[0]))


    

if __name__=="__main__":
    app.run(debug=True) 