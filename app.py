
from flask import Flask,render_template,request
import pickle
#import numpy as np
import pandas as pd


model=pickle.load(open('model.pkl','rb'))

le_file=pickle.load(open('label_encoder.pkl','rb'))

  
#le_file1=pickle.load(open('label_encoder2.pkl','rb'))

app=Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/result',methods=['POST'])
def result():
    

    df={}
    df["country"]=request.form['country']
    df["Year"]=request.form['Year']
    df["Value_pesticide"]=request.form['Value_pesticide']
    df["avg_temp"]=request.form['avg_temp']
    df["avg_rain"]=request.form['avg_rain']
 
    df["crop"]=request.form['Crop']
  
    df1=pd.DataFrame(df,index = [0, 1, 2, 3, 4,5])
    

    #df1['crop']=le_file1.transform(df1['crop'])
    
    
    df1['crop']=df1['crop'].map({"Maize":1, "Potatoes":2, "Rice, paddy":3,"Sorghum":4, "Wheat":5,
                               "Cassava":6,"Soybeans":7, "Sweet potatoes":8, "Plantains and others":9,"Yams":10})


    df1['country']=le_file.transform(df1['country'])


    #scaler=pickle.load(open('scaler.pkl','rb'))
    
    #df1[df1.columns]=scaler.fit_transform(df1)
    
    pred=model.predict(df1)

    return render_template('result.html',data=pred[0])


if __name__=='__main__':
    app.run(port=8000)
    
    
    
    
    
    
    
  