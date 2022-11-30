import pickle
import pandas as pd 
#import numpy as np
#import seaborn as sns 

data= pd.read_csv(r'C:\Users\vivek\OneDrive\Desktop\crop_yield\yield_production.csv',encoding="ISO-8859-1")
data.rename(columns = {'Value_crop':'Crop_Yield'}, inplace = True)
data.drop('sl.no',axis=1, inplace= True)
data.drop_duplicates(inplace= True)
count = data['country'].value_counts().rename_axis('country').reset_index(name='counts')

Unique_Country = data['country'].unique()
Col1 = 'avg_temp'

NullCount = []
for i in Unique_Country:
    s = data[data['country']==i][Col1].isnull().sum()
    NullCount.append(s)

df3 = pd.DataFrame({'country': Unique_Country,
              'Number of NaN Values in Temp': NullCount})
df3 = df3[df3['Number of NaN Values in Temp']!=0]

df4=df3.merge(count)
data1=df4[df4['Number of NaN Values in Temp']==df4['counts']]

Unique_Country = data['country'].unique()
Col1 = 'avg_rain'

NullCount = []
for i in Unique_Country:
    s = data[data['country']==i][Col1].isnull().sum()
    NullCount.append(s)

df2 = pd.DataFrame({'country': Unique_Country,
              'Number of NaN Values in Rain': NullCount})
df2 = df2[df2['Number of NaN Values in Rain']!=0]
df5= df4.merge(df2)
data2=df5[df5['Number of NaN Values in Rain']==df5['counts']]
data['avg_temp'] = data['avg_temp'].fillna(data.groupby('country')['avg_temp'].transform('median'))
data['avg_rain'] = data['avg_rain'].fillna(data.groupby('country')['avg_rain'].transform('median'))
data1=data.copy()

new_df= pd.read_excel(r'C:\Users\vivek\OneDrive\Desktop\crop_yield\avg_temp_rain1.xlsx')
new_data=pd.merge(data, new_df, on=['country','Year'], how='left')
new_data['avg_temp_x'] = new_data['avg_temp_x'].fillna(new_data.pop('avg_temp_y'))
new_data['avg_rain_x'] = new_data['avg_rain_x'].fillna(new_data.pop('avg_rain_y'))
new_data.rename(columns = {'avg_temp_x':'avg_temp','avg_rain_x':'avg_rain'}, inplace = True)
df=new_data.copy()

#handling outliers for 'Item_code'
iqr=new_data['Item_Code'].quantile(.75)-new_data['Item_Code'].quantile(.25)
up=new_data['Item_Code'].quantile(.75) + 1.5*iqr
low=new_data['Item_Code'].quantile(.25) - 1.5*iqr
outliers=new_data[(new_data['Item_Code']<low)|(new_data['Item_Code']>up)]
#using statistics will cap Item_Code>up to upper limit and Item_Code<low to lower limit
new_data.loc[new_data['Item_Code']<low,'Item_Code']=low
new_data.loc[new_data['Item_Code']>up,'Item_Code']=up

from scipy.stats.mstats import winsorize
new_data['avg_temp'] = winsorize(new_data["avg_temp"], limits = 0.01)
new_data['Crop_Yield'] = winsorize(new_data["Crop_Yield"], limits = 0.01)
new_data['Value_pesticide'] = winsorize(new_data["Value_pesticide"], limits = 0.01)

new_data.drop(['Area_Code','Year Code', 'Element', 'Domain','Item', 'Unit_pesticide', 'crop Code', 
               'Unit_crop','Domain_1','Element_1','Element Code','Item_Code'],axis=1, inplace= True)
new_data.rename(columns = {'Crop_Yield':'Crop_Yield(hg/ha)',
                           'Item.1':'crop'}, inplace = True)

Extremely_low = ['Soybeans','Sorghum']
low=['Wheat','Maize','Rice, paddy']
medium = ['Yams','Plantains and others']
high = ['Cassava','Sweet potatoes']
Extremely_high=['Potatoes']

# initiallising a new column
new_data['crop_range'] = new_data['crop']
new_data.loc[new_data['crop'].isin(Extremely_low),'crop_range'] = 'Extremely_low'
new_data.loc[new_data['crop'].isin(low),'crop_range'] = 'low'
new_data.loc[new_data['crop'].isin(medium),'crop_range'] = 'medium'
new_data.loc[new_data['crop'].isin(high),'crop_range'] = 'high'
new_data.loc[new_data['crop'].isin(Extremely_high),'crop_range'] = 'Extremely_high'

#label encoding

#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#for i in ['country','crop_range','crop']:
   # new_data[i]=le.fit_transform(new_data[i])

from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
le2 = LabelEncoder()
#le3 = LabelEncoder()
new_data['country']= le1.fit_transform(new_data['country'])
new_data['crop_range']= le2.fit_transform(new_data['crop_range'])
#new_data['crop']= le3.fit_transform(new_data['crop'])

new_data['crop']=new_data['crop'].map({"Maize":1, "Potatoes":2, "Rice, paddy":3,"Sorghum":4, "Wheat":5,"Cassava":6,"Soybeans":7, "Sweet potatoes":8, "Plantains and others":9,"Yams":10})


# one hot encoding
#from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder()
#transformed = ohe.fit_transform(new_data['crop']).reshape(-1,1).toarray()

#pickling
pickle.dump(le1,open('label_encoder.pkl','wb'))


#pickle.dump(le3,open('label_encoder2.pkl','wb'))
#pickle.dump(transformed,open('transformed.pkl','wb'))
            
y=new_data['Crop_Yield(hg/ha)']
x=new_data.drop(['Crop_Yield(hg/ha)','crop_range'],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

#scaling
from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
x_train =scaler.fit_transform(x_train)
x_test =scaler.fit_transform(x_test)

#pickle.dump(scaler,open('scaler.pkl','wb'))


#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import r2_score

from sklearn.ensemble import ExtraTreesRegressor
etr=ExtraTreesRegressor(n_estimators=100, random_state=0).fit(x_train, y_train)
#y_pred=etr.predict(x_test)
#print('Root mean squared error is ',mean_squared_error(y_test,y_pred))
#print('R2 score ',knn_model.score(x_test,y_test))


pickle.dump(etr,open('model.pkl','wb'))


            