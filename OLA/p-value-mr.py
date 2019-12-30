#svr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('train_set.csv')
x=dataset.iloc[:,1:14].values
y=dataset.iloc[:,14].values
#--------------

new_data=pd.read_csv('test.csv')
new_x=new_data.iloc[:,1:14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
new_x[:, 10] = labelencoder_X.fit_transform(new_x[:, 10])
onehotencoder = OneHotEncoder(categorical_features = [10])
new_x = onehotencoder.fit_transform(new_x).toarray()

new_x=new_x[:,1:]
#--------------

y=y.reshape(-1, 1)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, [1,2,3,4,5,6,7,11,12]])
x[:,[1,2,3,4,5,6,7,11,12]] = imputer.transform(x[:, [1,2,3,4,5,6,7,11,12]])
imputer1=Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer1 = imputer1.fit(x[:, [0,8,9]])
x[:,[0,8,9]] = imputer1.transform(x[:, [0,8,9]])
imputer_y= Imputer(missing_values = 'NaN', strategy = 'mean',axis=0)
imputer_y= imputer_y.fit(y[:,[0]])
y[:,[0]] = imputer_y.transform(y[:,[0]])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, 10] = labelencoder_X.fit_transform(x[:, 10])
onehotencoder = OneHotEncoder(categorical_features = [10])
x = onehotencoder.fit_transform(x).toarray()

x=x[:,1:]

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
y=y.reshape(len(y),1)
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)
sc_new_x=StandardScaler()
new_x=sc_new_x.fit_transform(new_x)

#fitting the regression model to the dataset
#create your regressor here
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x_train,y_train)

#predicting results 
y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))
new_y_pred=sc_y.inverse_transform(regressor.predict(sc_new_x.transform(np.array([[6.5]]))))


new_y=new_data.iloc[:,0].values
new_y=new_y.reshape(154235,1)
new_y_pred=new_y_pred.reshape(154235,1)
final_res=np.concatenate((new_y, new_y_pred),axis=1)

res=pd.DataFrame(final_res)

res.to_csv('submissionolx-new_svr.csv', header=True, index=False)
