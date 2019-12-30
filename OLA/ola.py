#multiple linaer regression


#score : 3.94277(evaluation using Root Mean Squared Error metric)


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing datasets
dataset=pd.read_csv('train_set.csv')

x=dataset.iloc[:,1:14].values
y=dataset.iloc[:,14].values
#--------------
#dealing with test data
new_data=pd.read_csv('test.csv')
new_x=new_data.iloc[:,1:14].values

#applying categorisation on test set
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
new_x[:, 10] = labelencoder_X.fit_transform(new_x[:, 10])
onehotencoder = OneHotEncoder(categorical_features = [10])
new_x = onehotencoder.fit_transform(new_x).toarray()

new_x=new_x[:,1:]
#--------------

y=y.reshape(-1, 1)
#imputing missing values 
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


#categorisation of train set
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, 10] = labelencoder_X.fit_transform(x[:, 10])
onehotencoder = OneHotEncoder(categorical_features = [10])
x = onehotencoder.fit_transform(x).toarray()

x=x[:,1:]

#splitting of train set into training set and validation set
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


#processing  model on training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#checking model from predicted and actual values of label in test set
y_pred=regressor.predict(x_test)

#predicted values of given test set
new_y_pred=regressor.predict(new_x)

#converting matrix of predicted values into dataframe and csv file
new_y=new_data.iloc[:,0].values
new_y=new_y.reshape(154235,1)
new_y_pred=new_y_pred.reshape(154235,1)
final_res=np.concatenate((new_y, new_y_pred),axis=1)

res=pd.DataFrame(final_res)

res.to_csv('submissionolx-new.csv', header=True, index=False)

