#Logistic regression

#score : 1.00000 (evaluated using  sklearn.metrics.accuracy_score())

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing datasets
dataset=pd.read_csv('train.csv')
X=dataset.iloc[:,[1,2,3,6,7,8,9]].values
y=dataset.iloc[:,10].values

#------------------------------
#processing test set
new_data=pd.read_csv('test.csv')
new_x=new_data.iloc[:,[1,2,3,6,7,8,9]].values

#imputing missing values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(new_x[:, 1:])
new_x[:,1:] = imputer.transform(new_x[:, 1:])

#categorisation of test set entries 
new_x[:,1] = np.zeros((1, 4469))
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_new_x = LabelEncoder()
new_x[:, 0] = labelencoder_new_x.fit_transform(new_x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
new_x = onehotencoder.fit_transform(new_x).toarray()



#-------------------------------
#imputing missing values of train set
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, [1,2]])
X[:, [1,2]] = imputer.transform(X[:, [1,2]])

X[:,1] = np.zeros((1, 40210))

#categorisation of train set
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()


#splitting dataset in training set and test set
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
new_x=sc_x.transform(new_x)


#fitting logistic regression to training data
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

#predicting test set results
y_pred=classifier.predict(x_test)
new_y_pred=classifier.predict(new_x)


#predicting and creating submission file
new_y=new_data.iloc[:,0].values
new_y=new_y.reshape(4469,1)
new_y_pred=new_y_pred.reshape(4469,1)
final_res=np.concatenate((new_y, new_y_pred),axis=1)

res=pd.DataFrame(final_res)

res.to_csv('submission-new-LR.csv', header=True, index=False) 

#data visualisation
#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

