#applying ann

#score : 2.92090(evaluation using Root Mean Squared Error metric)

# Importing the libraries
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.metrics import mean_squared_error 
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importing the dataset
dataset=pd.read_csv('drive/My Drive/train_set.csv')
x=dataset.iloc[:,1:14].values
y=dataset.iloc[:,14].values
#--------------
#working with test set
new_data=pd.read_csv('drive/My Drive/test.csv')
new_x=new_data.iloc[:,1:14].values

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

#applying ANN


# Initialising the ANN
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = x.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

#model summary
NN_model.summary()

# Fitting the ANN to the Training set
NN=NN_model.fit(x,y, epochs=20, batch_size=32, validation_split = 0.1)

#Making the predictions and evaluating the model
def make_submission(prediction, sub_name):
  my_submission = pd.DataFrame({'ID':pd.read_csv('drive/My Drive/test.csv').ID,'total_amount':prediction})
  my_submission.to_csv('{}.csv'.format(sub_name),index=False)
  print('A submission file has been made')

predictions = NN_model.predict(new_x)
make_submission(predictions[:,0],'submission(NN).csv')

#data visualisation
error = NN.history['mean_absolute_error']
val_error =NN.history['val_mean_absolute_error']
loss = NN.history['loss']
val_loss = NN.history['val_loss']
epochs = range(len(error))
plt.plot(epochs, error, 'bo', label='Training error')
plt.plot(epochs, val_error, 'b', label='Validation error')
plt.title('Training and validation error')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')

