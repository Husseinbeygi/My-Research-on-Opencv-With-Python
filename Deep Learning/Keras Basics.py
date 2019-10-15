# Imports
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report

from keras.models import Sequential
from keras.layers import Dense
# Read Data as Array
data = np.genfromtxt('/home/hussein/OpenCV/Deep Learning/data.csv',delimiter=',')

# Extrac labels & features from data
labels = data[:,4]
features = data[:,0:4]
# Rename the Lables for better understanding
y = labels
X = features
# split data in 30% test and 70% train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Scale the data to be between 0 and 1
Scalerobject = MinMaxScaler()
Scalerobject.fit(X_train)
Scaled_X_train = Scalerobject.transform(X_train)
Scaled_X_test  = Scalerobject.transform(X_test)
# Create the model for Nerual Network 
# 3 layers with activation function
model = Sequential()
model.add(Dense(4,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
# Comple the model
# https://keras.io/losses/
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# Train the nerual network
model.fit(Scaled_X_train,y_train,epochs=50,verbose=2)
# Evaluate the Model
predictions = model.predict_classes(Scaled_X_test)
print(predictions)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
# Save the model
model.save('Model_1.h5')