### Impors
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D
from sklearn.metrics import classification_report
# Extrac the Data from MNIST
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
# Hard Encode the data
# Example 5 ======> [0,0,0,0,0,1,0,0,0,0,0]
y_test_cat = to_categorical(y_test,10)
y_train_cat = to_categorical(y_train,10)

# Scale the data between 0 and 1
Scaled_x_test = x_test / x_test.max()
Scaled_x_train = x_train / x_train.max()
# Create the model
model = Sequential()
# add a Convolutional layer
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
# add a pool layer
model.add(MaxPool2D(pool_size=(2,2)))
# add a Convolutional layer
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
# add a pool layer
model.add(MaxPool2D(pool_size=(2,2)))
# 2d ----> 1d
model.add(Flatten())
# add Dense layers
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
# get a brife summary of layers and model
model.summary()
# train the modle
model.fit(Scaled_x_train,y_train_cat,epochs=10,verbose=1)
# Evaluate the model
model.evaluate(Scaled_x_test,y_test_cat)
# Set data to pridect
predictions = model.predict_classes(x_test)
# Get the Classification report
print(classification_report(y_test,predictions))