import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score

heart_data=pd.read_csv("D:\\Programs\\solving for india hackathon\\heart_disease_data.csv")
heart_data.head()

heart_data.tail()

heart_data.shape

heart_data.info()

#get statistical data

heart_data.describe()

print(heart_data['target'].value_counts()) #tells how many are prone to disease acc to dataset  1-->heart disease 0-->No heart disease


#separate target column with other columns

X=heart_data.drop(columns='target',axis=1) #when columns is used then axis=1 when row is used then axis=0
Y=heart_data['target']
print(X)

print(Y)

#Splitting the data into training and test data

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2) #test size specify the percentage of data to be loaded random is used to split the data in a particular manner
print(X_train.shape,X_test.shape)

#Logistic Regression Model

model=LogisticRegressionCV()

#training Machine learning model with training data

model.fit(X_train,Y_train)   #loading the training data and training the data


#accuracy on training data

X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Training data accuracy: ",training_data_accuracy)

X_test_prediction=model.predict(X_test)
test_data_accuracy_score=accuracy_score(X_test_prediction,Y_test)
print("Testing data accuracy: ",test_data_accuracy_score)

#building a predective system

input_data=(55,0,0,170,200,0,1,60,1,5.6,0,0,3)



#changing the input data into a numpy array

numpy_array=np.asarray(input_data)

#reshape the numpy array as we are predicting for only one instance

input_data_reshape=numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshape)


if prediction[0]==1:
    print("The person has heart disease")
    print("Accuracy is around: ",training_data_accuracy*100)

else:
    print("The person is free from heart disease")
    print("Accuracy is around: ",training_data_accuracy*100)


x1=X_train_prediction

y1=Y_train


x2 = X_test_prediction
y2 = Y_test

# plotting the line 1 points
plt.plot(x1, y1, label="X train prediction and Y train data")

x2 = X_test_prediction
y2 = Y_test

# plotting the line 2 points
plt.plot(x2, y2, label="X test prediction and Y test data")

# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')
# giving a title to my graph
plt.title('Comparsion between training data and testing data')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()
#MuditSinha,PoorviBhalerao,HarshitSrivastav,IshanVishwakarma
