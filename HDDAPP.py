import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score

import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
from PIL import Image, ImageTk
import tkinter as tk1


# Create a window
window = tk.Tk()
window.title("Heart Disease Detector App")
window.geometry("1280x720")

# Load the background image
bg_image = ImageTk.PhotoImage(Image.open("D:\\Programs\\SFIhackathon\\subbgg.png"))
bg_label = tk.Label(window, image=bg_image)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Define the form input fields
age_label = tk.Label(window, text=" Age: ",borderwidth=4,relief="raised",bg='black',fg='white',font=8)
age_label.place(x=100+50, y=50)
age_entry = tk.Entry(window,font=8)
age_entry.place(x=200+100, y=50)

sex_label = tk.Label(window, text=" Sex: ",borderwidth=4,relief="raised",bg='black',fg='white',font=8)
sex_label.place(x=50+100, y=100)
sex_entry = tk.Entry(window,font=8)
sex_entry.place(x=200+100, y=100)

cp_label = tk.Label(window, text=" CP: ",borderwidth=4,relief="raised",bg='black',fg='white',font=8)
cp_label.place(x=50+100, y=150)
cp_entry = tk.Entry(window,font=8)
cp_entry.place(x=200+100, y=150)

trestbps_label = tk.Label(window, text=" Trestbps: ",borderwidth=4,relief="raised",bg='black',fg='white',font=8)
trestbps_label.place(x=50+100, y=200)
trestbps_entry = tk.Entry(window,font=8)
trestbps_entry.place(x=200+100, y=200)

chol_label = tk.Label(window, text=" Cholestrol: ",borderwidth=4,relief="raised",bg='black',fg='white',font=8)
chol_label.place(x=50+100, y=250)
chol_entry = tk.Entry(window,font=8)
chol_entry.place(x=200+100, y=250)

fbs_label = tk.Label(window, text=" FBS: ",borderwidth=4,relief="raised",bg='black',fg='white',font=8)
fbs_label.place(x=50+100, y=300)
fbs_entry = tk.Entry(window,font=8)
fbs_entry.place(x=200+100, y=300)

restecg_label = tk.Label(window, text=" Restecg: ",borderwidth=4,relief="raised",bg='black',fg='white',font=8)
restecg_label.place(x=50+100, y=350)
restecg_entry = tk.Entry(window,font=8)
restecg_entry.place(x=200+100, y=350)

thalach_label = tk.Label(window, text=" Thalach: ",borderwidth=4,relief="raised",bg='black',fg='white',font=8)
thalach_label.place(x=750, y=50)
thalach_entry = tk.Entry(window,font=8)
thalach_entry.place(x=900, y=50)

exang_label = tk.Label(window, text=" Exang: ",borderwidth=4,relief="raised",bg='black',fg='white',font=8)
exang_label.place(x=750, y=100)
exang_entry = tk.Entry(window,font=8)
exang_entry.place(x=900, y=100)

oldpeak_label = tk.Label(window, text=" Oldpeak: ",borderwidth=4,relief="raised",bg='black',fg='white',font=8)
oldpeak_label.place(x=750, y=150)
oldpeak_entry = tk.Entry(window,font=8)
oldpeak_entry.place(x=900, y=150)

slope_label = tk.Label(window, text=" Slope: ",borderwidth=4,relief="raised",bg='black',fg='white',font=8)
slope_label.place(x=750, y=200)
slope_entry = tk.Entry(window,font=8)
slope_entry.place(x=900, y=200)

ca_label = tk.Label(window, text=" CA: ",borderwidth=4,relief="raised",bg='black',fg='white',font=8)
ca_label.place(x=750, y=250)
ca_entry = tk.Entry(window,font=8)
ca_entry.place(x=900, y=250)

thal_label = tk.Label(window, text=" Thal: ",borderwidth=4,relief="raised",bg='black',fg='white',font=8)
thal_label.place(x=750, y=300)
thal_entry = tk.Entry(window,font=8)
thal_entry.place(x=900, y=300)

# Define a function to submit the form
def submit_form():
    age = age_entry.get()
    sex = sex_entry.get()
    cp = cp_entry.get()
    trestbps = trestbps_entry.get()
    chol = chol_entry.get()
    fbs = fbs_entry.get()
    restecg = restecg_entry.get()
    thalach = thalach_entry.get()
    exang = exang_entry.get()
    oldpeak = oldpeak_entry.get()
    slope = slope_entry.get()
    ca = ca_entry.get()
    thal = thal_entry.get()
    with open("form_data.txt", "w+") as f:
        f.write(f"{age}\n{sex}\n{cp}\n{trestbps}\n{chol}\n{fbs}\n{restecg}\n{thalach}\n{exang}\n{oldpeak}\n{slope}\n{ca}\n{thal}")
    messagebox.showinfo("Form Submitted", "Data saved to file.")



# Define the submit button
save_button = tk.Button(window, text="Save", command=submit_form,borderwidth=4,relief="raised",bg='Green',fg='white',font=10)



save_button.place(x=540, y=500)


submit_button = tk.Button(window, text="Submit", command=window.destroy,borderwidth=4,relief="raised",bg='Green',fg='white',font=15)



save_button.place(x=640, y=450)
submit_button.place(x=634,y=550)


# Run the window


window.mainloop()


heart_data=pd.read_csv("D:\\Programs\\SFIhackathon\\heart_disease_data.csv")
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

#FILLING THE INPUT

file=open("form_data.txt","r")
data=[]

for line in file:
        data.append(line.strip())
#building a predective system
input_data=(float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5]),float(data[6]),float(data[7]),float(data[8]),float(data[9]),float(data[10]),float(data[11]),float(data[12]))



#changing the input data into a numpy array

numpy_array=np.asarray(input_data)

#reshape the numpy array as we are predicting for only one instance

input_data_reshape=numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshape)


if prediction[0]==1:

    print("The person  has heart disease")
    print("Accuracy is around: ",training_data_accuracy*100)

    newwindow = tk1.Tk()
    newwindow.title("Heart Disease Detector App")
    newwindow.geometry("1280x720")

    bg = tk1.PhotoImage(file="C:\\Users\\Ishan Vishwakarma\\Downloads\\bggg.png")
    label1 = tk1.Label(newwindow, image=bg)
    label1.place(x=0, y=0)

    label2 = tk1.Label(newwindow, text="Welcome to Heart Disease Detector App",borderwidth=8,relief="groove")
    label2.config(font=('Helvetica bold', 26,'bold'))
    label2.place(x=350, y=20)



    label4 = tk1.Label(newwindow, text=" You have heart disease",borderwidth=4,relief="raised")
    label4.config(font=('Helvetica bold', 26))
    label4.place(x=480, y=300)



    # Create Frame
    frame1 = tk1.Frame(newwindow)
    frame1.pack(pady=20)
    newwindow.mainloop()

else:

    print("The person is free from heart disease")
    print("Accuracy is around: ",100-training_data_accuracy*100)
    newwindow = tk1.Tk()
    newwindow.title("Heart Disease Detector App")
    newwindow.geometry("1280x720")

    bg = tk1.PhotoImage(file="C:\\Users\\Ishan Vishwakarma\\Downloads\\bggg.png")
    label1 = tk1.Label(newwindow, image=bg)
    label1.place(x=0, y=0)

    label2 = tk1.Label(newwindow, text="Welcome to Heart Disease Detector App",borderwidth=8,relief="solid")
    label2.config(font=('Helvetica bold',26,'bold'))
    label2.place(x=320, y=20)


    label3 = tk1.Label(newwindow, text=" 'Congratulations' ",borderwidth=6,relief="groove")
    label3.config(font=('Helvetica bold',26,'bold'))
    label3.place(x=480, y=250)

    label4 = tk1.Label(newwindow, text=" You are free from heart disease",borderwidth=4,relief="raised")
    label4.config(font=('Helvetica bold', 26))
    label4.place(x=400,y=400)





    #label2.pack(pady=20,padx=10)
    #label3.pack(pady=220,padx=2,anchor = 'ne')




    # Create Frame
    frame1 = tk1.Frame(newwindow)
    frame1.pack(pady=20)



    newwindow.mainloop()









        # A Label widget to show in toplevel







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
