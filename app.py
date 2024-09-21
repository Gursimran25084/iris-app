import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.datasets import load_iris # Changed the import statement to correctly import the load_iris function from the sklearn.datasets module
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
iris=load_iris()
#=iris.sample(frac=1,random_state=40)

#iris.feature_names

x=iris.data
print(x)

y=iris.target
print(y)

# split the dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# apply random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100)

classifier.fit(x_train,y_train)

pred_clf=classifier.predict(x_test)

acc_train=classifier.score(x_train,y_train)
acc_val=classifier.score(x_test,y_test)
print("Accuracy of training dataset:"+str(acc_train))
print("Accuracy of test dataset:"+str(acc_val))

#Model evaluation
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,pred_clf))

print(accuracy_score(y_test,pred_clf))


joblib.dump(classifier,"rf_model.sav")


joblib.dump(classifier,"rf_model.sav")


joblib.dump(classifier,"rf_model.sav")

import streamlit as st # import the streamlit library


from predict import Predict

st.title("Classifying Iris Flowers")
st.markdown("Toy model to play classify iris flowers into setosa,versicolor,virginica")

st.header("Plant Features")
col1,col2=st.columns(2)
with col1:

  st.text('Sepal characteristics')
  sepal_l=st.slider('sepal lenght (cm)',1.0,8.0,0.5)
  sepal_w=st.slider('sepal_width (cm)',2.0,4.4,0.5)

with col2:

  st.text('petal characteristics')
  petal_l=st.slider('petal length (cm)',1.0,7.0,0.5)
  petal_w=st.slider('petal_width (cm)',0.1,2.5,0.5)
if st.button("predict type of iris"):
    result=predict(np.array([[sepal_l,sepal_w,petal_l,petal_w]]))
    st.text(result[0])
    





