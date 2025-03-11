<H3>ENTER YOUR NAME: Tirupathi Jayadeep</H3>
<H3>ENTER YOUR REGISTER NO.: 212223240169</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 10-03-2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
## Import Libraries
```
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```
## Read the dataset
```
df=pd.read_csv("Churn_Modelling.csv")
print(df)
```
## Values of X and Y
```
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
```
## Check for outliers
```
df.describe()
```
## Check the missing data
```
print(df.isnull().sum())
df.fillna(df.mean().round(1), inplace=True)
print(df.isnull().sum())
y = df.iloc[:, -1].values
print(y)
```
## Checking for duplicates
```
df.duplicated()
```
## Dropping string value from dataset
```
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
```
## Normalize the dataset
```
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```
## Training and testing the model
```
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```

## OUTPUT:
## Reading Data
![image](https://github.com/user-attachments/assets/ab4d674e-c2b0-4c31-9cd6-13bc6c9a6c6e)
## Duplicates Identification
![image](https://github.com/user-attachments/assets/4cee307e-652c-4b55-8223-cbf2c87a73b0)
## Values of 'Y'
![image](https://github.com/user-attachments/assets/0abacebf-b5f5-483e-9710-a40dffef20a5)
## Outliers
![image](https://github.com/user-attachments/assets/cf76f74a-2243-4aac-8218-f08121735ecf)
## Checking datasets after dropping string values data from dataset
![image](https://github.com/user-attachments/assets/754e830b-5933-4767-97a6-49366ce2724f)
## Normalize the dataset
![image](https://github.com/user-attachments/assets/4029d204-8e67-4508-b489-464bc4102f7f)
## Split the dataset
![image](https://github.com/user-attachments/assets/564e15a0-263d-43d0-952f-cbbf15e42f7e)
## Training the Model
![image](https://github.com/user-attachments/assets/b4da8a5d-e215-4b97-a4ad-c280844019f3)
## Testing the Model
![image](https://github.com/user-attachments/assets/11ebc34e-48ed-4988-84ce-8106841c56a7)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


