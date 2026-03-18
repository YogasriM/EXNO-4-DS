# EXNO:4-DS 
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/877b7ff2-91af-45be-bb41-3f3e51c87cec)

```
data.isnull().sum()
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/24d52a3a-6d98-4504-aff8-71cf74757cc4)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/e9cfb80e-9357-4d4a-a985-4a5b25deabc4)

```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/a696948d-858f-44b5-9fb3-b7b0752cd8cb)

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/bcdbade0-9c3e-49f1-a75b-07ff53304fa7)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/2a426f2e-b39d-4966-8177-9c7bd7db4309)

```
data2
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/6b8a4d7f-ad12-490d-ade6-6eb773abe19c)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/5fe540f9-f0a5-4760-bad1-209336864d93)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/d541056f-6738-4195-bfb9-8a2797a98e85)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/69887669-630a-4162-8440-4362b75d4eb5)

```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/24c06b7c-937f-4753-bb30-f789eeb636c4)

```
x=new_data[features].values
print(x)
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/1788c3ec-1567-4397-83b9-bb26fd3bd576)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/db6a7473-c468-4125-9996-27d9126780e2)

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/9d36ea6e-84a3-43f3-9149-ed446d87eb70)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/f9ac5352-d49b-420e-8059-fc342d8724b6)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/77ec696c-0f9a-48ba-9261-b3c117af58f8)

```
data.shape
```
![image](https://github.com/Sriram8452/EXNO-4-DS/assets/118708032/ec6e0a89-10a3-470b-ab2a-1d4256e75300)

<img width="635" height="654" alt="Screenshot 2026-02-24 135445" src="https://github.com/user-attachments/assets/f7cbcf09-a797-40cf-b19b-f53edccec3ac" />

<img width="702" height="533" alt="Screenshot 2026-02-24 135452" src="https://github.com/user-attachments/assets/b398f102-6817-44f0-aac9-cdd981b7b81a" />


<img width="696" height="775" alt="Screenshot 2026-02-24 135505" src="https://github.com/user-attachments/assets/bcd1f681-ab45-41be-9da5-240d259987a2" />


<img width="712" height="518" alt="Screenshot 2026-02-24 135517" src="https://github.com/user-attachments/assets/8e4061f0-c036-4e7b-8321-07c613c4b80f" />

<img width="789" height="550" alt="Screenshot 2026-02-24 135524" src="https://github.com/user-attachments/assets/4d581870-dcc6-4c21-bde3-fd782e81fee3" />


<img width="759" height="576" alt="Screenshot 2026-02-24 135530" src="https://github.com/user-attachments/assets/023ba4e1-9c0d-432d-b978-4c68d9bea07e" />

<img width="780" height="368" alt="Screenshot 2026-02-24 135537" src="https://github.com/user-attachments/assets/b6c92e29-e799-40cf-97d1-0a24757369b9" />

# RESULT:
     
Thus the the given data is performed Feature Scaling and Feature Selection process and saved the data to a file successfully. 


