# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Sample data
# X = [[1, 2], [2, 3], [3, 4], [4, 5], [2, 1], [3, 2], [4, 3], [5, 4]]
# y = [0, 0, 0, 0, 1, 1, 1, 1]

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# # Create a Gaussian Naive Bayes Classifier
# model = GaussianNB()

# # Train the model using the training sets
# model.fit(X_train, y_train)

# # Predict Output
# predicted = model.predict(X_test)

# # Check accuracy
# print("Accuracy:", accuracy_score(y_test, predicted))

# # يفترض أن لديك النموذج model متاحًا بالفعل

# # بيانات الإدخال الجديدة
# new_data = [[5, 7]]

# # التنبؤ بالفئة الجديدة
# predicted_class = model.predict(new_data)

# print("Predicted class:", predicted_class)

# -----------------------------------------------------------

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# FOR ANALYZING DATA
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# FOR APPLYING ALGORITHM
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier        

dataset = pd.read_csv('./data.csv')

dataset.head()

# dataset.info()
print(dataset.info());

dataset = dataset.drop(labels=['id','Unnamed: 32'],axis = 1)

features = dataset.iloc[:,1:]
labels = dataset.iloc[:,0].values


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(features.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


to_be_dropped = ['perimeter_mean','radius_mean','compactness_mean',
                 'concave points_mean','radius_se','perimeter_se',
                 'radius_worst','perimeter_worst','compactness_worst',
                 'concave points_worst','compactness_se','concave points_se',
                 'texture_worst','area_worst']


features_temp = features.drop(to_be_dropped,axis = 1) 

f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(features_temp.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
