# Naive Base Classifier
import pandas as pd   # to load data
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report
)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

df = pd.read_csv('./data.csv')
df.head()

df.info()


sns.countplot(data=df,x='diagnosis',hue='radius_mean')
plt.xticks(rotation=45, ha='right')



# Dropping the 'Unnamed: 32' column
# pre_df = df.drop('Unnamed: 32', axis=1)

# Separating features and target variable

# X = df.drop('diagnosis', axis=1)  # Assuming you want to drop the 'diagnosis' column
X = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']]
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


# Fit the model on the transformed data


model = GaussianNB()
imputer = SimpleImputer(strategy='mean')

# Create a pipeline with the imputer and Gaussian Naive Bayes classifier
pipeline = make_pipeline(imputer, GaussianNB())

pipeline.fit(X_train, y_train)

model.fit(X_train, y_train)

# Predicting the target variable using the test data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average='weighted')

print('Accuracy', accuracy)
print("F1 Score", f1)



# Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# cmd = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
# cmd.plot()

# Classification Report
print(classification_report(y_test, y_pred))


plt.show()



# Assuming you have a new_data variable containing the new data point
new_data = [[14.2, 20.5, 94.7, 0.102, 0.145, 0.186, 0.104, 0.172, 0.175, 0.055, 0.543, 0.476, 3.07, 35.2, 0.008, 0.036, 0.031, 0.011, 0.024, 0.004, 15.3, 24.2, 102.5, 0.163, 0.369, 0.508, 0.242, 0.382, 0.091, 0.03]]


# Use the predict method on your model
# print(X.columns)
output = model.predict(new_data)

# Print the output
print("Predicted Output:", output)
