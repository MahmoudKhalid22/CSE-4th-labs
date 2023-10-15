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
pre_df = df.drop('Unnamed: 32', axis=1)

# Separating features and target variable

X = pre_df.drop('diagnosis', axis=1)  # Assuming you want to drop the 'diagnosis' column
y = pre_df['diagnosis']

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
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
cmd.plot()

# Classification Report
print(classification_report(y_test, y_pred))


