from sklearn.ensemble import RandomForestClassifier
from baseModel import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, random_state=None):
        model_rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        super().__init__(model_rf)

# Read the preprocessed DataFrame from 'train_preprocessed.csv'
df = pd.read_csv('train_preprocessed.csv')

# Assuming 'status_group' is the target variable
X = df.drop('status_group', axis=1)
y = df['status_group']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize and train the Random Forest model using the BaseModel class
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
base_model_rf = BaseModel(model_rf)
base_model_rf.train(X_train, y_train)

# Evaluate the Random Forest model using the BaseModel class
y_pred_rf = base_model_rf.predict(X_test)
accuracy_rf = base_model_rf.evaluate(X_test, y_test)
print(f'Accuracy of Random Forest: {accuracy_rf}')

# Classification report for Random Forest
print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Print Random Forest model summary
base_model_rf.summary()
