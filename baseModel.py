import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report


class BaseModel:
    def __init__(self, model):
        self.model = model
        self.preprocessor = None

    def transform_categorical_to_numeric(self, X):
        # Make a copy of the DataFrame to avoid modifying the original
        X_copy = X.copy()

        # Label encode the categorical columns
        label_encoder = LabelEncoder()

        categorical_columns = X_copy.select_dtypes(include="object").columns

        for column in categorical_columns:
            X_copy[column] = label_encoder.fit_transform(X_copy[column])

        # Create Column Transformer with two types of transformers
        oh_transformer = OneHotEncoder()

        cat_features = X_copy.select_dtypes(include="object").columns
        num_features = X_copy.select_dtypes(exclude="object").columns

        # Create a transform pipeline
        transformer = ColumnTransformer(
            [
                ("OneHotEncoder", oh_transformer, cat_features),
                ("StandardScaler", StandardScaler(), num_features),
            ]
        )

        # Transform the entire DataFrame
        X_transformed = pd.DataFrame(transformer.fit_transform(X_copy))
        self.preprocessor = transformer

        return X_transformed

    def train(self, X_train, y_train):
        X_train_transformed = self.transform_categorical_to_numeric(X_train)
        self.model.fit(X_train_transformed, y_train)

    def predict(self, X):
        X_transformed = self.transform_categorical_to_numeric(X)
        return self.model.predict(X_transformed)

    def evaluate(self, X_test, y_test):
        X_test_transformed = self.transform_categorical_to_numeric(X_test)
        y_pred = self.model.predict(X_test_transformed)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def summary(self):
        print(f"Model Summary: {type(self.model).__name__}")

# Read the preprocessed DataFrame from 'train_preprocessed.csv'
df = pd.read_csv('train_preprocessed.csv')

# Assuming 'status_group' is the target variable
X = df.drop('status_group', axis=1)
y = df['status_group']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model using the BaseModel class
model_lr = LogisticRegression(max_iter=1000)
base_model_lr = BaseModel(model_lr)
base_model_lr.train(X_train, y_train)

# Evaluate the Logistic Regression model using the BaseModel class
y_pred_lr = base_model_lr.predict(X_test)
accuracy_lr = base_model_lr.evaluate(X_test, y_test)
print(f'Accuracy of Logistic Regression: {accuracy_lr}')

# Classification report for Logistic Regression
print("Classification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_lr))

# Print model summary
base_model_lr.summary()
