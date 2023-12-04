# DataCleaning.py
from DataSourcing import DataSourcing
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

class DataCleaning(DataSourcing):
    COLUMNS_TO_DROP = ['gps_height', 'longitude', 'latitude', 'id', 'region_code', 'district_code',
                       'num_private', 'source', 'quality_group', 'quantity_group', 'payment',
                       'management_group', 'management', 'extraction_type', 'extraction_type_group',
                       'recorded_by', 'region', 'lga', 'ward', 'scheme_name', 'wpt_name', 'subvillage']

    def __init__(self, df_path=None, data=None, **kwargs):
        super().__init__(df_path)
        if data is not None:
            self.merged_data = data.copy()
        else:
            super().__init__(df_path)
            self.merged_data = self.df.copy() if self.df is not None else None

        self.dropped_columns = []

    def drop_columns_and_display_summary(self):
        """
        Drop specified columns and display a summary of the cleaned data.

        Returns:
        - None
        """
        self.drop_columns(self.COLUMNS_TO_DROP)

        # Display a summary of the cleaned data
        self.display_summary()

    def handle_duplicates(self):
        self.merged_data.drop_duplicates(inplace=True)

    def preprocess_numeric_columns(self, numeric_columns, fillna_value=None):
        self.merged_data[numeric_columns] = self.merged_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        if fillna_value is not None:
            self.merged_data[numeric_columns] = self.merged_data[numeric_columns].fillna(fillna_value)
        else:
            self.merged_data = self.merged_data.dropna(subset=numeric_columns)

        self.merged_data = self.merged_data.replace([np.inf, -np.inf], np.nan)

    def detect_outliers(self, numeric_columns):
        self.preprocess_numeric_columns(numeric_columns)

        z_scores = zscore(self.merged_data[numeric_columns])
        abs_z_scores = np.abs(z_scores)
        outliers = (abs_z_scores > 3).all(axis=1)

        self.merged_data = self.merged_data[~outliers]

    def standardize_data(self, numeric_columns):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.merged_data[numeric_columns] = scaler.fit_transform(self.merged_data[numeric_columns])

    def calculate_vif(self):
        numeric_columns = self.merged_data.select_dtypes(include=['float64', 'int64']).columns
        vif_data = pd.DataFrame()
        vif_data["Variable"] = numeric_columns
        vif_data["VIF"] = [variance_inflation_factor(self.merged_data[numeric_columns].values, i) for i in range(len(numeric_columns))]
        return vif_data

    def display_correlation_heatmap(self):
        numeric_columns = self.merged_data.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = self.merged_data[numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

    # Inside the DataCleaning class
    def drop_columns(self, columns_to_drop):
        self.merged_data = self.merged_data.drop(columns=columns_to_drop, axis=1)
        self.dropped_columns.extend(columns_to_drop)
    
    def save_cleaned_data(self, output_path="train_clean.csv"):
        """
        Save the cleaned DataFrame to a CSV file.

        Parameters:
        - output_path (str): The path where the CSV file will be saved.

        Returns:
        - None
        """
        if self.merged_data is not None:
            self.merged_data.to_csv(output_path, index=False)
            print(f"Cleaned DataFrame saved to {output_path}")

    def display_summary(self):
        if self.merged_data is not None:
            print("Data Cleaning Summary:")
            print(f"Number of Rows: {len(self.merged_data)}")
            print(f"Number of Columns: {len(self.merged_data.columns)}")
            print("\nColumn Names:")
            print(self.merged_data.columns)
            print("\nNull Values:")
            print(self.merged_data.isnull().sum())

# Example usage
df_path = "train_source.csv"
train_source_data = pd.read_csv(df_path)

data_cleaner = DataCleaning(data=train_source_data)

# Handle duplicates
data_cleaner.handle_duplicates()

# Detect and handle outliers in numeric columns
numeric_columns_to_handle_outliers = ['id', 'amount_tsh', 'gps_height', 'longitude', 'latitude',
        'num_private', 'region_code', 'district_code', 'population',
        'construction_year']
data_cleaner.detect_outliers(numeric_columns_to_handle_outliers)

# Standardize numeric columns
numeric_columns_to_standardize = ['id', 'amount_tsh', 'gps_height', 'longitude', 'latitude',
       'num_private', 'region_code', 'district_code', 'population',
       'construction_year']
data_cleaner.standardize_data(numeric_columns_to_standardize)

# Display correlation heatmap
data_cleaner.display_correlation_heatmap()

# Drop specific columns and display a summary
data_cleaner.drop_columns_and_display_summary()

# Save the cleaned data to a CSV file
data_cleaner.save_cleaned_data("train_clean.csv")

# Display a summary of the loaded and merged data
data_cleaner.display_summary()