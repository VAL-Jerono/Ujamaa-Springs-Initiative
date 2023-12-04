import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from DataPreprocessing import DataPreprocessing
class DataAnalysis(DataPreprocessing):
    def __init__(self, df=None, df_path=None):
        super().__init__(df=df, df_path=df_path)

    def handle_warnings(self):
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)

    def fillna_median(self):
        numeric_columns = self.merged_data.select_dtypes(include=['number']).columns
        self.merged_data[numeric_columns] = self.merged_data[numeric_columns].fillna(self.merged_data[numeric_columns].median())

    def plot_distribution_categorical(self, categorical_feature, target_column='status_group'):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=categorical_feature, hue=target_column, data=self.merged_data)
        plt.title(f'Distribution of {categorical_feature} by {target_column}')
        plt.show()

    def plot_pairplot_numeric(self, numeric_features, target_column='status_group'):
        plt.figure(figsize=(12, 8))
        sns.pairplot(self.merged_data, hue=target_column, vars=numeric_features)
        plt.suptitle(f'Pairplot of Numeric Features by {target_column}', y=1.02)
        plt.show()

    def perform_eda(self, target_column='status_group'):
        self.handle_warnings()
        self.fillna_median()

        # Identify numeric and categorical features
        numeric_features = self.merged_data.select_dtypes(include=['number']).columns
        categorical_features = self.merged_data.select_dtypes(exclude=['number']).columns

        # Plot pairplots for numeric features
        if not numeric_features.empty:
            self.plot_pairplot_numeric(numeric_features, target_column)

        # Plot distribution of categorical features
        for feature in categorical_features:
            self.plot_distribution_categorical(feature, target_column)

# Example usage
df_path = 'train_preprocessed.csv'
train_preprocessed_data = pd.read_csv(df_path)

analysis_instance = DataAnalysis(df=train_preprocessed_data)
analysis_instance.perform_eda(target_column='status_group')
