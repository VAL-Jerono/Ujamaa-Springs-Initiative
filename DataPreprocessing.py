import pandas as pd
from DataCleaning import DataCleaning
from DataSourcing import DataSourcing

class DataPreprocessing(DataCleaning):
    def __init__(self, df=None, df_path=None):
        super().__init__(df=df, df_path=df_path)
        if df is not None:
            super().__init__(data=df)
        elif df_path is not None:
            super().__init__(df_path)
        else:
            raise ValueError("Either 'df' or 'df_path' must be provided.")

    def funder_wrangler(self, row):
        if row['funder'] == 'Government Of Tanzania':
            return 'gov'
        elif row['funder'] == 'Danida':
            return 'danida'
        elif row['funder'] == 'Hesawa':
            return 'hesawa'
        elif row['funder'] == 'Rwssp':
            return 'rwssp'
        elif row['funder'] == 'World Bank':
            return 'world_bank'
        else:
            return 'other'

    def installer_wrangler(self, installer):
        if installer == 'DWE':
            return 'dwe'
        elif installer == 'Government':
            return 'gov'
        elif installer == 'RWE':
            return 'rwe'
        elif installer == 'Commu':
            return 'commu'
        elif installer == 'DANIDA':
            return 'danida'
        else:
            return 'other'

    def scheme_wrangler(self, row):
        if row['scheme_management'] == 'VWC':
            return 'vwc'
        elif row['scheme_management'] == 'WUG':
            return 'wug'
        elif row['scheme_management'] == 'Water authority':
            return 'wtr_auth'
        elif row['scheme_management'] == 'WUA':
            return 'wua'
        elif row['scheme_management'] == 'Water Board':
            return 'wtr_brd'
        else:
            return 'other'

    def calculate_days_since_recorded(self):
        if self.merged_data is not None:
            # Calculate 'days_since_recorded' column
            self.merged_data['date_recorded'] = pd.Timestamp('2013-12-03') - pd.to_datetime(self.merged_data['date_recorded'])
            self.merged_data.columns = ['days_since_recorded' if x == 'date_recorded' else x for x in self.merged_data.columns]
            self.merged_data['days_since_recorded'] = self.merged_data['days_since_recorded'].astype('timedelta64[D]').astype(int)
            print(self.merged_data['days_since_recorded'].describe())
        else:
            print("DataFrame is not loaded. Please load the DataFrame first.")

    def preprocess_data(self):
        if self.merged_data is not None:
            # Additional modification
            self.merged_data['public_meeting'] = self.merged_data['public_meeting'].fillna('Unknown')

            # Apply funder_wrangler
            self.merged_data['funder'] = self.merged_data.apply(lambda row: self.funder_wrangler(row), axis=1)

            # Apply installer_wrangler
            self.merged_data['installer'] = self.merged_data['installer'].apply(self.installer_wrangler)

            # Apply scheme_wrangler
            self.merged_data['scheme_management'] = self.merged_data.apply(lambda row: self.scheme_wrangler(row), axis=1)

            # Replace unknown values in 'permit' with 'Unknown'
            self.merged_data['permit'] = self.merged_data['permit'].fillna('Unknown')

            # Check if 'date_recorded' column exists before operations
            if 'date_recorded' in self.merged_data.columns:
                # Calculate 'days_since_recorded' column
                self.merged_data['date_recorded'] = pd.to_datetime(self.merged_data['date_recorded'])
                self.merged_data['days_since_recorded'] = (pd.to_datetime('2013-12-03') - self.merged_data['date_recorded']).dt.days

                # Drop 'date_recorded' column after using it
                self.merged_data = self.merged_data.drop(columns=['date_recorded'])

            # Create an instance of DataSourcing
            data_sourcer = DataSourcing()

            # Handle missing values in 'scheme_management'
            data_sourcer.handle_missing_values('scheme_management')

            # Turn 'construction_year' into a categorical column
            self.merged_data['construction_year'] = self.merged_data['construction_year'].apply(self.construction_wrangler)

            # Display remaining null values
            print("Remaining Null Values:")
            print(self.merged_data.isnull().sum())
        else:
            print("DataFrame is not loaded. Please load the DataFrame first.")

    def construction_wrangler(self, construction_year):
        if 1960 <= construction_year < 1970:
            return '60s'
        elif 1970 <= construction_year < 1980:
            return '70s'
        elif 1980 <= construction_year < 1990:
            return '80s'
        elif 1990 <= construction_year < 2000:
            return '90s'
        elif 2000 <= construction_year < 2010:
            return '00s'
        elif construction_year >= 2010:
            return '10s'
        else:
            return 'unknown'
    def save_final_data(self, output_path="final_preprocessed_data.csv"):
        """
        Save the final preprocessed DataFrame to a CSV file.

        Parameters:
        - output_path (str): The path where the CSV file will be saved.

        Returns:
        - None
        """
        if self.merged_data is not None:
            self.merged_data.to_csv(output_path, index=False)
            print(f"Final Preprocessed DataFrame saved to {output_path}")
        else:
            print("DataFrame is not available. Please run preprocessing steps first.")


# Assuming 'train_clean.csv' is your dataset, replace it with your actual file path.
df_path = 'train_clean.csv'

# Load the DataFrame from the CSV file
train_clean_data = pd.read_csv(df_path)

# Instantiate DataPreprocessing class with the DataFrame
preprocessing_instance = DataPreprocessing(df=train_clean_data)

# Calculate 'days_since_recorded' column
preprocessing_instance.calculate_days_since_recorded()

# Display data summary before preprocessing
preprocessing_instance.display_summary()

# Perform preprocessing steps
preprocessing_instance.preprocess_data()

# Display data summary after preprocessing
preprocessing_instance.display_summary()

# Save the preprocessed DataFrame to CSV (optional)
preprocessing_instance.save_final_data("train_preprocessed.csv")
