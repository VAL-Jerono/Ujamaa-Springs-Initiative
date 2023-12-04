import pandas as pd

class DataSourcing:
    def __init__(self, df_path=None):
        self.df_path = df_path
        self.df = None

        if df_path is not None:
            self.load_data()

    def load_data(self):
        try:
            train_value = pd.read_csv("training_set_values.csv")
            train_label = pd.read_csv("training_set_labels.csv")

            # Check if 'id' column is present in both DataFrames
            if 'id' in train_value.columns and 'id' in train_label.columns:
                # Merge using 'id' as the key
                self.df = train_value.merge(train_label, how="outer", on="id", sort=True)

                # Check for null values after merging
                if self.df.isnull().sum().any():
                    print("Warning: Null values present in the merged DataFrame.")
                    # Handle null values based on your strategy (e.g., user input)
                    # self.handle_missing_values()

            else:
                print("Error: 'id' column is missing in one or both DataFrames")

        except FileNotFoundError:
            print("Error: One or both CSV files not found.")

    def save_to_csv(self, output_path="train_source.csv"):
        if self.df is not None:
            self.df.to_csv(output_path, index=False)
            print(f"DataFrame saved to {output_path}")

    def display_summary(self):
        if self.df is not None:
            print("Data Summary:")
            print(f"Number of Rows: {len(self.df)}")
            print(f"Number of Columns: {len(self.df.columns)}")
            print("\nColumn Names:")
            print(self.df.columns)
            print("\nNull Values:")
            print(self.df.isnull().sum())

    def handle_missing_values(self, column_name):
        if self.df is not None and column_name in self.df.columns:
            self.df[column_name] = self.df[column_name].fillna('Unknown')
        else:
            print(f"Column '{column_name}' not found in the DataFrame.")
            
    def get_dataframe(self):
        return self.df


# # Example usage
df_path = "training_set_values.csv"

data_sourcer = DataSourcing(df_path)
data_sourcer.display_summary()
data_sourcer.save_to_csv('train_source.csv')  # Use the default output path

# # Get the modified DataFrame
# df = data_sourcer.get_dataframe() 