import pandas as pd
from DataSourcing import DataSourcing
from DataPreprocessing import DataPreprocessing
from baseModel import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
# Assuming you have a separate testing dataset in a file 
# named 'test_set_values.csv'
# Assuming you have a separate testing dataset in a file named 'test_set_values.csv'
test_data_path = "test_set_values.csv"

test_source_data = pd.read_csv(test_data_path)

data_cleaner = DataCleaning(data=test_source_data)
