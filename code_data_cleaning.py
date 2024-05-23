# Import the pandas library for data manipulation and analysis
import pandas as pd
# Read the 'concrete.csv' file into a DataFrame
df = pd.read_csv('/concrete.csv')
# Display the first four rows of the DataFrame
df.head(4)

# Retrieve and display the shape of the DataFrame to determine its dimensions
df.shape

# Retrieve and print the column names of the DataFrame
df.columns

# Renaming columns to provide clear and representative feature names
df = df.rename(columns={'cement': 'Cement', 'slag': 'Blast_Furnace_Slag', 'ash':'Fly_Ash', 'water': 'Water', 'superplastic': 'Superplasticizer', 'coarseagg': 'Coarse_Aggregate', 'fineagg':'Fine_Aggregate', 'age': 'Age_Days', 'strength': 'Strength'})

# Displaying the updated column names
df.columns

# Displaying concise summary information about the DataFrame
df.info()

# Checking and counting the number of duplicated observations in the DataFrame
df.duplicated().sum()

# Dropping duplicated rows to ensure data integrity and eliminate redundancy
df = df.drop_duplicates()

# After removing the duplicated rows, the DataFrame now has updated dimensions
df.shape

# The sum of the sum() function on df.isna() provides the total count of missing values
# In this case, the result is 0, indicating that the DataFrame has no missing values
df.isna().sum().sum()

# Checking the data type for each column in the DataFrame
df.dtypes

# Counting the number of unique values in each feature
# The result is sorted in descending order to identify features with the highest variability
df.nunique().sort_values(ascending=False)

# Generating descriptive statistics for the DataFrame
df.describe()
















