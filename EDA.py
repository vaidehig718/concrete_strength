# Importing the seaborn library for data visualization
import seaborn as sns

# Creating a pair plot to visualize the relationships between pairs of variables
# The diagonal elements are represented as KDE (Kernel Density Estimate) plots
sns.pairplot(df, diag_kind='kde')

# Importing the matplotlib library for data visualization
import matplotlib.pyplot as plt

# Creating a figure with a specific size for the heatmap
plt.figure(figsize=(14, 8))

# Creating a correlation heatmap to visualize the relationships between variables
# The 'annot=True' argument displays the correlation values on the heatmap
sns.heatmap(df.corr(), annot=True)

# convert the new dataframe from Python to Excel to do some data visualization in Tabluea
df.to_excel('Concrete_Cleaned.xlsx')
