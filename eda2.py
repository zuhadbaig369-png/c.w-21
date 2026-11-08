import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Titanic Dataset.csv")

data.head(5)
sns.countplot(x='Gender', hue='Survived', data=data)
plt.title('Survival by Gender')
plt.show()
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Survival by Passenger Class')
plt.show()
sns.histplot(data['Age'], bins=40, kde=False)
plt.title('Distribution of Passenger Age')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()
sns.boxplot(x='Gender', data=data)
plt.title("Passenger by Gender")
plt.show()
sns.countplot(x='Survived', hue='SibSp', data=data, palette='mako')
plt.title('Survival by Number of Siblings/Spouses')
plt.show()
sns.countplot(x='Survived', hue='Parch', data=data, palette='mako')
plt.title('Survival by Number of Parents/Children')
plt.show()
sns.histplot(data['Fare'], kde=True)
plt.title('Distribution of Fare')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()
sns.boxplot(data=data, x='Pclass', y='Age', hue='Pclass', legend=False, palette='winter')
plt.title('Age Distribution by Passenger Class')
plt.show()

numeric_data = data.select_dtypes(include=['number'])

plt.figure(figsize=(10, 7))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()