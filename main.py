import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("C:\\Users\\MANAB\\Downloads\\County_Level_FFS_Data_for_Shared_Savings_Program_Benchmark_PUF_2023_Offest_Assignable_2024_Starters.csv")
data.replace("*",pd.NA,inplace=True)
data.replace(".",pd.NA,inplace=True)
data.to_csv("Cleaned_Project_File.csv",index=False)
data.info()
data.describe()
print(data.isnull().sum())

df=pd.read_csv("Cleaned_Project_File.csv")
print(df)
print(df.isnull().sum())
df['PER_CAPITA_EXP_ESRD'].fillna(df['PER_CAPITA_EXP_ESRD'].mean(),inplace=True)
df['AVG_RISK_SCORE_ESRD'].fillna(df['AVG_RISK_SCORE_ESRD'].mean(),inplace=True)
df['AVG_DEMOG_SCORE_ESRD'].fillna(df['AVG_DEMOG_SCORE_ESRD'].mean(),inplace=True)
df['PERSON_YEARS_ESRD'].fillna(df['PERSON_YEARS_ESRD'].mean(),inplace=True)
df['PER_CAPITA_EXP_DIS'].fillna(df['PER_CAPITA_EXP_DIS'].mean(),inplace=True)
df['AVG_RISK_SCORE_DIS'].fillna(df['AVG_RISK_SCORE_DIS'].mean(),inplace=True)
df['AVG_DEMOG_SCORE_DIS'].fillna(df['AVG_DEMOG_SCORE_DIS'].mean(),inplace=True)
df['PERSON_YEARS_DIS'].fillna(df['PERSON_YEARS_DIS'].mean(),inplace=True)
df['PER_CAPITA_EXP_AGDU'].fillna(df['PER_CAPITA_EXP_AGDU'].mean(),inplace=True)
df['AVG_RISK_SCORE_AGDU'].fillna(df['AVG_RISK_SCORE_AGDU'].mean(),inplace=True)
df['AVG_DEMOG_SCORE_ AGDU'].fillna(df['AVG_DEMOG_SCORE_ AGDU'].mean(),inplace=True)
df['PERSON_YEARS_AGDU'].fillna(df['PERSON_YEARS_AGDU'].mean(),inplace=True)
df['PER_CAPITA_EXP_AGND'].fillna(df['PER_CAPITA_EXP_AGND'].mean(),inplace=True)
df['AVG_RISK_SCORE_AGND'].fillna(df['AVG_RISK_SCORE_AGND'].mean(),inplace=True)
df['AVG_DEMOG_SCORE_AGED/NON-DUAL'].fillna(df['AVG_DEMOG_SCORE_AGED/NON-DUAL'].mean(),inplace=True)
df['PERSON_YEARS_AGND'].fillna(df['PERSON_YEARS_AGND'].mean(),inplace=True)
columns = [
    'PER_CAPITA_EXP_ESRD',
    'AVG_RISK_SCORE_ESRD',
    'AVG_DEMOG_SCORE_ESRD',
    'PERSON_YEARS_ESRD',
    'PER_CAPITA_EXP_DIS',
    'AVG_RISK_SCORE_DIS',
    'AVG_DEMOG_SCORE_DIS',
    'PERSON_YEARS_DIS',
    'PER_CAPITA_EXP_AGDU',
    'AVG_RISK_SCORE_AGDU',
    'AVG_DEMOG_SCORE_ AGDU',
    'PERSON_YEARS_AGDU',
    'PER_CAPITA_EXP_AGND',
    'AVG_RISK_SCORE_AGND',
    'AVG_DEMOG_SCORE_AGED/NON-DUAL',
    'PERSON_YEARS_AGND'
]

print("Mean of Columns:\n")
for col in columns:
    print(f"{col} : {df[col].mean()}")

print("\n")
df['Average_Expenditure']=(df['PER_CAPITA_EXP_ESRD']+df['PER_CAPITA_EXP_DIS']+df['PER_CAPITA_EXP_AGDU']+df['PER_CAPITA_EXP_AGND'])/4
df['Total_Beneficiaries'] = df['PERSON_YEARS_ESRD'] + df['PERSON_YEARS_DIS'] + df['PERSON_YEARS_AGDU'] + df['PERSON_YEARS_AGND']
top10= df.nlargest(10,'Average_Expenditure')
print("Top 10 Counties with Highest Expenditure:\n")
print(top10[['COUNTY_NAME','Average_Expenditure']])
print("\n")
statewise = df.groupby('STATE_NAME')['Total_Beneficiaries'].sum().sort_values(ascending=False)
print("Statewise Total Beneficiaries:\n")
print(statewise)
print("\n")
df['Performance_Category'] = np.where(df['Average_Expenditure'] > df['Average_Expenditure'].mean(), 'High', 'Low')
performance=df.groupby('Performance_Category')['Total_Beneficiaries'].sum()
print(performance)
print("\n")
df.info()
print("\n")
df.describe()

plt.figure(figsize=(10,6))
sns.barplot(x='Average_Expenditure', y='COUNTY_NAME', data=top10)
plt.title("Top 10 Counties with Highest Expenditure")
plt.show()

plt.figure(figsize=(8,8))
statewise.head(10).plot.pie(autopct='%1.1f%%')
plt.title("Top 10 States by Total Beneficiaries")
plt.ylabel('')
plt.show()


df.corr(numeric_only=True)
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(x='Total_Beneficiaries', y='PER_CAPITA_EXP_AGND', hue='STATE_NAME', data=df)
plt.title('Total Beneficiaries vs Per Capita Expenditure (Aged Non-Dual)')
plt.xlabel('Total Beneficiaries')
plt.ylabel('Per Capita Expenditure')
plt.show()


plt.figure(figsize=(8,5))
sns.kdeplot(df['PER_CAPITA_EXP_ESRD'], fill=True, color='purple')
plt.title('Density Plot of Per Capita Expenditure ESRD')
plt.show()


plt.figure(figsize=(12,6))
sns.countplot(x='STATE_NAME', data=df, order=df['STATE_NAME'].value_counts().index, palette='coolwarm')
plt.xticks(rotation=90)
plt.title("Data Count per State")
plt.show()



selected_columns = [
    'PER_CAPITA_EXP_ESRD',
    'PER_CAPITA_EXP_DIS',
    'PER_CAPITA_EXP_AGDU',
    'PER_CAPITA_EXP_AGND',
    'AVG_RISK_SCORE_ESRD',
    'AVG_RISK_SCORE_DIS',
    'AVG_RISK_SCORE_AGDU',
    'AVG_RISK_SCORE_AGND'
]

plt.figure(figsize=(10,6))
sns.violinplot(data=df[selected_columns])
plt.xticks(rotation=45)
plt.title("Violin Plot of Key Features")
plt.show()


h=sns.pairplot(df[selected_columns+['STATE_NAME']],hue='STATE_NAME')
h._legend.remove()
plt.suptitle('Pair Plot of Key Expenditure and Risk Score Features',y=1.01)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 7))
sns.boxplot(data=df[selected_columns], palette='Set2')

plt.title('Boxplot of Per Capita Expenditure & Risk Scores', fontsize=16, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.xticks(rotation=45)  
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print(df.isnull().sum())
print("\n")
summary = df.groupby('STATE_NAME').agg({
    'Total_Beneficiaries':'sum',
    'Average_Expenditure':'mean'
}).reset_index()

print(summary)

print("\n")
print(df.corr(numeric_only=True))
df.to_csv("Modified_Project_File.csv")

