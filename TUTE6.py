#PHASE : 1
#STEP : 1
import pandas as pd
import numpy as np

np.random.seed(42)
EMP = 1000

employee_ids = np.arange(1, EMP + 1)
names = ['1' + str(i) for i in range(1, EMP + 1)]
ages = np.random.randint(22, 60, EMP)
departments = np.random.choice(['HR', 'IT', 'Sales', 'Marketing', 'Finance'], EMP)
salaries = np.random.randint(40000, 120000, EMP)
years_at_company = np.random.randint(1, 30, EMP)
performance_scores = np.random.randint(1, 6, EMP) 
attrition = np.random.choice(['Yes', 'No'], EMP, p=[0.2, 0.8])
employee_data = pd.DataFrame({
    'Employee ID': employee_ids,
    'Name': names,
    'Age': ages,
    'Department': departments,
    'Salary': salaries,
    'Years at Company': years_at_company,
    'Performance Score': performance_scores,
    'Attrition': attrition
})

employee_data.to_csv('Employee_data.csv', index=False)
print(employee_data)

print(employee_data.isnull())
print(employee_data.isnull().sum())
print(employee_data.drop_duplicates())
employee_data['Attrition'] = employee_data['Attrition'].map({'Yes': 1, 'No': 0})
print(employee_data['Attrition'])
employee_data['Department'] = employee_data['Department'].map({'HR': 1, 'IT': 2,'Sales': 3, 'Marketing': 4, 'Finance': 5})
print(employee_data['Department'])

#STEP : 2
print(employee_data.describe())

import seaborn as sns
sns.pairplot(employee_data)

import matplotlib.pyplot as plt
correlation_matrix = employee_data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

sns.boxplot(x=employee_data['Salary'])
sns.boxplot(x=employee_data['Performance Score'])
plt.show()

#STEP : 3
threshold = employee_data['Performance Score'].median()
prob_leave_high_performance = len(employee_data[(employee_data['Performance Score'] > threshold) & (employee_data['Attrition'] == 'Yes')]) / len(employee_data[employee_data['Performance Score'] > threshold])
print(prob_leave_high_performance)
from scipy import stats
departments = employee_data['Department'].unique()
performance_scores_by_dept = [employee_data[employee_data['Department'] == dept]['Performance Score'] for dept in departments]
f_stat, p_value = stats.f_oneway(*performance_scores_by_dept)
print(departments)
print(performance_scores_by_dept)
print(f_stat)
print(p_value)

#PHASE : 2
#STEP : 4
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
employee_data[['Salary', 'Performance Score']] = scaler.fit_transform(employee_data[['Salary', 'Performance Score']])
print(employee_data[['Salary', 'Performance Score']])

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
employee_data['Department'] = label_encoder.fit_transform(employee_data['Department'])
print(employee_data['Department'])

#STEP : 5
from sklearn.model_selection import train_test_split
X = employee_data.drop(['Attrition'], axis=1)
y = employee_data['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
plt.show()
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.show()

#STEP : 6
from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score
y_pred = regression_model.predict(X_test)
print(f"R^2: {r2_score(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
''''#PHASE : 3
#STEP : 7
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50)

model.evaluate(X_test, y_test)

#STEP : 8
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50)

import matplotlib.pyplot as plt
import seaborn as sns'''''

plt.figure(figsize=(10, 6))
sns.lineplot(x='Years at Company', y='Performance Score', data=employee_data)
plt.title('Performance Trends Over Years at Company')
plt.xlabel('Years at Company')
plt.ylabel('Performance Score')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Department', hue='Attrition', data=employee_data)
plt.title('Department-wise Attrition')
plt.xlabel('Department')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Salary', y='Performance Score', data=employee_data)
plt.title('Salary vs. Performance Score')
plt.xlabel('Salary')
plt.ylabel('Performance Score')
plt.show()
