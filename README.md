# Exploratory Data Analysis (EDA) – WorkFlow

## 1. Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## 2. Load Dataset

```python
df = pd.read_csv('data.csv')
df.head()
```

## 3. Understand Dataset

```python
# Check first 5 rows
df.head()

# Check last 5 rows
df.tail()

# Check random rows
df.sample(5)

# Check rows and columns
df.shape

# Check column names
df.columns

# Check data types and null values
df.info()

# Statistical summary
df.describe()

# Statistical summary of all columns
df.describe(include='all')
```

## 4. Type Conversion

```python
# Convert numeric columns
df['math_marks'] = pd.to_numeric(df['math_marks'], errors='coerce')
df['physics_marks'] = pd.to_numeric(df['physics_marks'], errors='coerce')
df['chemistry_marks'] = pd.to_numeric(df['chemistry_marks'], errors='coerce')
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Convert date column
df['exam_date'] = pd.to_datetime(df['exam_date'], errors='coerce')

# Convert data type using astype()

df['age'] = df['age'].astype('int64')
df['math_marks'] = df['math_marks'].astype('float64')
df['name'] = df['name'].astype('string')
```

## 5. Remove Duplicate

```python
# Check full row duplicates
df.duplicated().sum()

# Show duplicate rows
df[df.duplicated(keep=False)].sort_values(by='name')

# Remove full row duplicates
df.drop_duplicates(inplace=True)

# Check duplicate student records
df[df.duplicated(subset=['student_id'], keep=False)].sort_values(by='student_id')

# Remove duplicate student records
df.drop_duplicates(subset=['student_id'], inplace=True)
```

## 6. Handling Null Values

```python
# Check missing values
df.isnull().sum()

# Check rows containing null values
df[df.isnull().any(axis=1)]

# Check total missing values
df.isnull().sum().sum()

# Drop rows when all column values are null
df.dropna(how='all', inplace=True)

# Drop rows when important column values are null
df.dropna(subset=['name', 'student_id'], inplace=True)
```

### Fill Missing Values

```python
# Fill date null values
df['exam_date'] = df['exam_date'].bfill()

# Fill city null values
df['city'] = df['city'].fillna('Unknown')

# Fill age null values
df['age'] = df['age'].fillna(df['age'].mean())

# Fill gender null values
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

# Fill department null values
df['department'] = df['department'].fillna(df['department'].mode()[0])

# Fill marks null values
df['math_marks'] = df['math_marks'].fillna(df['math_marks'].mean())
df['physics_marks'] = df['physics_marks'].fillna(df['physics_marks'].mean())

# Fill chemistry marks using gender-wise mean
gender_mean = df.groupby('gender')['chemistry_marks'].mean()
df['chemistry_marks'] = df['chemistry_marks'].fillna(df['gender'].map(gender_mean))

# Check remaining null values
df.isnull().sum()
```

## 7. Inconsistent Categorical Values

### Clean City Names

```python
df['city'].unique()

df['city'] = df['city'].str.title().str.strip()

city_mapping = {
    'Dk': 'Dhaka',
}

df['city'] = df['city'].replace(city_mapping)

df['city'].unique()
```

### Clean Gender Names

```python
df['gender'].unique()

df['gender'] = df['gender'].str.title().str.strip()

gender_mapping = {
    'M': 'Male',
    'F': 'Female',
    'Male': 'Male',
    'Female': 'Female',
}

df['gender'] = df['gender'].map(gender_mapping)

df['gender'].unique()
```

## 8. Handle Invalid Values

### Check Age

```python
# Find invalid age values
df[~df['age'].between(0, 100)]

# Replace invalid age with mode
df['age'] = df['age'].where(
    df['age'].between(0, 100),
    other=df['age'].mode()[0]
)
```

### Check Marks

```python
# Find invalid marks
df[~df['math_marks'].between(0, 100)]
df[~df['physics_marks'].between(0, 100)]
df[~df['chemistry_marks'].between(0, 100)]
```

```python
# Replace invalid marks with NaN
df['math_marks'] = df['math_marks'].where(df['math_marks'].between(0, 100))
df['physics_marks'] = df['physics_marks'].where(df['physics_marks'].between(0, 100))
df['chemistry_marks'] = df['chemistry_marks'].where(df['chemistry_marks'].between(0, 100))
```

## 9. Feature Engineering

```python
# Calculate total marks
df['total_marks'] = df['math_marks'] + df['physics_marks'] + df['chemistry_marks']

# Calculate average marks
df['average_marks'] = df[['math_marks', 'physics_marks', 'chemistry_marks']].mean(axis=1)

# Create result column using np.where()
df['result'] = np.where(df['average_marks'] >= 40, 'Pass', 'Fail')


# Replace invalid marks using Series.where()
df['math_marks'] = df['math_marks'].where(
    df['math_marks'].between(0, 100),
    other=np.nan
)

# Explode a column
df = df.explode('subjects').reset_index(drop=True)

# Check updated dataset
df.head()
```

## 10. Statistical Analysis

### Basic Statistics

```python
# Mean
df[['math_marks', 'physics_marks', 'chemistry_marks']].mean()

# Median
df[['math_marks', 'physics_marks', 'chemistry_marks']].median()

# Mode
df[['math_marks', 'physics_marks', 'chemistry_marks']].mode()

# Minimum
df[['math_marks', 'physics_marks', 'chemistry_marks']].min()

# Maximum
df[['math_marks', 'physics_marks', 'chemistry_marks']].max()

# Standard deviation
df[['math_marks', 'physics_marks', 'chemistry_marks']].std()

# Variance
df[['math_marks', 'physics_marks', 'chemistry_marks']].var()
```

### Value Counts

`value_counts()` counts the occurrences of each unique value in a Series. By default, it sorts the result by frequency in descending order.

```python
# Count department
df['department'].value_counts()

# Get unique department names (index)
df['department'].value_counts().index

# Sort department names alphabetically
df['department'].value_counts().index.sort_values()

# Get department counts (values)
df['department'].value_counts().values

# Sort counts in ascending order
df['department'].value_counts().sort_values()

# Sort counts in descending order
df['department'].value_counts().sort_values(ascending=False)

# Sort by department name (index)
df['department'].value_counts().sort_index()

# Count including null values
df['department'].value_counts(dropna=False)

# Convert result into DataFrame
df['department'].value_counts().reset_index()
```

**Important:**
- `.index` returns the unique department names.
- `.values` returns the corresponding frequency counts.
- `.sort_values()` sorts by values (frequency).
- `.sort_index()` sorts by index (department names).
- `dropna=False` includes missing values in the count.
- `reset_index()` converts the result into a DataFrame.`

### GroupBy Analysis

```python
# Average marks by gender
df.groupby('gender')[['math_marks', 'physics_marks', 'chemistry_marks']].mean()

# Average marks by department
df.groupby('department')[['math_marks', 'physics_marks', 'chemistry_marks']].mean()

# Average marks by city
df.groupby('city')['average_marks'].mean()

# Department-wise student count
df.groupby('department')['student_id'].count()

# Multiple aggregate functions
df.groupby('department')['average_marks'].agg(['mean', 'max', 'min', 'count'])
```

### Correlation Analysis

```python
df[['math_marks', 'physics_marks', 'chemistry_marks', 'average_marks']].corr()
```

## 11. Data Visualization

### Histogram – Marks Distribution

```python
plt.figure(figsize=(8, 5))

plt.hist(df['math_marks'], bins=10)

plt.title('Math Marks Distribution')
plt.xlabel('Marks')
plt.ylabel('Number of Students')

plt.show()
```

### Boxplot – Detect Outliers

```python
plt.figure(figsize=(8, 5))

sns.boxplot(data=df[['math_marks', 'physics_marks', 'chemistry_marks']])

plt.title('Marks Distribution and Outliers')

plt.show()
```

### Countplot – Gender Distribution

```python
plt.figure(figsize=(6, 4))

sns.countplot(data=df, x='gender')

plt.title('Gender Distribution')

plt.show()
```

### Barplot – Department-wise Average Marks

```python
plt.figure(figsize=(8, 5))

sns.barplot(data=df, x='department', y='average_marks')

plt.title('Department-wise Average Marks')
plt.xlabel('Department')
plt.ylabel('Average Marks')

plt.xticks(rotation=45)

plt.show()
```

### Scatterplot – Math vs Physics

```python
plt.figure(figsize=(7, 5))

sns.scatterplot(data=df, x='math_marks', y='physics_marks', hue='gender')

plt.title('Math vs Physics Marks')

plt.show()
```

### Heatmap – Correlation

```python
plt.figure(figsize=(8, 5))

sns.heatmap(
    df[['math_marks', 'physics_marks', 'chemistry_marks', 'average_marks']].corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title('Correlation Between Marks')

plt.show()
```

### Pie Chart – Pass and Fail

```python
df['result'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    figsize=(6, 6)
)

plt.title('Pass vs Fail Distribution')

plt.ylabel('')

plt.show()
```

## 12. Final Data Check

```python
# Check dataset information
df.info()

# Check missing values
df.isnull().sum()

# Check duplicate rows
df.duplicated().sum()

# Check dataset shape
df.shape

# View final dataset
df.head()
```

## 13. Save Cleaned Dataset

```python
df.to_csv('cleaned_student_data.csv', index=False)
```

## 14. Key Findings

After completing EDA, write your observations:

1. Which department has the highest average marks?
2. Which subject has the highest average marks?
3. Which subject has the lowest average marks?
4. Is there any relationship between Math and Physics marks?
5. Which gender has a higher average score?
6. What is the pass percentage?
7. Are there any outliers in the marks?
8. Which city has the highest number of students?

## 15. Final EDA Checklist

- [ ] Dataset loaded successfully
- [ ] Dataset structure understood
- [ ] Data types corrected
- [ ] Duplicate records handled
- [ ] Missing values handled
- [ ] Inconsistent values cleaned
- [ ] Invalid values checked
- [ ] New features created
- [ ] Statistical analysis completed
- [ ] GroupBy analysis completed
- [ ] Correlation checked
- [ ] Visualizations created
- [ ] Key findings written
- [ ] Cleaned dataset saved

---

