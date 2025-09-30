# py_project_jobs

A data analysis project exploring trends in data-related job postings, salaries, and required skills using Python, pandas, and visualization libraries.

## Project Structure

```
my_project/
    01_EDA.ipynb                # Exploratory Data Analysis
    02_count_of_skills.ipynb    # Analysis of skill counts in job postings
    03_skill_demand.ipynb       # Skill demand analysis
    04_salary_trends.ipynb      # Salary trends over time
    05_skills_vs_salary.ipynb   # Relationship between skills and salary
    06_optimal_skill.ipynb      # Identifying optimal skills for higher salaries
visualizations/                 # Additional plots and figures
3_pandas_pivot_tables.ipynb     # Pivot table analysis of salaries by country and job title
README.md
```

## Dataset

- Uses the [lukebarousse/data_jobs](https://huggingface.co/datasets/lukebarousse/data_jobs) dataset from Hugging Face Datasets.

## Main Features

- **Exploratory Data Analysis (EDA):** Initial exploration and cleaning of the dataset.
- **Skill Analysis:** Counts and analyzes the most in-demand skills.
- **Salary Analysis:** Examines salary trends by country, job title, and skill.
- **Pivot Table Analysis:** Uses pandas pivot tables to compare median salaries across countries and job titles.
- **Visualization:** Uses matplotlib and seaborn for data visualization.

## Getting Started

1. Clone the repository.
2. Install dependencies:
    ```bash
    pip install pandas matplotlib seaborn datasets
    ```
3. Open the notebooks in Jupyter or VS Code and run the cells.


Here’s how to create a pivot table of median salaries by country and job title:

```python
import pandas as pd
from datasets import load_dataset

dataset = load_dataset('lukebarousse/data_jobs')
df = dataset['train'].to_pandas()
df['job_posted_date'] = pd.to_datetime(df['job_posted_date'])

job_countries = df['job_country'].value_counts().head(5).index
job_country_salary = df.pivot_table(
    values='salary_year_avg',
    index='job_country',
    columns='job_title_short',
    aggfunc='median'
)
job_country_salary = job_country_salary.loc[job_countries]
job_titles = ['Data Analyst', 'Data Engineer', 'Data Scientist']
job_country_salary = job_country_salary[job_titles]
print(job_country_salary)
```
# **The Analysis **

## 01_EDA.ipynb — Exploratory Data Analysis

This notebook performs an initial exploration of the data jobs dataset, focusing on Data Analyst roles in India and general salary trends.

---

### 1. Library Imports & Data Loading

```python
import pandas as pd
import matplotlib.pyplot as plt
import ast
import seaborn as sns
from datasets import load_dataset
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter

# Loading datasets
dataset = load_dataset('lukebarousse/data_jobs')
df = dataset['train'].to_pandas()

# Cleaning data
df['job_posted_date'] = pd.to_datetime(df['job_posted_date'])
df['job_skills'] = df['job_skills'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
```

---

### 2. Filtering for Data Analyst Roles in India

```python
df_ind_da = df[(df['job_country'] == 'India') & (df['job_title_short'] == 'Data Analyst')]
```

---

### 3. Top Job Locations

```python
df_loc = df_ind_da['job_location'].value_counts().head(10).to_frame()
df_loc
```

```python
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.barplot(data=df_loc, x='count', y='job_location', hue='count', palette="dark:#5A9_r", legend=False)
sns.despine
plt.title('Counts of Job Location For Data Analyst in India')
plt.xlabel('Count of Jobs')
plt.ylabel('Location')
plt.show()
```

---

### 4. Exploring Boolean Columns

```python
dict_colum ={
    'job_work_from_home': 'work_from_home',
    'job_no_degree_mention': 'degree_requirement',
    'job_health_insurance' : 'heath_insurance_provided'           
}

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(10,5)

for i , (column ,title ) in enumerate(dict_colum.items()):
    ax[i].pie(df[column].value_counts(), labels=['False', 'True'], autopct='%1.1f%%', startangle=90)
    ax[i].set_title(title)
```

---

### 5. Top Companies Hiring Data Analysts in India

```python
df_cp = df_ind_da['company_name'].value_counts().head(10).to_frame()
```

```python
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
df_cp = df_ind_da['company_name'].value_counts().head(10).to_frame()
sns.barplot(data=df_cp, x='count', y='company_name', hue='count', palette="dark:#5A9_r", legend=False)
sns.despine
plt.title('Top companies having Data Analyst job postings in India')
plt.xlabel('Count of Jobs')
plt.ylabel('Companies')
plt.show()
```

---

### 6. Highest Paying Data Jobs in India

```python
# filter for india
df_ind = df[df['job_country'] == 'India']

# remove na values from salary year avg column
df_notna = df_ind[df_ind['salary_year_avg'].notna()].copy()
```

```python
df_top_jobs = df_notna.groupby('job_title')['salary_year_avg'].mean().reset_index()
df_top_jobs = df_top_jobs.sort_values(by='salary_year_avg', ascending=False).head(10)
fig, ax = plt.subplots()

sns.barplot(data=df_top_jobs, x='salary_year_avg', y='job_title' , hue='job_title', palette='dark:b')
plt.xlabel('salaries')
plt.ylabel('Jobs')
plt.title('Highest paying  data jobs in India')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'${x/1000:,.0f}K'))
plt.show()
```

---

### 7. Top Paying Data Jobs (Global)

```python
df = df[df['salary_year_avg'].notna()]
df_jobs = df.groupby('job_title_short')['salary_year_avg'].median().sort_values(ascending=False).reset_index()
df_jobs.columns = ['job_title_short', 'salary_year_avg']
```

```python
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(data=df_jobs, x='salary_year_avg', y='job_title_short', ax=ax, hue='job_title_short', palette='dark:b')
plt.title('Top paying data jobs')
plt.ylabel('Jobs')
plt.xlabel('Salaries')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos : f'${x/1000:,.0f}k'))

plt.show
```

---

This notebook provides a comprehensive overview of the dataset, highlights key trends, and sets the stage for deeper analysis in subsequent notebooks.

## License

This project is for educational purposes.