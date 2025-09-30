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
# **Explotary Data Analysis**
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

# The Analysis

## 02_count_of_skills.ipynb — Analysis of Skill Counts in Job Postings

This notebook analyzes the frequency of different skills mentioned in data job postings for India, and visualizes the top skills for the most common job titles.

---

### 1. Library Imports & Data Loading

```python
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import ast
import seaborn as sns
from datasets import load_dataset

dataset = load_dataset('lukebarousse/data_jobs')
df = dataset['train'].to_pandas()
df['job_posted_date'] = pd.to_datetime(df['job_posted_date'])
df['job_skills'] = df['job_skills'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
```

---

### 2. Filtering and Exploding Skills

```python
df_ind = df[df['job_country'] == 'India']  # Filter for India
df_explode = df_ind.explode('job_skills')  # Each skill gets its own row
df_explode[['job_title_short', 'job_skills']]
```

---

### 3. Grouping and Counting Skills

```python
df_group = df_explode.groupby(['job_skills','job_title_short']).size()  # Group by skill and job title
df_group = df_group.reset_index(name='skill_count')  # Name the new column
df_group.sort_values(by='skill_count', ascending=False, inplace=True)  # Sort by skill count
df_group
```

---

### 4. Selecting Top Job Titles

```python
df_titles = df_group['job_title_short'].unique().tolist()  # Unique job titles
df_titles = sorted(df_titles[:3])  # Take top 3
df_titles
```

---

### 5. Visualizing Top Skills for Each Job Title

```python
fig, ax = plt.subplots(len(df_titles), 1, figsize=(8, 5))

for x, job_title in enumerate(df_titles):
    df_plot = df_group[df_group['job_title_short'] == job_title].head(5)  # Top 5 skills for this title
    sns.barplot(data=df_plot, x='skill_count', y='job_skills', ax=ax[x], hue='job_skills', palette='dark:b')
    ax[x].set_title(job_title)
    ax[x].set_xlabel('Job Counts')
    ax[x].set_ylabel(' ')
    ax[x].set_xlim(0, 14000)
    for n, v in enumerate(df_plot['skill_count']):
        ax[x].text(v+1, n, str(v), va='center')
    if x != 2:
        ax[x].set_xticks([])
    if x != 2:
        ax[x].set_xlabel('')

fig.suptitle('Count of Job Skills According to Data Jobs')
fig.tight_layout(h_pad=0.5)
sns.despine()
plt.show()
```

---

This notebook provides insight into which skills are most frequently required for the top data job titles in India, helping job seekers and employers understand the current demand in the market.

## 03_skill_demand.ipynb — Skill Demand Analysis

This notebook examines the demand for various skills in Data Analyst roles in India over time, showing how the popularity of top skills changes by month.

---

### 1. Library Imports & Data Loading

```python
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import ast
import seaborn as sns
from datasets import load_dataset

dataset = load_dataset('lukebarousse/data_jobs')
df = dataset['train'].to_pandas()
df['job_posted_date'] = pd.to_datetime(df['job_posted_date'])
df['job_skills'] = df['job_skills'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
```

---

### 2. Filtering for Data Analyst Roles in India

```python
df_da = df[(df['job_title_short'] == 'Data Analyst') & (df['job_country'] == 'India') ].copy()
df_da['month_no'] = df_da['job_posted_date'].dt.month
```

---

### 3. Exploding Skills and Creating Pivot Table

```python
df_da_skills = df_da.explode('job_skills')
df_pivot = df_da_skills.pivot_table(index="month_no", columns='job_skills', aggfunc='size', fill_value=0)
df_pivot.loc['Total'] = df_pivot.sum()
df_pivot_sort = df_pivot[df_pivot.loc['Total'].sort_values(ascending=False).index]
df_pivot_sort = df_pivot_sort.drop('Total')
df_pivot_sort
```

---

### 4. Calculating Skill Demand Percentages

```python
da_total = df_da.groupby('month_no').size()
df_percent = df_pivot_sort.div(da_total/100, axis=0)
df_percent = df_percent.reset_index()
df_percent['df_month_name'] = df_percent['month_no'].apply(lambda x :pd.to_datetime(x, format='%m').strftime('%b') )
df_percent = df_percent.set_index('df_month_name')
df_percent = df_percent.drop(columns='month_no')
df_percent = df_percent.iloc[: ,:5]  # Select top 5 skills
df_percent
```

---

### 5. Visualizing Skill Trends Over Months

```python
sns.lineplot(data=df_percent, dashes=False, legend=False)
sns.despine()
plt.xlabel('Months')
plt.ylabel('Likly hood of job posting')
plt.title('Trending job skills for Data Analysis in India')
from matplotlib.ticker import PercentFormatter
ax = plt.gca()
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
for x in range(5):
    plt.text(11.2, df_percent.iloc[-1,x], df_percent.columns[x])
plt.show()
```

---

This notebook provides insight into how the demand for top skills in Data Analyst roles in India changes throughout the year, helping job seekers and employers track trends in the data job market.

## 04_salary_trends.ipynb — Salary Trends Over Time

This notebook investigates salary trends for the most common data job titles in India, visualizing the distribution of annual salaries for each role.

---

### 1. Library Imports & Data Loading

```python
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import ast
import seaborn as sns
from datasets import load_dataset

dataset = load_dataset('lukebarousse/data_jobs')
df = dataset['train'].to_pandas()
df['job_posted_date'] = pd.to_datetime(df['job_posted_date'])
df['job_skills'] = df['job_skills'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
```

---

### 2. Filtering for Indian Jobs with Salary Data

```python
df_filter = df[df['job_country'] == 'India'].dropna(subset='salary_year_avg')
```

---

### 3. Selecting Top 6 Job Titles

```python
df_titles = df_filter['job_title_short'].value_counts()
df_titles = df_titles.sort_values(ascending=False).index[:6].to_list()
df_top6 = df_filter[df_filter['job_title_short'].isin(df_titles)]
```

---

### 4. Ordering Job Titles by Median Salary

```python
df_order = df_top6.groupby('job_title_short')['salary_year_avg'].median().sort_values(ascending=False).index
```

---

### 5. Visualizing Salary Distributions

```python
sns.set_theme(style='ticks')
sns.boxplot(data=df_top6, x="salary_year_avg", y='job_title_short', order=df_order)
plt.ylabel('')
plt.xlabel('Salary per Year')
plt.title('Salary distribution in India')
ticks_x = plt.FuncFormatter(lambda x, pos: f'${int(x/1000)}k')
plt.gca().xaxis.set_major_formatter(ticks_x)
plt.xlim(0, 300000)
sns.despine()
plt.show()
```

---

This notebook provides a clear comparison of salary distributions for the most common data job titles in India, helping users understand which roles offer higher pay and how salaries vary within each role.


## 05_skills_vs_salary.ipynb — Relationship Between Skills and Salary

This notebook explores the relationship between specific skills and salary levels for Data Analyst roles in India, identifying both the highest-paid and most in-demand skills.

---

### 1. Library Imports & Data Loading

```python
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import ast
import seaborn as sns
from datasets import load_dataset

dataset = load_dataset('lukebarousse/data_jobs')
df = dataset['train'].to_pandas()
df['job_posted_date'] = pd.to_datetime(df['job_posted_date'])
df['job_skills'] = df['job_skills'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
```

---

### 2. Filtering for Data Analyst Roles in India

```python
df_filter = df[(df['job_title_short'] == 'Data Analyst') & (df['job_country'] == 'India')].copy()
df_filter = df_filter.dropna(subset='salary_year_avg')
```

---

### 3. Exploding Skills and Grouping

```python
df_skills = df_filter.explode('job_skills')
df_groupby = df_skills.groupby('job_skills')['salary_year_avg'].agg(['count','median'])
```

---

### 4. Highest Paid Skills

```python
df_salary_sort = df_groupby.sort_values(by='median', ascending=False).head(10)
df_salary_sort
```

---

### 5. Most In-Demand Skills

```python
df_top_skills = df_skills.groupby('job_skills')['salary_year_avg'].agg(['count', 'median'])
df_top_skills = df_top_skills.sort_values(by='count', ascending=False).head(10)
df_top_skills = df_top_skills.sort_values(by='median', ascending=False)
```

---

### 6. Visualizing Highest Paid and Most In-Demand Skills

```python
fig, ax = plt.subplots(2,1)
sns.set_theme(style='ticks')

sns.barplot(data=df_salary_sort, x='median', y=df_salary_sort.index, ax=ax[0], hue='median', palette='dark:b_r', legend=False)
ax[0].set_title('Highest paid skills ')
ax[0].set_xlabel(' ')
ax[0].set_ylabel(' ')
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos:f'${int(x/1000)}k'))

sns.barplot(data=df_top_skills, x='median', y=df_top_skills.index, ax=ax[1], hue='median', palette='dark:b_r', legend=False)
ax[1].set_title('Top demand skills')
ax[1].set_xlabel('Salary')
ax[1].set_ylabel(' ')
ax[1].set_xlim(0,160000)
ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos:f'${int(x/1000)}k'))

fig.tight_layout()
sns.despine()
plt.show()
```

---

This notebook helps identify which skills are associated with higher salaries and which are most frequently required for Data Analyst positions in India, providing valuable insights for both job seekers and employers.

## 06_optimal_skill.ipynb — Identifying Optimal Skills for Higher Salaries

This notebook identifies the most optimal skills for maximizing salary potential for Data Analyst roles in India by analyzing both skill demand and median salary.

---

### 1. Library Imports & Data Loading

```python
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
import ast
import seaborn as sns
from datasets import load_dataset
from adjustText import adjust_text

dataset = load_dataset('lukebarousse/data_jobs')
df = dataset['train'].to_pandas()
df['job_posted_date'] = pd.to_datetime(df['job_posted_date'])
df['job_skills'] = df['job_skills'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
```

---

### 2. Filtering for Data Analyst Roles in India

```python
df_filter = df[(df['job_title_short'] == 'Data Analyst') & (df['job_country'] == 'India')].dropna(subset='salary_year_avg')
```

---

### 3. Exploding Skills and Grouping

```python
df_skills = df_filter.explode('job_skills')
df_group = df_skills.groupby('job_skills')['salary_year_avg'].agg(['count','median']).sort_values(by='count', ascending=False)
df_group = df_group.rename(columns={'count': 'skill_count','median': 'median_salary' })
df_da_count = len(df_filter)
df_group['skill_percent'] = df_group['skill_count']/df_da_count*100
df_group
```

---

### 4. Filtering for In-Demand Skills

```python
skill_percent = 11.7
skill_demand = df_group[df_group['skill_percent'] > skill_percent]
skill_demand
```

---

### 5. Visualizing Optimal Skills (Demand vs. Salary)

```python
skill_demand.plot(kind='scatter', x='skill_percent', y='median_salary')
texts = []
for x, txt in enumerate(skill_demand.index):
    texts.append(plt.text(skill_demand['skill_percent'].iloc[x], skill_demand['median_salary'].iloc[x], txt))

adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray',))
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'${x/1000}k'))
ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))

plt.title('Most optimal skills for Data Analyst in India')
plt.ylabel('Median Salary')
plt.xlabel('Percentage of skills Demand')
plt.tight_layout()
sns.despine()
plt.show()
```

---

This notebook helps identify which skills are both highly demanded and associated with higher salaries for Data Analyst positions in India, providing actionable insights for job seekers aiming to maximize their earning potential.
## License

This project is for educational purposes.