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
#The analysis 


## License

This project is for educational purposes.