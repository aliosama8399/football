"""
Apache Airflow DAG: Football Match Ingestion & Continuous Evaluation
====================================================================
Runs match scraping, preprocessing, and triggers continuous RAGAS and Match Prediction evaluation.
Can be triggered manually or run on a daily schedule.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "football-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "football_scrape_and_continuous_eval",
    default_args=default_args,
    description="Automated match scrape and evaluation trigger pipeline",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["football", "mlops", "eval", "scraping"],
) as dag:

    scrape_matches = BashOperator(
        task_id="scrape_matches_and_preprocess",
        bash_command="python /opt/airflow/data/auto_scrape_and_eval.py --now",
    )

    scrape_matches
