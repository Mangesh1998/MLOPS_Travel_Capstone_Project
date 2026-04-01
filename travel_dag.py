from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd

def load_data():
    flights = pd.read_csv('flights.csv')
    users = pd.read_csv('users.csv')
    hotels = pd.read_csv('hotels.csv')
    # Process data
    print("Data loaded")

def train_models():
    # Run training scripts
    import subprocess
    subprocess.run(['python', 'train_flight_model.py'])
    subprocess.run(['python', 'train_gender_model.py'])
    print("Models trained")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'travel_ml_pipeline',
    default_args=default_args,
    description='ML pipeline for travel data',
    schedule_interval=timedelta(days=1),
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag,
)

load_task >> train_task