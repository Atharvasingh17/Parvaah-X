import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_parvaah_x_data(num_projects=2000):
    np.random.seed(42)
    random.seed(42)

    sectors = ['Road Transport & Highways', 'Railways', 'Power', 'Petroleum', 'Coal', 'Water Resources', 'Urban Development', 'Telecommunications', 'Civil Aviation']
    agencies = ['NHAI', 'RVNL', 'NTPC', 'ONGC', 'CIL', 'NHPC', 'AAI', 'PWD']

    data = []

    for i in range(num_projects):
        project_id = f"PRJ-{i+1:04d}"
        sector = random.choice(sectors)
        agency = random.choice(agencies)
        
        # Base metrics
        original_cost = round(random.uniform(150.0, 5000.0), 2) # Projects > 150 cr
        
        # Let's create some realistic relationships for cost
        # Does it have a cost overrun?
        has_cost_overrun = random.random() < 0.35 # 35% chance
        if has_cost_overrun:
            revised_cost = round(original_cost * random.uniform(1.05, 1.6), 2)
        else:
            revised_cost = original_cost
            
        # Progress and Expenditure
        physical_progress = round(random.uniform(5.0, 95.0), 2)
        
        # Expenditure is usually somewhat correlated with progress, but can diverge if there's a risk
        expenditure_factor = random.uniform(0.8, 1.5)
        cumulative_expenditure = round(min(revised_cost * (physical_progress / 100) * expenditure_factor, revised_cost), 2)

        # Dates
        start_year = random.randint(2015, 2023)
        start_month = random.randint(1, 12)
        start_date = datetime(start_year, start_month, 1)
        
        planned_duration_months = random.randint(24, 72)
        planned_completion_date = start_date + timedelta(days=planned_duration_months * 30.44)

        # Does it have a time delay?
        has_time_delay = random.random() < 0.45 # 45% chance
        if has_time_delay:
            delay_months = random.randint(3, 48)
            current_expected_completion = planned_completion_date + timedelta(days=delay_months * 30.44)
        else:
            current_expected_completion = planned_completion_date

        # Milestones
        milestones_total = random.randint(5, 15)
        milestones_completed = int(milestones_total * (physical_progress / 100))
        # Delayed milestones are correlated with time delay
        if has_time_delay:
            milestones_delayed = random.randint(1, milestones_completed) if milestones_completed > 0 else 0
        else:
            milestones_delayed = 0

        # Create target variables for ML models (these are what we try to predict based on current snapshot)
        # We will assume this snapshot is 'current' but we know the 'future' for training
        
        # Target: Will it have Cost Overrun in the future?
        # We define a synthetic target: if current revised > original OR (expenditure > progress * original AND progress < 80)
        target_cost_overrun = 1 if (revised_cost > original_cost) or ((cumulative_expenditure / original_cost) > (physical_progress / 100) + 0.1) else 0
        
        # Target: Will it have Time Overrun in the future?
        target_time_overrun = 1 if (current_expected_completion > planned_completion_date) or (milestones_delayed > 0) else 0

        data.append({
            'Project_ID': project_id,
            'Project_Name': f"{sector} Project {i+1}",
            'Sector': sector,
            'Implementing_Agency': agency,
            'Original_Approved_Cost_Cr': original_cost,
            'Revised_Cost_Cr': revised_cost,
            'Cumulative_Expenditure_Cr': cumulative_expenditure,
            'Start_Date': start_date.strftime('%Y-%m-%d'),
            'Planned_Completion_Date': planned_completion_date.strftime('%Y-%m-%d'),
            'Current_Completion_Date': current_expected_completion.strftime('%Y-%m-%d'),
            'Physical_Progress_Pct': physical_progress,
            'Milestones_Total': milestones_total,
            'Milestones_Completed': milestones_completed,
            'Milestones_Delayed': milestones_delayed,
            'Target_Cost_Overrun': target_cost_overrun,
            'Target_Time_Overrun': target_time_overrun
        })

    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_parvaah_x_data.csv', index=False)
    print(f"Generated {len(df)} synthetic records in data/synthetic_parvaah_x_data.csv")

if __name__ == "__main__":
    generate_parvaah_x_data()
