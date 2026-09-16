import pandas as pd
import random
from datetime import datetime, timedelta
import os

def generate_realistic_data():
    real_projects = [
        {"Name": "Mumbai-Ahmedabad High Speed Rail", "Sector": "Railways", "Cost": 108000, "Agency": "NHSRCL"},
        {"Name": "Navi Mumbai International Airport", "Sector": "Civil Aviation", "Cost": 16700, "Agency": "CIDCO"},
        {"Name": "Chenab Railway Bridge", "Sector": "Railways", "Cost": 1400, "Agency": "Konkan Railway"},
        {"Name": "Zojila Tunnel", "Sector": "Road Transport & Highways", "Cost": 6800, "Agency": "NHIDCL"},
        {"Name": "Central Vista Redevelopment", "Sector": "Urban Development", "Cost": 20000, "Agency": "CPWD"},
        {"Name": "Noida International Airport (Jewar)", "Sector": "Civil Aviation", "Cost": 29560, "Agency": "YIAPL"},
        {"Name": "Mumbai Trans Harbour Link (MTHL)", "Sector": "Road Transport & Highways", "Cost": 17843, "Agency": "MMRDA"},
        {"Name": "Polavaram Irrigation Project", "Sector": "Water Resources", "Cost": 55548, "Agency": "PPA"},
        {"Name": "Ken-Betwa River Interlinking", "Sector": "Water Resources", "Cost": 44605, "Agency": "NWDA"},
        {"Name": "Delhi-Mumbai Expressway", "Sector": "Road Transport & Highways", "Cost": 100000, "Agency": "NHAI"},
        {"Name": "Bharatmala Pariyojana Phase-I", "Sector": "Road Transport & Highways", "Cost": 535000, "Agency": "NHAI"},
        {"Name": "Ganga Expressway", "Sector": "Road Transport & Highways", "Cost": 36230, "Agency": "UPEIDA"},
        {"Name": "Pune Metro Rail Project", "Sector": "Urban Transport", "Cost": 11420, "Agency": "Maha-Metro"},
        {"Name": "Bangalore Suburban Rail Project", "Sector": "Urban Transport", "Cost": 15767, "Agency": "K-RIDE"},
        {"Name": "Subansiri Lower Hydroelectric Project", "Sector": "Power", "Cost": 20000, "Agency": "NHPC"},
        {"Name": "Kudankulam Nuclear Power Plant", "Sector": "Power", "Cost": 39849, "Agency": "NPCIL"},
        {"Name": "New Domestic Terminal, Patna", "Sector": "Civil Aviation", "Cost": 1217, "Agency": "AAI"}
    ]
    
    extended_projects = []
    for i in range(1500):
        p = random.choice(real_projects)
        proj = {}
        proj['Project_Name'] = f"{p['Name']} - Phase {random.randint(1,5)}" if i > len(real_projects) else p['Name']
        proj['Sector'] = p['Sector']
        proj['Implementing_Agency'] = p['Agency']
        
        original_cost = p['Cost'] * random.uniform(0.5, 1.5)
        proj['Original_Approved_Cost_Cr'] = round(original_cost, 2)
        
        climate_risk = random.random()
        geo_risk = random.random()
        supply_risk = random.random()
        
        proj['Climate_Issue_Severity'] = round(climate_risk, 2)
        proj['Geopolitical_War_Impact'] = round(geo_risk, 2)
        proj['Supply_Chain_Disruption'] = round(supply_risk, 2)
        
        has_cost_overrun = (climate_risk + geo_risk + supply_risk) > 1.2
        if has_cost_overrun:
            revised_cost = round(original_cost * random.uniform(1.1, 1.8), 2)
        else:
            revised_cost = original_cost
            
        proj['Revised_Cost_Cr'] = revised_cost
        
        physical_progress = round(random.uniform(5.0, 95.0), 2)
        expenditure_factor = random.uniform(0.8, 1.5)
        cumulative_expenditure = round(min(revised_cost * (physical_progress / 100) * expenditure_factor, revised_cost), 2)
        
        proj['Physical_Progress_Pct'] = physical_progress
        proj['Cumulative_Expenditure_Cr'] = cumulative_expenditure
        
        start_date = datetime(2018, 1, 1) + timedelta(days=random.randint(0, 1500))
        planned_duration_months = random.randint(24, 84)
        planned_completion = start_date + timedelta(days=planned_duration_months * 30)
        
        has_time_overrun = (climate_risk + geo_risk + supply_risk) > 1.0
        if has_time_overrun:
            delay_months = int((climate_risk * 12) + (geo_risk * 6) + (supply_risk * 18)) + 1
            current_expected = planned_completion + timedelta(days=delay_months * 30)
        else:
            current_expected = planned_completion
            proj['Climate_Issue_Severity'] = 0.0
            proj['Geopolitical_War_Impact'] = 0.0
            proj['Supply_Chain_Disruption'] = 0.0
            
        proj['Project_ID'] = f"PRJ-{i+1:04d}"
        proj['Start_Date'] = start_date.strftime('%Y-%m-%d')
        proj['Planned_Completion_Date'] = planned_completion.strftime('%Y-%m-%d')
        proj['Current_Completion_Date'] = current_expected.strftime('%Y-%m-%d')
        
        proj['Target_Cost_Overrun'] = 1 if revised_cost > original_cost else 0
        proj['Target_Time_Overrun'] = 1 if current_expected > planned_completion else 0
        
        extended_projects.append(proj)
        
    os.makedirs('data', exist_ok=True)
    df_out = pd.DataFrame(extended_projects)
    df_out.to_csv('data/real_augmented_parvaah_x_data.csv', index=False)
    print(f"Generated {len(df_out)} realistic projects into data/real_augmented_parvaah_x_data.csv.")

if __name__ == "__main__":
    generate_realistic_data()
