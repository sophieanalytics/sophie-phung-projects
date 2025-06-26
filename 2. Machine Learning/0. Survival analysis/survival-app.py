import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
from lifelines import CoxPHFitter 
import joblib
import helpers
import numpy as np 


# UI header
st.set_page_config(page_title = 'Customer survival prediction', layout = 'wide')
st.title('[DEMO] Customer survival prediction')
st.markdown('Real-time churn risk analysis and survival predictions')

# Load the pre-trained model 
cph = joblib.load('data/cox_model.pkl')
data = pd.read_csv('data/Churn_Modelling.csv')

# transform data
df, needed_df = helpers.feature_engineering(data)

# calculated data
cal_data = helpers.predict_customers(needed_df, df, cph, df.columns)

cal_data = cal_data.sort_values(by='risk_score', ascending = False)
cal_data = cal_data.applymap(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)



# Summary 


def create_kpi_cards(df):
   # Calculate metrics from dataframe
   high_risk = len(df[df['risk_type'] == 'high'])
   medium_risk = len(df[df['risk_type'] == 'medium'])
   low_risk = len(df[df['risk_type'] == 'low'])
   total_sellers = len(df)
   
   # Create 4 columns for KPI cards
   col1, col2, col3, col4, space = st.columns([1,1,1,1,3])
   
   # KPI data with colors
   kpis = [
       (col1, str(high_risk), "HIGH RISK", "#dc3545"),      # Red
       (col2, str(medium_risk), "MEDIUM RISK", "#fd7e14"),   # Orange
       (col3, str(low_risk), "LOW RISK", "#28a745"),         # Green  
       (col4, str(total_sellers), "TOTAL SELLERS", "#6f42c1") # Purple
   ]
   
   # Create each KPI card
   for col, value, label, color in kpis:
       with col:
           st.markdown(f"""
           <div style="
               background-color: #f8f9fa;
               border: 1px solid #e9ecef;
               border-radius: 12px;
               padding: 24px;
               text-align: center;
               margin: 8px 0;
               box-shadow: 0 2px 4px rgba(0,0,0,0.1);
           ">
               <div style="
                   color: {color};
                   font-size: 48px;
                   font-weight: bold;
                   margin-bottom: 8px;
               ">{value}</div>
               <div style="
                   color: #6c757d;
                   font-size: 14px;
                   font-weight: 500;
                   text-transform: uppercase;
                   letter-spacing: 0.5px;
               ">{label}</div>
           </div>
           """, unsafe_allow_html=True)


# Display KPI cards
create_kpi_cards(cal_data)



##### FILTER

cols = st.columns([3,2,2])
with cols[0]:
    search = st.text_input("Search customers by name")
with cols[1]:
    risk_filter = st.selectbox("Risk Level", ["All", "HIGH", "MEDIUM", "LOW"])
with cols[2]:
    period = st.selectbox("Prediction Horizon", ["Next 30 Days", "Next 60 Days", "Next 90 Days"])
    
# Filter data (sample logic)
filtered_data = cal_data.copy()
if risk_filter != "All":
    filtered_data = filtered_data[filtered_data['risk_type'].str.contains(risk_filter)]

if search:
    filtered_data = filtered_data[filtered_data['customer_name'].str.contains(search, case=False)]
    


# ------------------
# Table details
# ------------------

def create_customer_card_from_df(row):
   # Header section
   col1, col2 = st.columns([3,1])
   
   with col1:
       st.markdown(f"<p style='color: #cacaca; margin-bottom: 2px;'> Customer Name </p>", unsafe_allow_html=True)
       st.markdown(f"<h3 style='margin-bottom: 5px;'> {row['customer_name']}</h3>", unsafe_allow_html=True)
    #    st.markdown(f"<p style='color: #6c757d; font-size: 14px;'>{row['ID']} • <a href='mailto:{row['Email']}' style='color: #007bff;'>{row['Email']}</a></p>", unsafe_allow_html=True)
   
   with col2:
       if row['risk_type'] == "high":
           st.markdown(
               '<div style="background-color: #ffa0a0; color: #dc3545; padding: 6px 12px; border-radius: 20px; font-weight: bold; text-align: center; font-size: 16px;">HIGH RISK</div>', 
               unsafe_allow_html=True
           )
       elif row['risk_type'] == "medium":
           st.markdown(
               '<div style="background-color: #ffd1a0; color: #f0a924; padding: 6px 12px; border-radius: 20px; font-weight: bold; text-align: center; font-size: 16px;">MEDIUM</div>', 
               unsafe_allow_html=True
           )
       elif row['risk_type'] == "low":
           st.markdown(
               '<div style="background-color: #a0ffa9; color: #ffebee; padding: 6px 12px; border-radius: 20px; font-weight: bold; text-align: center; font-size: 16px;">LOW</div>', 
               unsafe_allow_html=True
           )
   
   # Metrics section with boxes
   col1, col2, col3, col4, col5, space = st.columns([1,1,1,1,1,3])

   
   metrics = [
       ("Churn Risk Score", row['risk_score']),
       ("Median Predicted Lifespan (yrs)", row['median_survival_time']),
       ("Prob. of Surviving 5 Years", row['live_5years']),
       ("Yrs When 70% Likely to Churn", row['churn_70%']),
       ("Estimated Salary (USD)", row['salary'])
   ]
   
   for col, (label, value) in zip([col1, col2, col3, col4, col5], metrics):
       with col:
           st.markdown(f"""
           <div style="
               background-color: #f8f9fa;
               border: 1px solid #e9ecef;
               border-radius: 8px;
               padding: 20px;
               text-align: center;
               margin: 0 5px;
               height: 120px;
               display: flex;
               flex-direction: column;
               justify-content: center;
               align-items: center;
           ">
               <div style="
                   color: #6c757d;
                   font-size: 14px;
                   margin-bottom: 12px;
                   font-weight: 500;
                   line-height: 1.3;
                   text-align: center;
               ">{label}</div>
               <div style="
                   color: #212529;
                   font-size: 30px;
                   font-weight: bold;
               ">{value}</div>
           </div>
           """, unsafe_allow_html=True)

# Display cards
st.header("Customers Ranked by Churn Risk")

for index, row in filtered_data.head().iterrows():
   create_customer_card_from_df(row)
   if index < len(df) - 1:  # Don't add divider after last row
       st.divider()