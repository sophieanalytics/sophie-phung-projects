# --------------
# Predicting customers
# --------------


def feature_engineering(df):
    
    import pandas as pd 
    import numpy as np 
    
    num_cols = ['Age', 'Tenure', 'Balance', 'EstimatedSalary']
    cat_cols = ['NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited',  'Gender']
    
    # log transformation
    for col in num_cols:
        df[f'{col}_log'] = np.log(df[col]+1)
        
    # one-hot encoding
    df = pd.get_dummies(df, columns = cat_cols, drop_first=True) 
    
    final_cols = [
        # 'Tenure',
       # 'CreditScore_log',
       'Age_log',
       'Balance_log', 
       # 'EstimatedSalary_log', 
       'NumOfProducts_2',
       'NumOfProducts_3', 'NumOfProducts_4',
       # 'HasCrCard_1', 
       'IsActiveMember_1',
    #    'Exited_1', 
       # 'Geography_Germany', 'Geography_Spain', 
       'Gender_Male']
    
    needed_cols = ['Surname', 'EstimatedSalary', 'Exited_1']
    
    return df[final_cols], df[final_cols + needed_cols]




def predict_customers(needed_df, df, cph, feature_cols):
    
    import pandas as pd 
    import numpy as np 
    
    active_customers = needed_df[needed_df['Exited_1'] == False]
        
    # predict risk score
    risk_scores = cph.predict_partial_hazard(active_customers[feature_cols])

    # median_survival_time
    median_survival = cph.predict_median(active_customers[feature_cols])

    # churn_0year

    surv_funcs = cph.predict_survival_function(active_customers[feature_cols])
    surv_funcs_below1year = surv_funcs.loc[0]

    # churn_90d
    surv_funcs_5years = surv_funcs.loc[5]

    # churn_90%
    churn_70_per = []
    threshold = 0.3
    for i in range(active_customers.shape[0]):
            surv = surv_funcs.iloc[:, i]
            t = surv[surv <= threshold].index.min()
            churn_70_per.append(t if pd.notna(t) else np.nan)


    new_data = pd.DataFrame({
        'customer_name': active_customers['Surname'],
        'risk_score': risk_scores,
        'median_survival_time': median_survival,
        'live_0year': surv_funcs_below1year,
        'live_5years': surv_funcs_5years,
        'churn_70%': churn_70_per,
        'salary': active_customers['EstimatedSalary'],
        'gender': active_customers['Gender_Male'].apply(lambda x: 'Male' if x == 1 else 'Female')
    })  
    
    def define_risks(row):
        if row >= 5:
            return 'high'
        elif row >= 2:
            return 'medium'
        else:
            return 'low'


    new_data['risk_type'] = new_data['risk_score'].apply(define_risks)
    
    return new_data
    
    