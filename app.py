import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Credit Card Defaulter Prediction App
This app predicts whether an credit card user is going to default in the next month
""")

def user_input_features():
    EDUCATION = st.sidebar.slider('EDUCATION',1,4,2)
    MARRIAGE = st.sidebar.slider('MARRIAGE',1,2,1)
    AGE = st.sidebar.slider('AGE',10,60,25)
    LIMIT_BAL = st.sidebar.slider('LIMIT_BAL',100,1000000,50000)
    PAY_1 = st.sidebar.slider('PAY_1',0,100000,5000)
    BILL_AMT1 = st.sidebar.slider('BILL_AMT1',0,100000,5000)
    BILL_AMT2 = st.sidebar.slider('BILL_AMT2',0,100000,10000)
    BILL_AMT3 = st.sidebar.slider('BILL_AMT3',0,100000,5000)
    BILL_AMT4 = st.sidebar.slider('BILL_AMT4',0,100000,3000)
    BILL_AMT5 = st.sidebar.slider('BILL_AMT5',0,100000,7000)
    BILL_AMT6 = st.sidebar.slider('BILL_AMT6',0,100000,5000)
    PAY_AMT1 = st.sidebar.slider('PAY_AMT1',0,100000,5000)
    PAY_AMT2 = st.sidebar.slider('PAY_AMT2',0,100000,8000)
    PAY_AMT3 = st.sidebar.slider('PAY_AMT3',0,100000,2000)
    PAY_AMT4 = st.sidebar.slider('PAY_AMT4',0,100000,3000)
    PAY_AMT5 = st.sidebar.slider('PAY_AMT5',0,100000,7000)
    PAY_AMT6 = st.sidebar.slider('PAY_AMT6',0,100000,5000)
    data = {'LIMIT_BAL': LIMIT_BAL,'EDUCATION':EDUCATION,'MARRIAGE':MARRIAGE,'AGE':AGE,'PAY_1':PAY_1,
            'BILL_AMT1':BILL_AMT1,'BILL_AMT2':BILL_AMT2,'BILL_AMT3':BILL_AMT3, 'BILL_AMT4':BILL_AMT4, 'BILL_AMT5':BILL_AMT5, 'BILL_AMT6':BILL_AMT6,
            'PAY_AMT1':PAY_AMT1, 'PAY_AMT2':PAY_AMT2, 'PAY_AMT3':PAY_AMT3, 'PAY_AMT4':PAY_AMT4, 'PAY_AMT5':PAY_AMT5, 'PAY_AMT6':PAY_AMT6}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

credit_raw = pd.read_csv('cleaned_data.csv')
credit = credit_raw.drop(columns=['default payment next month'])
df = pd.concat([input_df,credit],axis=0)

features_response = df.columns.tolist()
items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university','default payment next month']
features_response = [item for item in features_response if item not in items_to_remove]
df = pd.concat([input_df,df[features_response]],axis=0)

# Encoding of ordinal features
#encode = ['EDUCATION','MARRIAGE']

#for col in encode:
#    dummy = pd.get_dummies(df[col], prefix=col)
#    df = pd.concat([df,dummy], axis=1)
#    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Reads in saved classification model
load_clf = pickle.load(open('credit_card_clf2.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
defaulter_condition = np.array(['Not default payment next month','Default payment next month'])
st.write(defaulter_condition[prediction])


st.subheader('Prediction Probability')
st.write(prediction_proba)
