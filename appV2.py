
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

url='https://raw.githubusercontent.com/federicoding/Airline_Satisfaction/main/Airline_Dataset.csv'
df = pd.read_csv(filepath_or_buffer=url,
                 sep=';')

@st.cache

df['Customer Type'] = df['Customer Type'].map({'Loyal Customer':'Returning Customer', 'disloyal Customer':'First-time Customer'})

df = df.dropna(axis=0)

df['Departure Delay in Minutes'] = df['Departure Delay in Minutes'].astype('float')

df = df.rename(columns={'Leg room service':'Leg room'})

from string import capwords
df.columns = [capwords(i) for i in df.columns]
df = df.rename(columns={'Departure/arrival Time Convenient':'Departure/Arrival Time Convenience'})

df = df[(df['Inflight Wifi Service']!=0)&(df['Departure/Arrival Time Convenience']!=0)&(df['Ease Of Online Booking']!=0)&(df['Gate Location'])&(df['Food And Drink']!=0)&(df['Online Boarding']!=0)&(df['Seat Comfort']!=0)&(df['Inflight Entertainment']!=0)&(df['On-board Service']!=0)&(df['Leg Room']!=0)&(df['Baggage Handling']!=0)&(df['Checkin Service']!=0)&(df['Inflight Service']!=0)&(df['Cleanliness']!=0)]

df['Satisfaction'] = df['Satisfaction'].map({'satisfied':1,'neutral or dissatisfied':0})
df = df.reset_index()
df = df.drop('index',axis=1)
df['Total Delay'] = df['Departure Delay In Minutes'] + df['Arrival Delay In Minutes']

DF = df.copy()
df = df.drop('Id',axis=1)

df = df.reindex(columns=['Satisfaction']+list(df.columns)[:-2]+['Total Delay'])
df = df.drop(['Departure Delay In Minutes','Arrival Delay In Minutes'],axis=1)

df['Class'] = df['Class'].map({'Eco':'Economy','Eco Plus':'Economy','Business':'Business'})

df1 = pd.get_dummies(df,columns=['Gender','Customer Type','Type Of Travel','Class'],drop_first=True)

df1 = df1.drop(['Total Delay','Flight Distance','Age','Gate Location','Departure/Arrival Time Convenience',
                'Gender_Male'],axis=1)

df1.rename(columns= {'Inflight Wifi Service': 'Inflight_Wifi_Service',
                     'Ease Of Online Booking': 'Ease_Of_Online_Booking',
                     'Food And Drink': 'Food_And_Drink',
                     'Online Boarding': 'Online_Boarding',
                     'Seat Comfort': 'Seat_Comfort',
                     'Inflight Entertainment': 'Inflight_Entertainment',
                     'On-board Service': 'On_board_Service',
                     'Leg Room': 'Leg_Room',
                     'Baggage Handling': 'Baggage_Handling',
                     'Checkin Service': 'Checkin_Service',
                     'Inflight Service': 'Inflight_Service',
                     'Cleanliness': 'Cleanliness',
                     'Customer Type_Returning Customer': 'Customer_Type_Returning_Customer',
                     'Type Of Travel_Personal Travel': 'Type_Of_Travel_Personal_Travel',
                     'Class_Economy': 'Class_Economy'})

y = df1['Satisfaction']
X = df1.drop('Satisfaction',axis=1)

rf = RandomForestClassifier(max_depth=17, random_state=42)
rf.fit(X,y)
#print("Random Forest score: {:.4f}".format(rf.score(X,y)))

st.write("""
# Simple AirLine Satisfaction Prediction App
This app predicts the **Satisfaction** of a customer!
""")


st.sidebar.header('User Imput Parameters')

def user_input_features():
  Inflight_Wifi_Service = st.sidebar.selectbox('Inflight Wifi Service',[1,2,3,4,5])
  Ease_Of_Online_Booking = st.sidebar.selectbox('Ease Of Online Booking',[1,2,3,4,5])
  Food_And_Drink = st.sidebar.selectbox('Food And Drink',[1,2,3,4,5])
  Online_Boarding = st.sidebar.selectbox('Online Boarding',[1,2,3,4,5])
  Seat_Comfort = st.sidebar.selectbox('Seat Comfort',[1,2,3,4,5])
  Inflight_Entertainment = st.sidebar.selectbox('Inflight Entertainment',[1,2,3,4,5])
  On_board_Service = st.sidebar.selectbox('On-board Service',[1,2,3,4,5])
  Leg_Room = st.sidebar.selectbox('Leg Room',[1,2,3,4,5])
  Baggage_Handling = st.sidebar.selectbox('Baggage Handling',[1,2,3,4,5])
  Checkin_Service = st.sidebar.selectbox('Checkin Service',[1,2,3,4,5])
  Inflight_Service = st.sidebar.selectbox('Inflight Service',[1,2,3,4,5])
  Cleanliness = st.sidebar.selectbox('Cleanliness',[1,2,3,4,5])
  Customer_Type_Returning_Customer = st.sidebar.selectbox('Customer Type_Returning Customer',[0,1])
  Type_Of_Travel_Personal_Travel = st.sidebar.selectbox('Type Of Travel_Personal Travel',[0,1])
  Class_Economy = st.sidebar.selectbox('Class_Economy',[0,1])
  data = {'Inflight Wifi Service': Inflight_Wifi_Service,
          'Ease Of Online Booking': Ease_Of_Online_Booking,
          'Food And Drink': Food_And_Drink,
          'Online Boarding': Online_Boarding,
          'Seat Comfort': Seat_Comfort,
          'Inflight Entertainment': Inflight_Entertainment,
          'On-board Service': On_board_Service,
          'Leg Room': Leg_Room,
          'Baggage Handling': Baggage_Handling,
          'Checkin Service': Checkin_Service,
          'Inflight Service': Inflight_Service,
          'Cleanliness': Cleanliness,
          'Customer Type_Returning Customer': Customer_Type_Returning_Customer,
          'Type Of Travel_Personal Travel': Type_Of_Travel_Personal_Travel,
          'Class_Economy': Class_Economy}
  features = pd.DataFrame(data, index=[0])
  return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

prediction = rf.predict(df)
prediction_proba = rf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write('satisfied':1,'neutral or dissatisfied':0)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)