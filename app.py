
import pickle
import streamlit as st

#loading the trained model
pickle_in = open('Airline_satif_model.pkl','rb')
classifier = pickle.load(pickle_in)

#@st.cache()
#defining the function which will make the prediction using the data imputed
def prediction(Inflight_Wifi_Service,Ease_Of_Online_Booking,Food_And_Drink,Online_Boarding,Seat_Comfort,Inflight_Entertainment,On_board_Service,Leg_Room,Baggage_Handling,Checkin_Service,Inflight_Service,Cleanliness,Customer_Type_Returning_Customer,Type_Of_Travel_Personal_Travel,Class_Economy):
  #Making predictions
  prediction = classifier.predict([[Inflight_Wifi_Service,Ease_Of_Online_Booking,Food_And_Drink,Online_Boarding,Seat_Comfort,Inflight_Entertainment,On_board_Service,Leg_Room,Baggage_Handling,Checkin_Service,Inflight_Service,Cleanliness,Customer_Type_Returning_Customer,Type_Of_Travel_Personal_Travel,Class_Economy]])

  if prediction == 0:
    pred = 'Disatisfied'
  else:
    pred = 'Satisfied'
  return pred
#Main function to define the webpage
#def main():       
#    # front end elements of the web page 
#    html_temp = """ 
#    <div style ="background-color:yellow;padding:13px"> 
#    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
#    </div> 
#    """
#    # display the front end aspect
#    st.markdown(html_temp, unsafe_allow_html = True) 
# following lines create boxes in which user can enter data required to make prediction
Inflight_Wifi_Service = st.selectbox('Inflight Wifi Service',(1,2,3,4,5))
Ease_Of_Online_Booking = st.selectbox('Ease Of Online Booking',(1,2,3,4,5))
Food_And_Drink = st.selectbox('Food And Drink',(1,2,3,4,5))
Online_Boarding  = st.selectbox('Online Boarding',(1,2,3,4,5))
Seat_Comfort = st.selectbox('Seat Comfort',(1,2,3,4,5))
Inflight_Entertainment = st.selectbox('Inflight Entertainment',(1,2,3,4,5))
On_board_Service = st.selectbox('On-board Service',(1,2,3,4,5))
Leg_Room = st.selectbox('Leg Room',(1,2,3,4,5))
Baggage_Handling = st.selectbox('Baggage Handling',(1,2,3,4,5))
Checkin_Service = st.selectbox('Checkin Service',(1,2,3,4,5))
Inflight_Service = st.selectbox('Inflight Service',(1,2,3,4,5))
Cleanliness = st.selectbox('Cleanliness',(1,2,3,4,5))
Customer_Type_Returning_Customer = st.selectbox('Customer Type_Returning Customer',(0,1))
Type_Of_Travel_Personal_Travel = st.selectbox('Type Of Travel_Personal Travel',(0,1))
Class_Economy = st.selectbox('Class_Economy',(0,1))

result =""

# when 'Predict' is clicked, make the prediction and store it 
if st.button("Predict"):
  result = prediction(Inflight_Wifi_Service,Ease_Of_Online_Booking,Food_And_Drink,Online_Boarding,Seat_Comfort,Inflight_Entertainment,On_board_Service,Leg_Room,Baggage_Handling,Checkin_Service,Inflight_Service,Cleanliness,Customer_Type_Returning_Customer,Type_Of_Travel_Personal_Travel,Class_Economy) 
  st.success('Your client is {}'.format(result))
  print(result)

#if __name__=='__main__': 
#    main()