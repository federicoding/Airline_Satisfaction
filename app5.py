
import pickle
import streamlit as st

#loading the trained model
classifier = pickle.load(open("Airline_satif_model.pkl", "rb"))

@st.cache()
def prediction_func(Inflight_Wifi_Service,Ease_Of_Online_Booking,
               Food_And_Drink,Online_Boarding,Seat_Comfort,
               Inflight_Entertainment,On_board_Service,Leg_Room,
               Baggage_Handling,Checkin_Service,Inflight_Service,
               Cleanliness,Customer_Type_Returning_Customer,
               Type_Of_Travel_Personal_Travel,Class_Economy):
  #Make predictions
  prediction = classifier.predict(df)
  prediction_proba = classifier.predict_proba(df)

  return prediction, prediction_proba

# On tente un truc un peu joli

def main():
  #front end elements
  html_temp = """
  <div style ="background-color:yellow;padding:13px"> 
  <h1 style ="color:black;text-align:center;">Streamlit Airline Satisfaction Prediction ML App</h1> 
  </div> 
  """

  #display the front end aspect
  st.markdown(thml_temp, unsafe_allow_thml = True)

  #rajout des lignes pour selectionner le valeurs du voyageur
  def user_input_features():
    Inflight_Wifi_Service = st.selectbox('Inflight Wifi Service',[1,2,3,4,5])
    Ease_Of_Online_Booking = st.selectbox('Ease Of Online Booking',[1,2,3,4,5])
    Food_And_Drink = st.selectbox('Food And Drink',[1,2,3,4,5])
    Online_Boarding = st.selectbox('Online Boarding',[1,2,3,4,5])
    Seat_Comfort = st.selectbox('Seat Comfort',[1,2,3,4,5])
    Inflight_Entertainment = st.selectbox('Inflight Entertainment',[1,2,3,4,5])
    On_board_Service = st.selectbox('On-board Service',[1,2,3,4,5])
    Leg_Room = st.selectbox('Leg Room',[1,2,3,4,5])
    Baggage_Handling = st.selectbox('Baggage Handling',[1,2,3,4,5])
    Checkin_Service = st.selectbox('Checkin Service',[1,2,3,4,5])
    Inflight_Service = st.selectbox('Inflight Service',[1,2,3,4,5])
    Cleanliness = st.selectbox('Cleanliness',[1,2,3,4,5])
    Customer_Type_Returning_Customer = st.selectbox('Customer Type_Returning Customer',[0,1])
    Type_Of_Travel_Personal_Travel = st.selectbox('Type Of Travel_Personal Travel',[0,1])
    Class_Economy = st.selectbox('Class_Economy',[0,1])
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

  df = uset_input_features()

  #Bouton pour lancer la prédiction et store les values...

  if st.button('Predict'):
    result = prediction_func(df)
    st.success('Your Passenger prediction is {0}, with probability {1}'.format(result[0],result[1]))

if __name__=='__main__':
  main()