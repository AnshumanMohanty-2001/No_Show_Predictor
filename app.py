import pickle
import datetime as dt
import streamlit as st
from sklearn.preprocessing import LabelEncoder

weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

today_date = dt.date.today()

with open('best_final_model.pkl', 'rb') as file:
    model = pickle.load(file)

def predict_probability(age, scholarship, Hypertension, Diabetes, Handicap, SMS_received, transformed_waiting_days, Weekday_Appointment):
    # Predict the probability
    probability = model.predict_proba([[age, scholarship, Hypertension, Diabetes, Handicap, SMS_received, transformed_waiting_days, Weekday_Appointment]])[:, 1][0] * 100
    return probability

# Define the Box-Cox transformation function
def boxcox_transform(value, lambda_value = -0.1299):
    return ((value+1) ** lambda_value - 1) / lambda_value

def encoding_vals(col_name):
    if col_name == 'Yes':
        return 1
    return 0

def main():
    st.title("Patient Show Predictor")



    Age = st.number_input("Age", value=0, min_value=0, max_value=102)
    st.markdown(f"<br>", unsafe_allow_html=True)

    # Create columns for inputs
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        sms_options = st.radio("SMS Sent", ["Yes", "No"])
        sms_options = encoding_vals(sms_options)

    with col2:
        Scholarship = st.radio("Scholarship", ["Yes", "No"])
        Scholarship = encoding_vals(Scholarship)

    with col3:
        Hypertension = st.radio("Hypertension", ["Yes", "No"])
        Hypertension = encoding_vals(Hypertension)

    with col4:
        Diabetes = st.radio("Diabetes", ["Yes", "No"])
        Diabetes = encoding_vals(Diabetes)

    with col5:
        Handicap = st.radio("Handicap", ["Yes", "No"])
        Handicap = encoding_vals(Handicap)

    st.markdown(f"<br>", unsafe_allow_html=True)
    Date_of_Appointment = st.date_input("Date of Appointment", min_value=today_date)

    st.markdown(f"<br>", unsafe_allow_html=True)
    Date_of_Scheduling_Appointment = st.date_input("Date of Scheduling Appointment", max_value=Date_of_Appointment)


    waiting_days = (Date_of_Appointment - Date_of_Scheduling_Appointment).days
   

    transformed_waiting_days = boxcox_transform(waiting_days)


    day_name = Date_of_Appointment.strftime('%A')



    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Fit LabelEncoder on weekday labels
    label_encoder.fit(weekday_labels)

    # Transform user weekday value to label
    encoded_user_weekday = label_encoder.transform([day_name])[0]

    print("Encoded Weekday Value:", encoded_user_weekday)

    # Button to submit the form
    if st.button("Submit"):
        # Predict probability percentage
        probability = predict_probability(Age, Scholarship, Hypertension, Diabetes, Handicap, sms_options, transformed_waiting_days, encoded_user_weekday)
        # Show success alert with appropriate message based on probability
        if probability == 100:
            st.success("Patient will Show Up")
        elif probability == 0:
            st.error("Patient will not Show Up")
        


if __name__ == "__main__":
    main()