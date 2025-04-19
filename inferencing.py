import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the machine learning model
model = joblib.load('XGB_Model.pkl')
scaler = joblib.load('Scaling.pkl')

def main():
    st.title('UTS Model Deployment - 2702219572')

    # Numerical Column
    no_of_adults = st.number_input('Number of Adults', min_value=0, max_value=4, value=2)
    no_of_children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
    no_of_weekend_nights = st.number_input('Weekend Nights', min_value=0, max_value=6, value=1)
    no_of_week_nights = st.number_input('Week Nights', min_value=0, max_value=17, value=2)
    type_of_meal_plan = st.selectbox('Meal Plan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    required_car_parking_space = st.selectbox('Car Parking Space Required?', [0.0, 1.0])
    room_type_reserved = st.selectbox('Room Type Reserved', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
    lead_time = st.number_input('Lead Time', min_value=0, max_value=443, value=50)
    arrival_year = st.number_input('Arrival Year', min_value=2017, max_value=2018, value=2018)
    arrival_month = st.number_input('Arrival Month', min_value=1, max_value=12, value=6)
    arrival_date = st.number_input('Arrival Date', min_value=1, max_value=31, value=15)
    market_segment_type = st.selectbox('Market Segment', ['Online', 'Offline', 'Corporate', 'Aviation', 'Complementary'])
    repeated_guest = st.selectbox('Repeated Guest?', [0, 1])   
    no_of_previous_cancellations = st.number_input('Previous Cancellations', min_value=0, max_value=13, value=0)
    no_of_previous_bookings_not_canceled = st.number_input('Previous Not Cancellations', min_value=0, max_value=13, value=0)
    avg_price_per_room = st.number_input('Average Price per Room', min_value=0.0, value=375.5)
    no_of_special_requests = st.number_input('Special Requests', min_value=0, max_value=5, value=0)

    numerical_columns = [
        'no_of_adults',
        'no_of_children',
        'no_of_weekend_nights',
        'no_of_week_nights',
        'lead_time',
        'arrival_year',
        'arrival_month',
        'arrival_date',
        'no_of_previous_cancellations',
        'no_of_previous_bookings_not_canceled',
        'avg_price_per_room',
        'no_of_special_requests'
    ]

    # Encode
    meal_map = {'Not Selected': 0, 'Meal Plan 1': 1, 'Meal Plan 2': 2, 'Meal Plan 3': 3}
    room_map = {'Room_Type 1': 1, 'Room_Type 2': 2, 'Room_Type 3': 3, 'Room_Type 4': 4, 'Room_Type 5': 5, 'Room_Type 6': 6, 'Room_Type 7': 7}
    segment_map = {'Online': 0, 'Offline': 1, 'Corporate': 2, 'Complementary': 3, 'Aviation': 4}

    data = {
        'no_of_adults': int(no_of_adults),
        'no_of_children': int(no_of_children),
        'no_of_weekend_nights': int(no_of_weekend_nights),
        'no_of_week_nights': int(no_of_week_nights),
        'type_of_meal_plan': meal_map[type_of_meal_plan],
        'required_car_parking_space': int(required_car_parking_space),
        'room_type_reserved': room_map[room_type_reserved],
        'lead_time': int(lead_time),
        'arrival_year': int(arrival_year),
        'arrival_month': int(arrival_month),
        'arrival_date': int(arrival_date),
        'market_segment_type': segment_map[market_segment_type],
        'repeated_guest': int(repeated_guest),
        'no_of_previous_cancellations': int(no_of_previous_cancellations),
        'no_of_previous_bookings_not_canceled': int(no_of_previous_bookings_not_canceled),
        'avg_price_per_room': float(avg_price_per_room),
        'no_of_special_requests': int(no_of_special_requests)
    }

    df = pd.DataFrame([list(data.values())], columns = ['no_of_adults',
        'no_of_children',
        'no_of_weekend_nights',
        'no_of_week_nights',
        'type_of_meal_plan',
        'required_car_parking_space',
        'room_type_reserved',
        'lead_time',
        'arrival_year',
        'arrival_month',
        'arrival_date',
        'market_segment_type',
        'repeated_guest',
        'no_of_previous_cancellations',
        'no_of_previous_bookings_not_canceled',
        'avg_price_per_room',
        'no_of_special_requests'
    ])
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    y_map = {0: 'Booking Not Canceled', 1: 'Booking Canceled'}

    if st.button('Make Prediction'):
        features = df
        result = make_prediction(features)
        st.success(f'The prediction is: {y_map[result]}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()