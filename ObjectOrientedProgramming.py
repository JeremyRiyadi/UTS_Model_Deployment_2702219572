import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, LabelEncoder

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """ Load dataset from CSV file. """
        self.df = pd.read_csv(self.file_path)

    def split_data(self, target_column, test_size=0.2, random_state=42):
        """ Separate features and target variable and split the data into train and test. """
        x = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    def preprocess_data(self):
        """ Handle missing values, duplicates, and preprocessing steps. """
        # Drop Identifier
        self.x_train.drop(columns=["Booking_ID"], inplace=True)
        self.x_test.drop(columns=["Booking_ID"], inplace=True)

        # Handle missing values for numerical and categorical columns
        self.x_train['avg_price_per_room'] = self.x_train['avg_price_per_room'].fillna(self.x_train['avg_price_per_room'].mean())
        self.x_test['avg_price_per_room'] = self.x_test['avg_price_per_room'].fillna(self.x_test['avg_price_per_room'].mean())
        self.x_train['type_of_meal_plan'] = self.x_train['type_of_meal_plan'].fillna(self.x_train['type_of_meal_plan'].mode()[0])
        self.x_test['type_of_meal_plan'] = self.x_test['type_of_meal_plan'].fillna(self.x_test['type_of_meal_plan'].mode()[0])
        self.x_train['required_car_parking_space'] = self.x_train['required_car_parking_space'].fillna(self.x_train['required_car_parking_space'].mode()[0])
        self.x_test['required_car_parking_space'] = self.x_test['required_car_parking_space'].fillna(self.x_test['required_car_parking_space'].mode()[0])
        
        # Remove duplicates
        self.x_train = self.x_train.drop_duplicates()
        self.y_train = self.y_train.loc[self.x_train.index]

        self.x_test = self.x_test.drop_duplicates()
        self.y_test = self.y_test.loc[self.x_test.index]

class Preprocessor:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def encode_categorical(self):
        """ Encoding categorical columns. """
        meal_map = {'Not Selected': 0, 'Meal Plan 1': 1, 'Meal Plan 2': 2, 'Meal Plan 3': 3}
        room_map = {'Room_Type 1': 1, 'Room_Type 2': 2, 'Room_Type 3': 3, 'Room_Type 4': 4, 'Room_Type 5': 5, 'Room_Type 6': 6, 'Room_Type 7': 7}
        segment_map = {'Online': 0, 'Offline': 1, 'Corporate': 2, 'Complementary': 3, 'Aviation': 4}

        self.x_train['type_of_meal_plan'] = self.x_train['type_of_meal_plan'].map(meal_map)
        self.x_test['type_of_meal_plan'] = self.x_test['type_of_meal_plan'].map(meal_map)

        self.x_train['room_type_reserved'] = self.x_train['room_type_reserved'].map(room_map)
        self.x_test['room_type_reserved'] = self.x_test['room_type_reserved'].map(room_map)

        self.x_train['market_segment_type'] = self.x_train['market_segment_type'].map(segment_map)
        self.x_test['market_segment_type'] = self.x_test['market_segment_type'].map(segment_map)

    def encode_target(self, mapping):
        """ Encode target variable using a provided mapping. """
        self.y_train = self.y_train.map(mapping)
        self.y_test = self.y_test.map(mapping)

    def scale_data(self, num_cols):
        """ Scaling numerical data. """
        from sklearn.preprocessing import RobustScaler
        self.scaler = RobustScaler()
        self.x_train[num_cols] = self.scaler.fit_transform(self.x_train[num_cols])
        self.x_test[num_cols] = self.scaler.transform(self.x_test[num_cols])

class ModelHandler:
    def __init__(self, model_type, params=None):
        self.model_type = model_type
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_prob = None
        self.params = params

    def train_model(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(**self.params)
        elif self.model_type == 'xgb':
            self.model = XGBClassifier(**self.params)

        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        """ Make predictions and evaluate the model performance. """
        self.y_pred = self.model.predict(self.x_test)
        self.y_pred_prob = self.model.predict_proba(self.x_test)[:, 1]
        return classification_report(self.y_test, self.y_pred)

    def fine_tune(self, param_grid):
        """ Perform hyperparameter tuning with GridSearchCV. """
        grid_search = GridSearchCV(self.model, param_grid, scoring='f1_macro', cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(self.x_train, self.y_train)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_
    
    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

class ModelEvaluator:
    @staticmethod
    def confusion_matrix(y_test, y_pred):
        """ Generate confusion matrix. """
        return confusion_matrix(y_test, y_pred)

    @staticmethod
    def classification_report(y_test, y_pred):
        """ Generate classification report. """
        return classification_report(y_test, y_pred)

# ✅ Dataset Name
file_path = 'C:/Users/Jeremy Djohar Riyadi/Downloads/Dataset_B_hotel.csv'

# Data Loading and Preprocessing
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.split_data(target_column='booking_status')
data_handler.preprocess_data()

preprocessor = Preprocessor(data_handler.x_train, data_handler.x_test, data_handler.y_train, data_handler.y_test)
preprocessor.encode_categorical()
mapping = {
    'Canceled': 1,
    'Not_Canceled': 0
}
preprocessor.encode_target(mapping)

data_handler.y_train = preprocessor.y_train
data_handler.y_test = preprocessor.y_test

num_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'lead_time', 'arrival_year', 'arrival_month', 'arrival_date', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests']
preprocessor.scale_data(num_cols)

with open("Scaler.pkl", 'wb') as f:
    pickle.dump(preprocessor.scaler, f)

# Model Training
rf_params = {'n_estimators': 100, 'max_depth': 3, 'criterion': 'gini', 'random_state': 42}
model_handler = ModelHandler('rf', params=rf_params)
model_handler.train_model(data_handler.x_train, data_handler.y_train, data_handler.x_test, data_handler.y_test)

# Evaluate Model
print("Random Forest Classification Report:\n", model_handler.evaluate_model())

# Hyperparameter Tuning
rf_param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 8], 'criterion': ['gini', 'entropy']}
best_rf_params = model_handler.fine_tune(rf_param_grid)
print(f"Best Random Forest Parameters: {best_rf_params}")
print("Random Forest Classification Report:\n", model_handler.evaluate_model())

# XGBoost Model
xgb_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'random_state': 42}
xgb_model_handler = ModelHandler('xgb', params=xgb_params)
xgb_model_handler.train_model(data_handler.x_train, data_handler.y_train, data_handler.x_test, data_handler.y_test)

# Evaluate XGBoost Model
print("XGBoost Classification Report:\n", xgb_model_handler.evaluate_model())

# Hyperparameter Tuning for XGBoost
xgb_param_grid = {'n_estimators': [100, 150], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0], 'gamma': [0, 1]}
best_xgb_params = xgb_model_handler.fine_tune(xgb_param_grid)
print(f"Best XGBoost Parameters: {best_xgb_params}")
print("XGBoost Fine-Tune Classification Report:\n", xgb_model_handler.evaluate_model())

# ✅ Save the newly trained model
model_filename = 'XGB_model.pkl'
xgb_model_handler.save_model(model_filename)
print(f"🎉 Model saved successfully as {model_filename}")