import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

class ForexPredictor:
    def __init__(self, model_file, input_file, look_back=3):
        self.model_file = model_file
        self.input_file = input_file
        self.look_back = look_back
        self.model = load_model(model_file)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def preprocess_data(self):
        # Load and scale the data
        df = pd.read_csv(self.input_file)
        dataset = df['close'].values
        dataset = dataset.astype('float32')
        dataset = dataset.reshape(-1, 1)
        dataset = self.scaler.fit_transform(dataset)
        return dataset
        
    def predict(self, steps_ahead=1):
        dataset = self.preprocess_data()
        
        # Create the input for the model with the last 'look_back' data points
        input_data = dataset[-self.look_back:]
        input_data = input_data.reshape(1, 1, self.look_back)
        
        # Make predictions for the specified number of time steps ahead
        predictions = []
        for _ in range(steps_ahead):
            prediction = self.model.predict(input_data)
            predictions.append(prediction)
            
            # Update the input data to include the latest prediction and remove the oldest data point
            input_data = np.append(input_data[:, :, 1:], prediction, axis=2)
        
        # Inverse transform the predictions to their original scale
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions

if __name__ == "__main__":
    model_file = "models/forex_lstm_model.h5"
    input_file = "data/forex_data.csv"
    
    forex_predictor = ForexPredictor(model_file, input_file)
    steps_ahead = 4  # Number of 15-minute intervals to predict ahead
    predictions = forex_predictor.predict(steps_ahead)
    
    for i, prediction in enumerate(predictions, start=1):
        print(f"Prediction {i} (in {i*15} minutes): {prediction[0]:.5f}")
