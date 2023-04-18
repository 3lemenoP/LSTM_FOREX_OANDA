import numpy as np
import pandas as pd
import time
import requests
import plotly.graph_objs as go
import plotly.subplots as sp
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import oandapyV20
from oandapyV20 import API
from oandapyV20.endpoints import accounts, instruments, orders, trades

access_token = '56bbc94c833afc8606c6b1420b93453b-34b65646039c8c90c001aba7e7af6330'
api = API(access_token=access_token)

def get_account_id(api):
    try:
        accounts_request = accounts.AccountList()
        api.request(accounts_request)
        response = accounts_request.response
        account_id = response['accounts'][0]['id']
        return account_id
    except Exception as e:
        print("Error while fetching account ID:", str(e))
        return None

def plot_forex_chart(candles, predictions, instrument):
    data = pd.DataFrame(candles, columns=['Close'])
    data.index = pd.to_datetime(data.index, unit='s')

    # Add the predictions to the original data
    for i, pred in enumerate(predictions):
        data.loc[data.index[-1] + pd.Timedelta(minutes=15 * (i + 1))] = pred

    # Create a plotly line chart
    fig = sp.make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'), secondary_y=False)

    # Set axis labels and chart title
    fig.update_layout(title=f'{instrument} Predictions', xaxis_title='Time', yaxis_title='Price')

    # Display the chart
    fig.show()



class ForexPredictor:
    def __init__(self, model_file, api_key, instrument, granularity, look_back, account_id):
        self.model = load_model(model_file)
        self.api_key = api_key
        self.instrument = instrument
        self.granularity = granularity
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.account_id = account_id


    def get_candles(self):
        try:
            account_id = self.account_id
            instrument = self.instrument
            granularity = self.granularity
            count = self.look_back

            params = {"granularity": granularity, "count": count}
            endpoint = instruments.InstrumentsCandles(instrument, params=params)
            api.request(endpoint)
            response = endpoint.response

            candles = []
            for candle in response['candles']:
                candles.append(float(candle['mid']['c']))
            return candles

        except Exception as e:
            print("Error while fetching candles:", str(e))
            return None

    
        # If all retries fail, return None
            print("Failed to get candles after all retries.")
            return None



    def preprocess_data(self, data):
        data = np.array(data)  # Convert list to NumPy array
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X = scaled_data[-self.look_back:]  # Get the last look_back (3) values
        X = np.reshape(X, (1, 1, X.shape[0]))  # Adjust the shape to (None, 1, 3)
        return X




    def predict_on_new_candle(self, steps_ahead=5):
        candles = self.get_candles()

        if candles is None:
            print("Failed to get candles for prediction.")
            return None

        predictions = []
        for _ in range(steps_ahead):
            X = self.preprocess_data(candles)
            prediction = self.model.predict(X)
            predicted_price = self.scaler.inverse_transform(prediction)
            predictions.append(predicted_price[0][0])

            # Append the predicted price to candles and remove the oldest value
            candles.append(predicted_price[0][0])
            candles.pop(0)

        return predictions



if __name__ == "__main__":
    model_file = "/Users/maxlicciardi/LSTM_OANDA/LSTM_FOREX_OANDA/forex_lstm/models/forex_lstm_model.h5"
    api_key = "5a3897d1a03ebf8418a4d25c08fabb57-3d8be3f4c3bda05296a29bd949d664e7"
    instrument = "EUR_USD"
    granularity = "M15"
    look_back = 32
    steps_ahead = 10

    account_id = get_account_id(api)
    if account_id is None:
        print("Failed to get account ID.")
    else:
        forex_predictor = ForexPredictor(model_file, api_key, instrument, granularity, look_back, account_id)
        predictions = forex_predictor.predict_on_new_candle(steps_ahead)
        if predictions is not None:
            for i, prediction in enumerate(predictions):
                print(f"Predicted price {i+1}: {prediction:.5f}")

            # Plot the chart with predictions
            plot_forex_chart(forex_predictor.get_candles(), predictions, instrument)
        else:
            print("Failed to generate predictions.")


