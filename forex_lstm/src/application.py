import tkinter as tk
from forex_predictor import ForexPredictor, get_account_id
from oandapyV20 import API
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot
import webbrowser
import os

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Forex Predictor')
        self.geometry('800x600')
        self.forecast()

    def forecast(self):
        model_file = "/Users/maxlicciardi/LSTM_OANDA/LSTM_FOREX_OANDA/forex_lstm/models/forex_lstm_model.h5"
        api_key = "56bbc94c833afc8606c6b1420b93453b-34b65646039c8c90c001aba7e7af6330"
        instrument = "EUR_USD"
        granularity = "M15"
        look_back = 32
        steps_ahead = 60

        access_token = '56bbc94c833afc8606c6b1420b93453b-34b65646039c8c90c001aba7e7af6330'
        api = API(access_token=access_token)
        account_id = get_account_id(api)

        if account_id is None:
            print("Failed to get account ID.")
        else:
            forex_predictor = ForexPredictor(model_file, api_key, instrument, granularity, look_back, account_id)
            candles = forex_predictor.get_candles()

            last_value = candles[-1]
            predictions = [last_value] + forex_predictor.predict_on_new_candle(steps_ahead - 1)

            if predictions is not None:
                for i, prediction in enumerate(predictions[1:], 1):
                    print(f"Predicted price {i+1}: {prediction:.5f}")

                self.plot_chart(candles[-50:], predictions)
            else:
                print("Failed to generate predictions.")

    def plot_chart(self, candles, predictions):
        n_candles = len(candles)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(n_candles)), y=candles, mode='lines', name='Last values'))
        fig.add_trace(go.Scatter(x=list(range(n_candles, n_candles + len(predictions))), y=predictions, mode='lines', name='Predicted values'))

        temp_file = "temp_plot.html"
        plot(fig, filename=temp_file, auto_open=False)
        webbrowser.open('file://' + os.path.realpath(temp_file))

if __name__ == '__main__':
    app = Application()
    app.mainloop()
