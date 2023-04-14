from forex_lstm_model import ForexLSTM

if __name__ == "__main__":
    input_file = "/Users/maxlicciardi/LSTM_OANDA/LSTM_FOREX_OANDA/forex_lstm/data/forex_data.csv"
    model_file = "/Users/maxlicciardi/LSTM_OANDA/LSTM_FOREX_OANDA/forex_lstm/models/forex_lstm_model.h5"
    look_back = 3
    epochs = 100
    batch_size = 1

    forex_lstm = ForexLSTM(input_file, look_back, epochs, batch_size)
    forex_lstm.load_and_preprocess_data()
    forex_lstm.create_and_compile_model()
    forex_lstm.train_model()
    trainPredict, testPredict = forex_lstm.evaluate_model()
    forex_lstm.save_model(model_file)
