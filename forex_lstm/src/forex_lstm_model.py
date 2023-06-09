import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class ForexLSTM:
    def __init__(self, input_file, look_back=1, steps_out=5, epochs=100, batch_size=1):
        self.input_file = input_file
        self.look_back = look_back
        self.steps_out = steps_out
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None

    def load_and_preprocess_data(self, train_ratio=0.8):
        # Load and scale the data
        df = pd.read_csv(self.input_file)
        dataset = df['close'].values
        dataset = dataset.astype('float32')
        dataset = dataset.reshape(-1, 1)
        dataset = self.scaler.fit_transform(dataset)

        # Split the data into training and testing sets
        train_size = int(len(dataset) * train_ratio)
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        # Create the input (X) and output (Y) datasets
        def create_dataset(dataset, look_back, steps_out):
            X, Y = [], []
            for i in range(len(dataset) - look_back - steps_out):
                X.append(dataset[i:(i + look_back), 0])
                Y.append(dataset[(i + look_back):(i + look_back + steps_out), 0])
            return np.array(X), np.array(Y)

        self.trainX, self.trainY = create_dataset(train, self.look_back, self.steps_out)
        self.testX, self.testY = create_dataset(test, self.look_back, self.steps_out)

        # Reshape the input data to be compatible with the LSTM model
        self.trainX = self.trainX.reshape(self.trainX.shape[0], 1, self.trainX.shape[1])
        self.testX = self.testX.reshape(self.testX.shape[0], 1, self.testX.shape[1])

    def create_and_compile_model(self, lstm_units=100):
        # Create the LSTM model with 3 LSTM layers, 2 Dense layers, and dropout layers
        self.model = Sequential()
        self.model.add(LSTM(lstm_units, input_shape=(1, self.look_back), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(lstm_units, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(lstm_units // 2))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(lstm_units // 2, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.steps_out, activation='linear'))

        # Compile the model
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train_model(self):
        # Train the model
        self.model.fit(self.trainX, self.trainY, epochs=self.epochs, batch_size=self.batch_size,verbose=2)

    def evaluate_model(self):
        # Generate predictions
        trainPredict = self.model.predict(self.trainX)
        testPredict = self.model.predict(self.testX)

        # Invert the predictions and calculate the RMSE
        trainPredict = self.scaler.inverse_transform(trainPredict)
        trainY_inverse = self.scaler.inverse_transform(self.trainY)
        testPredict = self.scaler.inverse_transform(testPredict)
        testY_inverse = self.scaler.inverse_transform(self.testY)

        # Calculate RMSE for each step
        train_scores, test_scores = [], []
        for i in range(self.steps_out):
            train_score = np.sqrt(mean_squared_error(trainY_inverse[:, i], trainPredict[:, i]))
            test_score = np.sqrt(mean_squared_error(testY_inverse[:, i], testPredict[:, i]))
            train_scores.append(train_score)
            test_scores.append(test_score)
            print(f'Train Score (step {i+1}): {train_score:.2f} RMSE')
            print(f'Test Score (step {i+1}): {test_score:.2f} RMSE')

        return trainPredict, testPredict

def save_model(self, model_file):
    self.model.save(model_file)
    print(f"Model saved to: {model_file}")


    def evaluate_model(self):
        # Generate predictions
        trainPredict = self.model.predict(self.trainX)
        testPredict = self.model.predict(self.testX)

        # Invert the predictions and calculate the RMSE
        trainPredict = self.scaler.inverse_transform(trainPredict)
        trainY_inverse = self.scaler.inverse_transform([self.trainY])
        testPredict = self.scaler.inverse_transform(testPredict)
        testY_inverse = self.scaler.inverse_transform([self.testY])

        train_score = np.sqrt(mean_squared_error(trainY_inverse[0], trainPredict[:, 0]))
        test_score = np.sqrt(mean_squared_error(testY_inverse[0], testPredict[:, 0]))

        print(f'Train Score: {train_score:.2f} RMSE')
        print(f'Test Score: {test_score:.2f} RMSE')

        return trainPredict, testPredict
    
    def save_model(self, model_file):
        self.model.save(model_file)
        print(f"Model saved to: {model_file}")


if __name__ == "__main__":
    input_file = "/Users/maxlicciardi/LSTM_OANDA/LSTM_FOREX_OANDA/forex_lstm/data/forex_data.csv"
    model_file = "/Users/maxlicciardi/LSTM_OANDA/LSTM_FOREX_OANDA/forex_lstm/models/forex_lstm_model.h5"
    look_back = 32
    steps_out = 15
    epochs = 100
    batch_size = 4

    forex_lstm = ForexLSTM(input_file, look_back, steps_out, epochs, batch_size)
    forex_lstm.load_and_preprocess_data()
    forex_lstm.create_and_compile_model()
    forex_lstm.train_model()
    trainPredict, testPredict = forex_lstm.evaluate_model()
    forex_lstm.save_model(model_file)


