OandaAPI: 0cdbfcbfc6e6ed2cdf4db30d7aaa5474-c60fbb78ce2ad07bfd09cbb569210c5e

LSTM Forex Prediction Model
This project aims to predict Forex price movements using an LSTM (Long Short-Term Memory) model, a type of recurrent neural network (RNN). The model is trained on historical Forex data obtained from OANDA.

Usage
1. Download Forex Data
To download historical Forex data from OANDA, run the forex_data_downloader.py script. The script downloads the data and saves it as a CSV file in the data folder.

python src/forex_data_downloader.py

2. Train and Evaluate the LSTM Model
To train and evaluate the LSTM model, run the forex_lstm_model.py script. The script trains the model on the historical Forex data, evaluates its performance, and saves the trained model as an HDF5 file in the models folder.

Customization
You can customize various parameters in the project, such as the look-back period, the number of LSTM units, and the training parameters. These parameters can be adjusted in the forex_lstm_model.py script, in the ForexLSTM class definition or in the if __name__ == "__main__": block.

Dependencies
The project requires the following Python libraries:

numpy
pandas
requests
keras
tensorflow
sklearn
To install the required libraries, use the following command:

pip install numpy pandas requests keras tensorflow scikit-learn
