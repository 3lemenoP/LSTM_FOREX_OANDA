import pandas as pd
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.instruments import InstrumentsCandles
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse


def download_forex_data(api_key, instrument, granularity, start, end, output_file):
    api = API(access_token=api_key)

    start_date = parse(start)
    end_date = parse(end)

    all_data = []

    while start_date < end_date:
        next_date = start_date + relativedelta(months=1)
        if next_date > end_date:
            next_date = end_date

        params = {
            "granularity": granularity,
            "from": start_date.isoformat(),
            "to": next_date.isoformat()
        }

        try:
            r = InstrumentsCandles(instrument=instrument, params=params)
            api.request(r)
        except V20Error as e:
            print("Error: {}".format(e))
            return

        for candle in r.response['candles']:
            if candle['complete']:
                all_data.append([candle['time'], candle['volume'], candle['mid']['o'], candle['mid']['h'], candle['mid']['l'], candle['mid']['c']])

        start_date = next_date

    df = pd.DataFrame(all_data, columns=['time', 'volume', 'open', 'high', 'low', 'close'])
    df.to_csv(output_file, index=False)
    print("Data saved to: {}".format(output_file))


if __name__ == "__main__":
    api_key = "0cdbfcbfc6e6ed2cdf4db30d7aaa5474-c60fbb78ce2ad07bfd09cbb569210c5e"
    instrument = "EUR_USD"
    granularity = "H1"
    start = "2020-01-01T00:00:00Z"
    end = "2023-01-01T00:00:00Z"
    output_file = "/Users/maxlicciardi/LSTM_OANDA/LSTM_FOREX_OANDA/forex_lstm/data/forex_data.csv"

    download_forex_data(api_key, instrument, granularity, start, end, output_file)
