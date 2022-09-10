import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib.pylab import rcParams
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
rcParams['figure.figsize'] = 20, 10

# Read the dataset
df = pd.read_csv("NSE-TATA.csv")
df.head()

# Analyse the closing prices from the dataframe
df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
df.index = df['Date']

plt.figure(figsize=(16, 8))
plt.plot(df["Close"], label='Close Price history')


# Sort the dataset on date time and filter "Date" and "Close" columns
data = df.sort_index(ascending=True, axis=0)
new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_dataset["Date"][i] = data['Date'][i]
    new_dataset["Close"][i] = data["Close"][i]

# Normalize the newly filtered dataset
new_dataset.index = new_dataset.Date
new_dataset.drop("Date", axis=1, inplace=True)
final_dataset = new_dataset.values

# Apply feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
final_dataset = scaler.fit_transform(final_dataset)

# Split data into training and test sets
look_back = 60
train = final_dataset[0:int(0.8*(len(final_dataset)))]
test = final_dataset[int(0.8*(len(final_dataset)))-look_back:]


def create_dataset(n_X, look_back):
    """A function to split a time series dataset into x and y, and to convert
    said x data into a supervised learning data format according to an input
    look-back period.
    """
    dataX, dataY = [], []
    for i in range(0, len(n_X)-look_back):
        a = n_X[i:(i+look_back), ]
        dataX.append(a)
        dataY.append(n_X[i + look_back, ])
    return np.array(dataX), np.array(dataY)


# Spilt the training and test datasets, and convert time series data into the
# form of supervised learning data according to the look-back period set
train_x, train_y = create_dataset(train, look_back)
test_x, test_y = create_dataset(test, look_back)

# Build and train the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True,
                    input_shape=(train_x.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(train_x, train_y, epochs=10, batch_size=10, verbose=2)

# Save the predicted outcomes to CSV file
closing_price = lstm_model.predict(test_x)
closing_price = scaler.inverse_transform(closing_price)

pd.DataFrame(closing_price).to_csv('closing_price_v2.csv', index=False)

# Save the LSTM model
lstm_model.save("saved_lstm_model_v2.h5")

# Plot predicted prices against actual prices
train_data = new_dataset[0:int(0.8*(len(new_dataset)))]
valid_data = new_dataset[int(0.8*(len(new_dataset))):]
valid_data['Predictions'] = closing_price
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close', "Predictions"]])
