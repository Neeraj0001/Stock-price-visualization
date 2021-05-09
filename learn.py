import investpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow import keras

df = investpy.get_stock_historical_data(stock='MSFT',
                                        country='United States',
                                        from_date='01/01/2010',
                                        to_date='31/12/2020')
df.reset_index(inplace=True)
df2 = df[['Date', 'Close']]
plt.plot(df2['Date'], df2['Close'])
plt.title('Microsoft Stock Closing Price ')
plt.xlabel('Date in (year)')
plt.ylabel('Close Price')
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.savefig('Actual_Cost_Price.png', dpi=300)
scaler = MinMaxScaler(feature_range=(0, 1))
df3 = scaler.fit_transform(np.array(df2['Close']).reshape(-1, 1))
training_size = int(df2[(df2['Date'] <= '2019-12-31')].shape[0])
test_size = len(df2) - training_size
train_data, test_data = df3[0:training_size, :], df3[training_size - 101:len(df2), :]
print(train_data.shape)
print(test_data.shape)


def create_dataset(dataset, timestep=1):
    dataX = []
    dataY = []
    for i in range(len(dataset) - timestep - 1):
        a = dataset[i:(i + timestep), 0]
        dataX.append(a)
        dataY.append(dataset[i + timestep, 0])
    return np.array(dataX), np.array(dataY)


X_train, y_train = create_dataset(train_data, 100)
X_test, y_test = create_dataset(test_data, 100)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=["accuracy"],
)
print(model.summary())

model.fit(X_train, y_train, batch_size=64, epochs=50, verbose=2)
model.evaluate(X_test, y_test, batch_size=64, verbose=2)
print(X_test.shape)
test_predict = model.predict(X_test)
test_predict = scaler.inverse_transform(test_predict)
test_predict = test_predict.flatten()
cols = ['Predicted_Cost']
df5 = pd.DataFrame(test_predict, columns=cols)
df5.reset_index(inplace=True)
df4 = df2[(df2['Date'] > '2019-12-31')]
df4.reset_index()
print(df4)
plt.clf()
plt.plot(df4['Date'], df4['Close'], c='r', label="Actual Close Price")
plt.plot(df4['Date'], df5['Predicted_Cost'], '--b', label="After Training Close Price")
plt.legend()
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.savefig('After_Prediction_Cost_Price.png', dpi=300)
plt.ioff()
