import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title='DashBoard', page_icon=":chart_with_upward_trend:", layout='wide')


########################################--ROW---########################################################################
st.title(':bar_chart: Stock Market Trend')
st.markdown("<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)

user_input = st.text_input('Enter Stock Ticker', 'AAPL')



########################################--ROW---########################################################################
col1,col2 = st.columns((2))
startDate = pd.to_datetime('2000-01-01')
endDate = pd.to_datetime('2023-11-30')

with col1:
    date1 = st.date_input("Start Date", startDate)
with col2:
    date2 = st.date_input("End Date", endDate)

    
df = yf.download(user_input, start=date1, end=date2)



########################################--ROW---########################################################################
col1,col2 = st.columns((2))

with col1:
    st.subheader(f'Data for {user_input} from {date1.strftime("%d-%m-%Y")} to {date2.strftime("%d-%m-%Y")}')
    st.write(df.style.background_gradient(cmap="Blues"))
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Data", data = csv, file_name=f"{user_input}_stock.csv", mime='text/csv', help='Click here to download the data as CSV file')


with col2:    
    # Visualizations
    st.subheader('Closing Price vs Time Chart')
    fig = px.line(df.Close, template='seaborn')
    st.plotly_chart(fig, use_container_width=True)
    


########################################--ROW---########################################################################
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
ma200 = df.Close.rolling(200).mean()
# ma200[int(len(ma200)*0.70):]

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
fig = px.line(x=ma200[int(len(ma200)*0.70):].index, 
              y=[
                  ma100[int(len(ma200)*0.70):], 
                  ma200[int(len(ma200)*0.70):], 
                  df.Close[int(len(ma200)*0.70):]])
fig.data[0].name="MA100"
fig.data[0].hovertemplate = "MA100"
fig.data[0].line.color="lightgreen"
fig.data[1].name="MA200"
fig.data[1].hovertemplate = "MA200"
fig.data[1].line.color="olive"
fig.data[2].name="Original"
fig.data[2].hovertemplate = "Original"
fig.data[2].line.color="purple"
st.plotly_chart(fig, use_container_width=True)



########################################--ROW---########################################################################
#function to generate feature technical indicators
def get_technical_indicators(dataset): 
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['Close'].rolling(window = 7).mean()
    dataset['ma21'] = dataset['Close'].rolling(window = 21).mean()
    #Create Bollinger Bands
    dataset['20sd'] = dataset['Close'].rolling(window = 20).std()
    dataset['upper_band'] = (dataset['Close'].rolling(window = 20).mean()) + (dataset['20sd']*2)
    dataset['lower_band'] = (dataset['Close'].rolling(window = 20).mean()) - (dataset['20sd']*2)
    return dataset

df1 = get_technical_indicators(df)



last_days=500
dataset = df1.iloc[-last_days:, :]

# fig = plt.figure()
st.subheader("Uppper Bound & Lower Bound")
fig = px.line(dataset, y=['ma7', "Close", "ma21", "upper_band","lower_band"])
fig['layout'].update(title='Bollinger_Bands vs MA7 vs MA21',titlefont=dict(size=20), xaxis=dict(title='Date', titlefont=dict(size=15)), 
                       yaxis=dict(title='Price', titlefont=dict(size=15)))
fig.data[0].line.color = "dodgerblue"
fig.data[1].line.color = "darkgreen"
fig.data[2].line.color = "magenta"
fig.data[3].line.color = "aliceblue"
fig.data[4].line.color = "aliceblue"

st.plotly_chart(fig, use_container_width=True)




# ########################################---ROW---########################################################################
data = df.filter(['Close'])
# dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil(len(dataset) * .90 ))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
# rmse

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data

# fig = plt.figure(figsize=(16,6))
# plt.title('Model')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
# st.pyplot(fig)

# fig = px.line()

st.subheader("LSTM Model")
data['Predictions'] = np.nan
data = data.reset_index()
data.loc[training_data_len:,'Predictions'] = predictions
pred_graph = px.line(x=data['Date'], y=[data['Close'],data['Predictions']], template='seaborn')
pred_graph.data[0].name="Original Value"
pred_graph.data[0].hovertemplate = "Original Value"
pred_graph.data[0].line.color="#FF0000"
pred_graph.data[1].name="Predictions"
pred_graph.data[1].hovertemplate = "Predictions"
pred_graph.data[1].line.color="#00FF00"
pred_graph['layout'].update(xaxis=dict(title='Date', titlefont=dict(size=20)), 
                       yaxis=dict(title='Closing Price', titlefont=dict(size=20)))
st.plotly_chart(pred_graph, use_container_width=True)





# ########################################---ROW---########################################################################
col1,col2 = st.columns((2))

with col1:
    st.subheader("Model Summary")
    model.summary(print_fn=lambda x: st.text(x))
    col11,col12 = st.columns((2))
    with col11:
        st.write("Root Mean Squared Error")
    with col12:
        st.write(rmse)

with col2:
    st.write(valid.style.background_gradient(cmap="Greens"))
    csv = valid.to_csv(index=True).encode('utf-8')
    st.download_button("Download Data", data = csv, file_name=f"{user_input}_Pred.csv", mime='text/csv', help='Click here to download the data as CSV file')


