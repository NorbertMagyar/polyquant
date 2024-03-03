# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:51:42 2024

@author: u0099498
"""
from matplotlib import pyplot as plt
import numpy as np
import yfinance as yf
import scipy.stats
import scipy.signal
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta
import pickle
import bs4 as bs
import requests
import datetime
from decimal import Decimal
import pytz
import sys
import modules
import holidays

timezone = pytz.timezone('America/New_York')

# Downloads daily candles from polygon.io for all S&P500 companies between specified dates
def downloaddata(start_date,end_date,api='polygon'):
 resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
 soup = bs.BeautifulSoup(resp.text, 'lxml')
 table = soup.find('table', {'class': 'wikitable sortable'})
 table2 = soup.find('table', {'class': 'wikitable sortable', 'id': 'changes'})
 tickers = []
 for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)
 
 #Tickers discontinued since 2018
 #for row in table2.findAll('tr')[2:]:
 #   if row.findAll('td')[0].text[-4:] >= '2011':
 #    ticker = row.findAll('td')[3].text
 #    if ticker != '': tickers.append(ticker)
 
 tickers = [s.replace('\n', '') for s in tickers]
 
 start = datetime.datetime.strptime(start_date,"%Y-%m-%d")
 end = datetime.datetime.strptime(end_date,"%Y-%m-%d")

 if (api == 'yfinance'):  
  data = yf.download(tickers, start, end, interval='1d',progress=False)
  with open("sp500data-"+start_date+"-"+end_date+".pkl", 'wb') as f:
    pickle.dump(data, f)
 elif (api == 'polygon'):
  data = [{}]*len(tickers)

  for i,ticker in enumerate(tickers):
   data[i] = modules.client.get_aggregate_bars(ticker,start,end,multiplier=1,timespan='day')
   print('\rNo. '+str(i)+': '+ticker,end="")
   sys.stdout.write("\033[K")  
  with open("sp500data-"+start_date+"-"+end_date+"-poly.pkl", 'wb') as f:
    pickle.dump(data, f)
 return data

#Loader for daily data from yfinance
def loaddata():
   with open("sp500data-2010-01-28-2024-01-24.pkl", 'rb') as f:
     data = pd.read_pickle(f) 
   sp500 = yf.download('^GSPC', start='2010-01-28', end='2024-01-24', interval='1d',progress=False)
   close_delta = data['Close'].diff()
   x = []; y = []; dates = []; tckr = [];
   for day in data.index[205:]:
       print('\r'+str(day),end="") 
       date = day.strftime("%Y-%m-%d")
       prevdate = prevday(data, day)
       for ticker in data['Close']:
           prevprice = data['Close',ticker].loc[prevdate]
           spprice = sp500['Close'].loc[prevdate]
           price = data['Open',ticker].loc[date]
           chg = (data['Close',ticker].loc[date]-price)/price
           #Search top gain or loss stocks
           if (abs(prevprice - price)/prevprice > 0.05) and (abs(prevprice - price)/prevprice < 0.5):
              xnew =    [(SMA_data(sp500,prevdate,5)-spprice)/spprice,
                        (SMA_data(sp500,prevdate,15)-spprice)/spprice,
                        (SMA_data(sp500,prevdate,50)-spprice)/spprice,
                        (SMA_data(sp500,prevdate,200)-spprice)/spprice,
                        (sp500['Open'].loc[date] - spprice)/spprice,
                         RSI_cd(data,ticker,close_delta,prevdate)/100-0.5,
                        (SMA(data,ticker,prevdate,3)-prevprice)/prevprice,
                        (SMA(data,ticker,prevdate,7)-prevprice)/prevprice,
                        (SMA(data,ticker,prevdate,21)-prevprice)/prevprice,
                        (SMA(data,ticker,prevdate,50)-prevprice)/prevprice,
                        (SMA(data,ticker,prevdate,100)-prevprice)/prevprice,
                        (SMA(data,ticker,prevdate,200)-prevprice)/prevprice,
                        (price-prevprice)/prevprice]
              if (np.sum(np.isnan(xnew)) == 0) and (abs(chg) < 0.5) and (chg != 0.0):
                  x.append(xnew) 
                  y.append(chg)
                  dates.append(date); tckr.append(ticker)
   out = {};
   out['x'] = np.array(x); out['y'] = np.array(y); out['dates'] = np.array(dates); out['tckr'] = np.array(tckr)
   with open("data-2011-2024", 'wb') as f:
       pickle.dump(out, f)
   return np.array(x), np.array(y), np.array(dates), np.array(tckr)

def loaddatals():
    with open("sp500data-2018-02-19-2024-02-19.pkl", 'rb') as f:
       data = pd.read_pickle(f)
    with open("sp500data-2019-01-28-2024-01-24-poly.pkl", 'rb') as f:
       df = pickle.load(f)
       df = dictdf(df)
    data.rename({'DAY':'CDAY'},axis='columns',inplace=True)
    with open("losers5perc-2019-01-28-2024-01-24-poly.pkl", 'rb') as f:
       datals = pd.read_pickle(f)
    with open("gainers5perc-2019-01-28-2024-01-24-poly.pkl", 'rb') as f:
       datagn = pd.read_pickle(f)   
    datals =  datals + datagn
    sp500 = yf.download('^GSPC', start='2010-01-28', end='2024-01-24', interval='1d',progress=False)
    x = []; y = []; dates = []; tckr = [];
    for day in range(len(datals)):
       # Avoid highly anomalous days around Covid crash
       if (day < 280 and day > 270) or (day < 1585 and day > 1515): continue
       print('\r'+str(day),end="") 
       if datals[day] is not None:
          prevdate = datals[day].index[0].strftime("%Y-%m-%d")
          date = datals[day].index[-1].strftime("%Y-%m-%d")
          for ticker in datals[day].columns.levels[0].values:
              spprice = sp500['Close'].loc[prevdate]
              # VWAP of the previous day
              prevprice = df.loc[datals[day].index[0].strftime("%Y-%m-%d"),ticker]['VWAP']
              # Average 5 min volume of the previous day
              prevvol = df.loc[datals[day].index[0].strftime("%Y-%m-%d"),ticker]['Volume']/78.  
              daydata = datals[day][ticker].loc[datals[day].index.strftime("%H:%M") == "09:30"].iloc[-1].values
              if np.isnan(daydata[0]) == False:
                 # trainsource contains: 
                 # [0]: Avg 5 min. Volume of previous day, normalized to max volume
                 # [2]-[5]: Open-Close-High-Low (OCHL) price of first 5 min ticker of day
                 xnew = [RSI(data,ticker,prevdate)/100-0.5,
                        (SMA_data(sp500,prevdate,3)-spprice)/spprice,
                        (SMA_data(sp500,prevdate,21)-spprice)/spprice,
                        (SMA_data(sp500,prevdate,50)-spprice)/spprice,
                        (SMA_data(sp500,prevdate,200)-spprice)/spprice,
                        (sp500['Open'].loc[date] - spprice)/spprice,
                        (SMA(data,ticker,prevdate,3)-prevprice)/prevprice,
                        (SMA(data,ticker,prevdate,7)-prevprice)/prevprice,
                        (SMA(data,ticker,prevdate,21)-prevprice)/prevprice,
                        (SMA(data,ticker,prevdate,50)-prevprice)/prevprice,
                        (SMA(data,ticker,prevdate,100)-prevprice)/prevprice,
                        (SMA(data,ticker,prevdate,200)-prevprice)/prevprice,
                        (prevvol-daydata[0])/np.max((daydata[0],prevvol))*0.1,
                        (daydata[2]-prevprice)/prevprice,(daydata[3]-prevprice)/prevprice,
                        (daydata[4]-prevprice)/prevprice,(daydata[5]-prevprice)/prevprice]
                 if (np.sum(np.isnan(xnew)) == 0):
                         x.append(xnew) 
                 else:
                         continue
                 #y.append((datals[day].loc[date][ticker,'VWAP'].mean()-daydata[3])/prevprice)
                 #y.append((df.loc[date,ticker]['Close']-daydata[3])/daydata[3])
                 y.append((datals[day].loc[date][ticker,'VWAP'].between_time("09:35","15:55").mean()-daydata[3])/daydata[3])
              dates.append(date); tckr.append(ticker)
    out = {};
    out['x'] = np.array(x); out['y'] = np.array(y); out['dates'] = np.array(dates);  out['tckr'] = np.array(tckr)
    with open("datals-2019-2024", 'wb') as f:
       pickle.dump(out, f)
    return np.array(x), np.array(y), dates, tckr

def saveplots(data,ticker,day,name):
    fig = candlefig()
    mdata = dictdf(modules.client.get_aggregate_bars(
            ticker,data.index[data.index.get_loc(day)-1].strftime("%Y-%m-%d"),
            day,multiplier=5,timespan='minute'))
    plotcandle(fig, mdata, ticker)
    pio.write_image(fig, name+'.png',scale=6, width=1580, height=640)
    return
    
# Transforms the dict data as downloaded from polygon.io to a pd.dataframe for easier manipulation
def dictdf(data):  
   b = ['Volume','VWAP','Open','Close','High','Low']
   # Multi-day 
   if type(data) is dict:
     index = pd.to_datetime([datetime.datetime.fromtimestamp(x['t']//1000,tz=timezone).\
                             strftime("%Y-%m-%d %H:%M:%S") for x in data['results']])
     dfi = pd.DataFrame(data['results'],index=index).drop(['t','n'],axis=1)
     a = [data['ticker']] * (len(dfi.columns))
     dfi.columns = pd.MultiIndex.from_arrays([a,b])
     df = dfi
   else:
     try:
       data[0]['results']
     except (KeyError,IndexError) as e:
       return 
     else:
       if datetime.datetime.fromtimestamp(data[0]['results'][0]['t']//1000,tz=timezone).hour == 0:
         timestr = "%Y-%m-%d"
       else:
         timestr = "%Y-%m-%d %H:%M:%S"
       for i in range(0,len(data)):
         try:
           data[i]['results']
         except KeyError:
           continue 
         else:
          index = pd.to_datetime([datetime.datetime.fromtimestamp(x['t']//1000,tz=timezone).\
                               strftime(timestr) for x in data[i]['results']])
          dfi = pd.DataFrame(data[i]['results'],index=index).drop(['t','n'],axis=1)
          a = [data[i]['ticker']] * (len(dfi.columns))
          dfi.columns = pd.MultiIndex.from_arrays([a,b])
          if (i==0):
           df = dfi
          else:
           df = df.join(dfi)
   return df

""" Taken from: https://blog.quantinsti.com/build-technical-indicators-in-python/ """
# Simple Moving Average 
def SMA(data, ticker, date, ndays): 
    return data['Close',ticker].rolling(ndays).mean().loc[date]

def SMA_data(data, date, ndays): 
    return data['Close'].rolling(ndays).mean().loc[date]

# Exponentially-weighted Moving Average 
def EWMA(data, ticker, date, ndays): 
    return data['Close',ticker].ewm(span = ndays, min_periods = ndays - 1).mean().loc[date]

# Compute the Bollinger Bands 
def BBANDS(data, ticker, date, window):
    MA = data.loc[date]['Close',ticker].rolling(window).mean()
    SD = data.loc[date]['Close',ticker].rolling(window).std()
    return MA, MA + 2*SD, MA - 2*SD

# Returns RSI values
def RSI(data, ticker, date, periods = 14):
    
    close_delta = data['Close',ticker].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()

    RSI = ma_up / ma_down
    RSI = 100 - (100/(1 + RSI))
    return RSI.loc[date]

# Returns RSI values
def RSI_cd(data, ticker, close_delta, date, periods = 14):
    
    close_delta = close_delta[ticker]
    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()

    RSI = ma_up / ma_down
    RSI = 100 - (100/(1 + RSI))
    return RSI.loc[date]

def findticker(data,ticker):
    for i in range(len(data)):
        if data[i]['ticker'] == ticker: return i
    
def getmodel(no,size):
    match no:
        case 1:
            model = Sequential()
            model.add(InputLayer(input_shape=(size,1)))
            model.add(LSTM(128)); model.add(Dropout(0.3))
            model.add(Dense(units=16))
            model.add(Dense(units=1))
        case 2: #Worked well for losers
            model = Sequential()
            model.add(InputLayer(input_shape=(data_x_train.shape[1],1)))
            model.add(LSTM(64)); model.add(Dropout(0.2))
            model.add(Dense(units=16))
            model.add(Dense(units=1))
        case 3: #This worked well for both gainers and losers
            model = Sequential()
            model.add(InputLayer(input_shape=(data_x_train.shape[1],1)))
            model.add(LSTM(128)); model.add(Dropout(0.2))
            model.add(Dense(units=32))
            model.add(Dense(units=1))
        case 4:
            model = Sequential()
            model.add(InputLayer(input_shape=(data_x_train.shape[1],1)))
            model.add(LSTM(256, return_sequences=True)); model.add(Dropout(0.3))
            model.add(LSTM(256)); model.add(Dropout(0.3))
            model.add(Dense(64))
            model.add(Dense(32))
            model.add(Dense(16, activation="softmax"))
            model.add(Dense(units=1,activation="sigmoid"))
            opt = keras.optimizers.Adam(learning_rate=0.0002)
    return model

def plotlearn(history,type='loss'):
    plt.figure()
    plt.plot(history.history[type])
    plt.plot(history.history['val_' + type])
    plt.title('model '+type)
    plt.ylabel(type)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    return

# Plot a single graph
def candlefig():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.03, subplot_titles=('Price', 'Volume'), 
                row_width=[0.2, 0.7])
    return fig

def dmcandlefig():
    fig = make_subplots(rows=2, cols=2, specs=[[{"rowspan" : 2}, {}],[None, {}]], shared_xaxes=True, 
                 vertical_spacing=0.03, subplot_titles=('Daily Price', 'Minute Price', 'Volume'), 
                 row_width=[0.2, 0.7])
    return fig

def plotcandle(fig,data,ticker):
   fig.add_trace(go.Candlestick(x=data[ticker].index,
                open=data[ticker,'Open'], high=data[ticker,'High'],
                low=data[ticker,'Low'], close=data[ticker,'Close'], name=ticker), row=1, col=1)
   fig.add_trace(go.Bar(x=data[ticker].index, y=data[ticker,'Volume'], showlegend=False), row=2, col=1)
   us_holidays = pd.to_datetime(list(holidays.US(years=range(data.index[0].year, data.index[-1].year + 1)).keys()))
   us_holidays += pd.offsets.Hour(9) + pd.offsets.Minute(30)
   if (data.index[1]-data.index[0] < datetime.timedelta(days=1)):
      fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
         # NOTE: Below values are bound (not single values), ie. hide x to y
         dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
         dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
         dict(values=us_holidays)
         ]
      )
   else:
       fig.update_xaxes(
         rangeslider_visible=False,
         rangebreaks=[
          # NOTE: Below values are bound (not single values), ie. hide x to y
          dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
          dict(values=us_holidays)
          ]
       )
   fig.update_layout(
     title='Stock Analysis',
     yaxis_title=f' '+ticker+' Stock'
   )
   return

def plotcandlenow_log(data,log):
   fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.03, subplot_titles=('Price', 'Volume'), 
                row_width=[0.2, 0.7])
   fig.add_trace(go.Candlestick(x=data[log['Ticker']].index,
                open=data[log['Ticker'],'Open'], high=data[log['Ticker'],'High'],
                low=data[log['Ticker'],'Low'], close=data[log['Ticker'],'Close'], name=log['Ticker']), row=1, col=1)
   fig.add_trace(go.Bar(x=data[log['Ticker']].index, y=data[log['Ticker'],'Volume'], showlegend=False), row=2, col=1)
   us_holidays = pd.to_datetime(list(holidays.US(years=range(data.index[0].year, data.index[-1].year + 1)).keys()))
   us_holidays += pd.offsets.Hour(9) + pd.offsets.Minute(30)
   if (data.index[1]-data.index[0] < datetime.timedelta(days=1)):
      fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
         # NOTE: Below values are bound (not single values), ie. hide x to y
         dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
         dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
         dict(values=us_holidays)
         #dict(values=[x.strftime("%Y-%m-%d") for x in list(holidays.US(years=data[ticker].index[0].year).keys())])  # hide holidays (Christmas and New Year's, etc)
         ]
      )
   else:
       fig.update_xaxes(
         rangeslider_visible=False,
         rangebreaks=[
          # NOTE: Below values are bound (not single values), ie. hide x to y
          dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
          dict(values=us_holidays)
          ]
       )
   fig.update_layout(
     title='Stock Analysis: ' + str(log),
     yaxis_title=f' '+log['Ticker']+' Stock'
   )
   fig.show()
   return

def plotcandlenow(data,ticker):
   fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.03, subplot_titles=('Price', 'Volume'), 
                row_width=[0.2, 0.7])
   fig.add_trace(go.Candlestick(x=data[ticker].index,
                open=data[ticker,'Open'], high=data[ticker,'High'],
                low=data[ticker,'Low'], close=data[ticker,'Close'], name=ticker), row=1, col=1)
   #fig.add_trace(go.Scatter(x=data[ticker].index,y=data[ticker]['VWAP'],line_shape='spline', name='VWAP'), row=1,col=1)
   fig.add_trace(go.Bar(x=data[ticker].index, y=data[ticker,'Volume'], showlegend=False), row=2, col=1)
   us_holidays = pd.to_datetime(list(holidays.US(years=range(data.index[0].year, data.index[-1].year + 1)).keys()))
   us_holidays += pd.offsets.Hour(9) + pd.offsets.Minute(30)
   if (data.index[1]-data.index[0] < datetime.timedelta(days=1)):
      fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
         # NOTE: Below values are bound (not single values), ie. hide x to y
         dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
         dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
         dict(values=us_holidays)
         #dict(values=[x.strftime("%Y-%m-%d") for x in list(holidays.US(years=data[ticker].index[0].year).keys())])  # hide holidays (Christmas and New Year's, etc)
         ]
      )
   else:
       fig.update_xaxes(
         rangeslider_visible=False,
         rangebreaks=[
          # NOTE: Below values are bound (not single values), ie. hide x to y
          dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
          dict(values=us_holidays)
          ]
       )
   fig.update_layout(
     title='Stock Analysis',
     yaxis_title=f' '+ticker+' Stock'
   )
   fig.show()
   return

def plotdmcandle(fig,df,data,ticker):
   datday = df.loc[df.index.isin(pd.date_range(data[ticker].index[-1].date()-datetime.timedelta(6*30),
                                               data[ticker].index[-1].date()))]
   fig.add_trace(go.Candlestick(x=datday.index,
                open=datday[ticker,'Open'], high=datday[ticker,'High'],
                low=datday[ticker,'Low'], close=datday[ticker,'Close'], name=ticker), row=1, col=1)
   fig.add_trace(go.Candlestick(x=data[ticker].index,
                open=data[ticker,'Open'], high=data[ticker,'High'],
                low=data[ticker,'Low'], close=data[ticker,'Close'], name=ticker), row=1, col=2)
   #fig.add_trace(go.Scatter(x=data[ticker].index,y=data[ticker]['VWAP'],line_shape='spline', name='VWAP'), row=1,col=1)
   fig.add_trace(go.Bar(x=data[ticker].index, y=data[ticker,'Volume'], showlegend=False), row=2, col=2)
   us_holidays = pd.to_datetime(list(holidays.US(years=range(data.index[0].year, data.index[-1].year + 1)).keys()))
   us_holidays += pd.offsets.Hour(9) + pd.offsets.Minute(30)
   fig.layout['xaxis2'].update(
        rangeslider_visible=False,
        rangebreaks=[
         # NOTE: Below values are bound (not single values), ie. hide x to y
         dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
         dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
         dict(values=us_holidays)
         #dict(values=[x.strftime("%Y-%m-%d") for x in list(holidays.US(years=data[ticker].index[0].year).keys())])  # hide holidays (Christmas and New Year's, etc)
         ]
   )
   fig.layout['xaxis'].update(
         rangeslider_visible=False,
         rangebreaks=[
          # NOTE: Below values are bound (not single values), ie. hide x to y
          dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
          dict(values=us_holidays)
          ]
   )
   fig.update_layout(
     title='Stock Analysis',
     yaxis_title=f' '+ticker+' Stock'
   )
   return

def plotdmcandlenow(df,data,ticker):
   datday = df.loc[df.index.isin(pd.date_range(data[ticker].index[-1].date()-datetime.timedelta(6*30),
                                               data[ticker].index[-1].date()))]
   fig = make_subplots(rows=2, cols=2, specs=[[{"rowspan" : 2}, {}],[None, {}]], shared_xaxes=True, 
                vertical_spacing=0.03, subplot_titles=('Daily Price', 'Minute Price', 'Volume'), 
                row_width=[0.2, 0.7])
   fig.add_trace(go.Candlestick(x=datday.index,
                open=datday[ticker,'Open'], high=datday[ticker,'High'],
                low=datday[ticker,'Low'], close=datday[ticker,'Close'], name=ticker), row=1, col=1)
   fig.add_trace(go.Candlestick(x=data[ticker].index,
                open=data[ticker,'Open'], high=data[ticker,'High'],
                low=data[ticker,'Low'], close=data[ticker,'Close'], name=ticker), row=1, col=2)
   #fig.add_trace(go.Scatter(x=data[ticker].index,y=data[ticker]['VWAP'],line_shape='spline', name='VWAP'), row=1,col=1)
   fig.add_trace(go.Bar(x=data[ticker].index, y=data[ticker,'Volume'], showlegend=False), row=2, col=2)
   us_holidays = pd.to_datetime(list(holidays.US(years=range(data.index[0].year, data.index[-1].year + 1)).keys()))
   us_holidays += pd.offsets.Hour(9) + pd.offsets.Minute(30)
   fig.layout['xaxis2'].update(
        rangeslider_visible=False,
        rangebreaks=[
         # NOTE: Below values are bound (not single values), ie. hide x to y
         dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
         dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
         dict(values=us_holidays)
         #dict(values=[x.strftime("%Y-%m-%d") for x in list(holidays.US(years=data[ticker].index[0].year).keys())])  # hide holidays (Christmas and New Year's, etc)
         ]
   )
   fig.layout['xaxis'].update(
         rangeslider_visible=False,
         rangebreaks=[
          # NOTE: Below values are bound (not single values), ie. hide x to y
          dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
          dict(values=us_holidays)
          ]
   )
   fig.update_layout(
     title='Stock Analysis',
     yaxis_title=f' '+ticker+' Stock'
   )
   fig.show()
   return

def plotlinenow(data,ticker):
   fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.03, subplot_titles=('Price', 'Volume'), 
                row_width=[0.2, 0.7])
   fig.add_trace(go.Scatter(x=data[ticker].index,y=data[ticker]['VWAP'],line_shape='spline', name='VWAP'), row=1,col=1)
   fig.add_trace(go.Bar(x=data[ticker].index, y=data[ticker,'Volume'], showlegend=False), row=2, col=1)
   us_holidays = pd.to_datetime(list(holidays.US(years=range(data.index[0].year, data.index[-1].year + 1)).keys()))
   us_holidays += pd.offsets.Hour(9) + pd.offsets.Minute(30)
   if (data.index[1]-data.index[0] < datetime.timedelta(days=1)):
      fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
         # NOTE: Below values are bound (not single values), ie. hide x to y
         dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
         dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
         dict(values=us_holidays)
         #dict(values=[x.strftime("%Y-%m-%d") for x in list(holidays.US(years=data[ticker].index[0].year).keys())])  # hide holidays (Christmas and New Year's, etc)
         ]
      )
   else:
       fig.update_xaxes(
         rangeslider_visible=False,
         rangebreaks=[
          # NOTE: Below values are bound (not single values), ie. hide x to y
          dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
          dict(values=us_holidays)
          ]
       )
   fig.update_layout(
     title='Stock Analysis',
     yaxis_title=f' '+ticker+' Stock'
   )
   fig.show()
   return

def plotcandledata(data):
   fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.03, subplot_titles=('Price', 'Volume'), 
                row_width=[0.2, 0.7])
   fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close']), row=1, col=1)
   fig.add_trace(go.Bar(x=data.index, y=data['Volume'], showlegend=False), row=2, col=1)
   us_holidays = pd.to_datetime(list(holidays.US(years=range(data.index[0].year, data.index[-1].year + 1)).keys()))
   us_holidays += pd.offsets.Hour(9) + pd.offsets.Minute(30)
   if (data.index[1]-data.index[0] < datetime.timedelta(days=1)):
      fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
         # NOTE: Below values are bound (not single values), ie. hide x to y
         dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
         dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
         dict(values=us_holidays)
         #dict(values=[x.strftime("%Y-%m-%d") for x in list(holidays.US(years=data[ticker].index[0].year).keys())])  # hide holidays (Christmas and New Year's, etc)
         ]
      )
   else:
       fig.update_xaxes(
         rangeslider_visible=False,
         rangebreaks=[
          # NOTE: Below values are bound (not single values), ie. hide x to y
          dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
          #dict(values=us_holidays)
          ]
       )
   fig.update_layout(
     title='Stock Analysis'
   )
   fig.show()
   return

def plotbuysell():
    ax1.scatter(data.index[np.where(buysell==-1)],data['Close'].iloc[np.where(buysell==-1)],color='r',\
                marker=r'$\downarrow$',s=((10*amount[np.where(buysell==-1)])**2).astype(int))
    ax1.scatter(data.index[np.where(buysell==1)],data['Close'].iloc[np.where(buysell==1)],color='g',\
                marker=r'$\uparrow$',s=((10*amount[np.where(buysell==1)])**2).astype(int))
    ax1.fill_between(data.index,y1=np.min(data['Close'].values),\
                     y2=np.max(data['Close'].values),where=(buysell==-1),color='r',alpha=0.25)
    ax1.fill_between(data.index,y1=np.min(data['Close'].values),\
                  y2=np.max(data['Close'].values),where=(buysell==1),color='g',alpha=0.25)
    ax1.fill_between(data.index,y1=np.min(data['Close'].values),\
                  y2=np.max(data['Close'].values),where=(buysell==0),color='gray',alpha=0.5)
    return

def percprof(asset,time):
    monthprofit = np.zeros(len(profit)//time)
    for i in range(0,len(profit)//time):
       monthprofit[i] = (asset[(i+1)*time-1]-asset[i*time])/asset[i*time]*100
    return monthprofit

def plotbar():
    trade = percentprofit(price[0]+np.cumsum(profit)/np.mean(amount),30)
    plt.figure()
    plt.bar(np.arange(len(trade)),trade)
    return

def plothist():
    plt.hist(np.ndarray.flatten(gain),bins=100, density=True, alpha=0.6, color='g',edgecolor='black', linewidth=1.2)
    mu, std = scipy.stats.norm.fit(np.ndarray.flatten(gain))
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = scipy.stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.4f,  std = %.4f" % (mu, std)
    plt.title(title)

def findtopmovers(data,day,top,asc):
    Chg=(data.loc[day].unstack()['Open']-data.loc[prevday(data,day)].unstack()['Close'])\
        /data.loc[prevday(data,day)].unstack()['Close']
    # Select only movers with more than 5%
    Chg = Chg.loc[Chg < -0.05] if asc else Chg.loc[Chg > 0.05]
    return Chg.sort_values(ascending=asc)

def prevday(data,day):
    return data.index[data.index.get_loc(day)-1].strftime("%Y-%m-%d")