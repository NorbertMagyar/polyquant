# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:51:42 2024

@author: u0099498
"""
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats
import scipy.signal
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pickle
import bs4 as bs
import requests
import datetime
import pytz
import sys
import modules
import holidays

timezone = pytz.timezone('America/New_York')

# Downloads daily candles from polygon.io for all S&P500 companies between specified dates
def downloaddata(start_date,end_date):
 resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
 soup = bs.BeautifulSoup(resp.text, 'lxml')
 table = soup.find('table', {'class': 'wikitable sortable'})
 tickers = []
 for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)
 
 tickers = [s.replace('\n', '') for s in tickers]
 
 start = datetime.datetime.strptime(start_date,"%Y-%m-%d")
 end = datetime.datetime.strptime(end_date,"%Y-%m-%d")
 data = [{}]*len(tickers)
 i = 0
 for ticker in tickers:
  data[i] = modules.client.get_aggregate_bars(ticker,start,end,multiplier=1,timespan='day')
  i += 1
  print('\rNo. '+str(i)+': '+ticker,end="")
  sys.stdout.write("\033[K")
  
 with open("sp500data-"+start_date+"-"+end_date+"-poly.pkl", 'wb') as f:
   pickle.dump(data, f)
  
 return data

def saveplots(data,ticker,day,name):
    fig = candlefig()
    mdata = dictdf(modules.client.get_aggregate_bars(
            ticker,data.index[data.index.get_loc(day)-1].strftime("%Y-%m-%d"),
            day,multiplier=5,timespan='minute'))
    plotcandle(fig, mdata, ticker)
    pio.write_image(fig, name+'.png',scale=6, width=1280, height=640)
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
       data[0]
     except IndexError:
       return 
     else:
       if datetime.datetime.fromtimestamp(data[0]['results'][0]['t']//1000,tz=timezone).hour == 0:
         timestr = "%Y-%m-%d"
       else:
         timestr = "%Y-%m-%d %H:%M:%S"
       for i in range(0,len(data)):

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

def findticker(data,ticker):
    for i in range(len(data)):
        if data[i]['ticker'] == ticker: return i

def makeplot():
   global fig, ax1, ax2
   fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
   ax1.plot(data.index,price)
   ax2.plot(data.index,(price-data['Close'].iloc[0])*99./100.)
   ax2.axhline(y=data['Close'].iloc[-1]-data['Close'].iloc[0],linewidth=4,color='r')
   ax2.axhline(y=0,linewidth=2,color='black')
   return

# Plot a single graph
def candlefig():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.03, subplot_titles=('Price', 'Volume'), 
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
    Chg = Chg.loc[Chg < -0.04] if asc else Chg.loc[Chg > 0.04]
    return Chg.sort_values(ascending=asc)[:top]

def prevday(data,day):
    return data.index[data.index.get_loc(day)-1].strftime("%Y-%m-%d")