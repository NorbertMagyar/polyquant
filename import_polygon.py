# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 23:18:49 2023

@author: u0099498
"""
from matplotlib import pyplot as plt
import numpy as np
import os.path
import yfinance as yf
import scipy.stats
import scipy.signal
from scipy.interpolate import UnivariateSpline
import time
from datetime import date, timedelta, datetime
from pytz import timezone
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import polygon
import pickle
from polygon import StocksClient
import savesp500data_polygon

pio.renderers.default='browser'
plt.rcParams["figure.figsize"] = [14, 7]
plt.rcParams.update({'font.size': 16})

def dictdf(data):
   b = ['Volume','VWAP','Open','Close','High','Low']
   for i in range(0,len(data)):
       index = [datetime.fromtimestamp(x['t']//1000,tz=tz).strftime("%Y-%m-%d") for x in data[i]['results']] 
       dfi = pd.DataFrame(data[i]['results'],index=index).drop(['t','n'],axis=1)
       a = [data[i]['ticker']] * (len(dfi.columns))
       dfi.columns = pd.MultiIndex.from_arrays([a,b])
       if (i==0):
         df = dfi
       else:
         df = df.join(dfi)
   return df

tz = timezone('EST')

# Define the tickers you want to download
tickers = ['^GSPC']

# Define the start and end date for the data
start_date = '2019-01-28'
end_date = '2024-01-24'
delta = timedelta(days=1)

client = polygon.StocksClient('JvxuTJB5r_gPuVKnV9fFhwFGjteLQHBo', connect_timeout=15)

if not os.path.isfile("sp500data-" + start_date +"-" + end_date +"-poly.pkl"):
    data = savesp500data_polygon.downloaddata(start_date,end_date)
else: 
    with open("sp500data-"+start_date+"-"+end_date+"-poly.pkl", 'rb') as f:
      data = pickle.load(f)

diff = (time.mktime(datetime.strptime(end_date, "%Y-%m-%d").timetuple())-\
time.mktime(datetime.strptime(start_date, "%Y-%m-%d").timetuple()))/3600/24/365

# Download the data using yfinance
sp500 = yf.download(tickers, start=start_date, end=end_date, interval='1d',progress=False)

def buyit():
 global profit  
 profit = np.zeros(len(data))
 stoploss = 0.01
 for i in range(2,len(data)-1):
        if (data["Low"].iloc[i]-data["Close"].iloc[i-1])/data["Close"].iloc[i-1] >= -stoploss:
          profit[i] += (data["Close"].iloc[i]-data["Close"].iloc[i-1])*99./100.
        else:
          profit[i] += -stoploss*data["Close"].iloc[i-1]
 ax2.plot(data.index,np.cumsum(profit), label='buyit')
 return

def buysellrand():
 global profit  
 profit = np.zeros(len(data))
 stoploss = 0.01
 buy = 0.03
 active = 0
 price = 0
 buysell = 0
 for i in range(2,len(data)-1):
        if active == 0:
            buysell = np.random.randint(2)
            price = data["Close"].iloc[i-1]
            active = 1
        if active == 1:
            if buysell == 1:
                if (data["Low"].iloc[i]-price)/price <= -stoploss:
                   profit[i] += -stoploss*price
                   active = 0
                if (data["High"].iloc[i]-price)/price >= buy:
                   profit[i] += buy*price*99./100.
                   active = 0 
            else:
                if (data["High"].iloc[i]-price)/price >= stoploss:
                  profit[i] += -stoploss*price
                  active = 0
                if (data["Low"].iloc[i]-price)/price <= -buy:
                  profit[i] +=  buy*price*99./100.
                  active = 0 
 ax2.plot(data.index,np.cumsum(profit), label='buysellrand')
 return

def moving_av():
    global profit, buysell, unix_timestamp, month, days, amount, slope
    #13 3 - AAPL, 0.1 LHA.DE, 
    #10 2 - short king
    rnl = 50
    rng = 10
    rns = 2
    slopelim = 0.2
    stoploss = 0.02
    profit = np.zeros(len(data));
    buysell = np.zeros(len(data));
    amount = np.zeros(len(data));
    slope = np.zeros(len(data));
    bought = 0; sold = 0;
    price = 0; month = np.zeros(len(data)); days = np.zeros(len(data));
    bla = np.array(data['Close'].index)
    unix_timestamp = (bla - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    for i in range(rnl,len(data)-1):
        extr = UnivariateSpline(unix_timestamp[i-rng:i],data['Close'].iloc[i-rng:i],k=1)
        month[i] = extr(unix_timestamp[i])
        extr = UnivariateSpline(unix_timestamp[i-rns:i],data['Close'].iloc[i-rns:i],k=1)
        days[i]  = extr(unix_timestamp[i])
        slope[i] = np.polyfit(np.arange(rnl)/rnl, (data['Close'].iloc[i-rnl:i]-data['Close'].iloc[i-rnl])/np.mean(data['Close'].iloc[i-rnl:i]), 1)[0]
        price = data["Close"].iloc[i-1]
        #bought pre-market on day i
        if ((month[i] > days[i]) & (slope[i] < slopelim)):
        #if (month[i] > days[i]):
             amount[i] = np.min((abs(month[i]-days[i])/price*100,50))
             sold = 1;
        if ((month[i] < days[i]) | (slope[i] >= slopelim)):
        #if (month[i] < days[i]):
             amount[i] = np.min((abs(month[i]-days[i])/price*100,50))
             bought = 1;
        #during the day
        if (((data["Low"].iloc[i] - price)/price < -stoploss) & (bought == 1)):
             profit[i] -= stoploss*amount[i]*price
             bought = 0;
        if (((price - data["High"].iloc[i])/price < -stoploss) & (sold == 1)):
             profit[i] -= stoploss*amount[i]*price
             sold = 0;  
        #end of the day (closing positions)
        if bought == 1: profit[i] += amount[i]*(data["Close"].iloc[i] - price)*99./100.;  buysell[i] = 1
        if sold == 1:   profit[i] += amount[i]*(price - data["Close"].iloc[i])*99./100.; buysell[i] = -1
        #keeping track of buying/selling
        sold = 0; bought = 0;
    ax2.plot(data.index,np.cumsum(profit)/np.mean(amount), label='moving_average')
    return 

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

def plotcandle(data,ticker):
   ind = findticker(data,ticker)
   fig = go.Figure(data=[go.Candlestick(x=data[ind].index,
                open=data['Open',ticker], high=data['High',ticker],
                low=data['Low',ticker], close=data['Close',ticker])
                     ])
   fig.update_xaxes(
     rangeslider_visible=True,
     rangebreaks=[
         # NOTE: Below values are bound (not single values), ie. hide x to y
         dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
         dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
         # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
     ]
   )
   fig.update_layout(
     title='Stock Analysis',
     yaxis_title=f' '+ticker+' Stock'
   )
   fig.show()
   return
def plotcandle1m(data,ticker):
   fig = go.Figure(data=[go.Candlestick(x=data.index,
                 open=data['Open'], high=data['High'],
                 low=data['Low'], close=data['Close'])
                      ])
   fig.update_xaxes(
     rangeslider_visible=True,
     rangebreaks=[
         # NOTE: Below values are bound (not single values), ie. hide x to y
         dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
         dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
         # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
     ]
   )
   fig.update_layout(
     title='Stock Analysis',
     yaxis_title=f' '+ticker+' Stock'
   )
   fig.show()
   return
def plotcandlesp500(data):
   fig = go.Figure(data=[go.Candlestick(x=data.index,
                 open=data['Open'], high=data['High'],
                 low=data['Low'], close=data['Close'])
                      ])
   fig.update_xaxes(
     rangeslider_visible=True,
     rangebreaks=[
         # NOTE: Below values are bound (not single values), ie. hide x to y
         dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
         # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
     ]
   )
   fig.update_layout(
     title='Stock Analysis',
     yaxis_title='SP500'
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

def findtopmovers(data,day,prevday,top,asc):
    Chg=(data.loc[day.strftime("%Y-%m-%d"),'Open']-data.loc[prevday.strftime("%Y-%m-%d"),'Close'])\
        /data.loc[prevday.strftime("%Y-%m-%d"),'Close']
    return Chg.sort_values(ascending=asc)[:top]

#day =  datetime.strptime(start_date,"%Y-%m-%d").date()
#endday = datetime.strptime(end_date,"%Y-%m-%d").date()
day =  datetime.strptime('2023-11-25',"%Y-%m-%d").date()
endday = datetime.strptime('2024-01-20',"%Y-%m-%d").date()
delta = timedelta(days=1)

numtop = 10
i = 0 
chg = np.zeros((data.shape[0],numtop))
gain = np.zeros((data.shape[0],numtop))
profit = np.zeros((data.shape[0],numtop))
#day += delta

while day <= endday:
    day += delta
    if pd.Timestamp(day) in data.index:
        print('\r'+day.strftime("%Y-%m-%d"),end="")
        prevday = day
        while (pd.Timestamp(prevday) not in data.index) or (prevday == day):
            prevday -= delta
        TopChg = findtopmovers(data,day,prevday,numtop,True)
        chg[i,:] = TopChg
        ii = 0
        for ticker in TopChg.index:
           takeprofit = 0.2
           stoploss =  0.01
           stoplossprofit = 0.05
           gain[i,ii] = (data.loc[day.strftime("%Y-%m-%d"),'Close'][ticker]-data.loc[day.strftime("%Y-%m-%d"),'Open'][ticker])\
           /data.loc[day.strftime("%Y-%m-%d"),'Open'][ticker]
           if (day-prevday) > timedelta(days=1):
               m1data = yf.download(ticker, start=prevday.strftime("%Y-%m-%d"), end=(day+delta).strftime("%Y-%m-%d"), interval='5m', progress=False)
           else:
               m1data = yf.download(ticker, start=(prevday-delta).strftime("%Y-%m-%d"), end=(day+delta).strftime("%Y-%m-%d"), interval='5m', progress=False)
           #plotcandle1m(m1data,ticker)
           traded = False
           buyprice = m1data.loc[day.strftime("%Y-%m-%d"),'Open'][0]
           shares = int(5000/buyprice)
           #for step in m1data.loc[day.strftime("%Y-%m-%d")].index: 
                #if m1data.loc[step,'High'] >= buyprice + buyprice*takeprofit:
                #    profit[i,ii] += shares*buyprice*takeprofit - 3; traded = True; 
                #    if traded == True: break;
                #if m1data.loc[step,'Low'] <= buyprice - buyprice*stoploss:
                #    profit[i,ii] -= shares*buyprice*stoploss + 3; traded = True; 
                #    if traded == True: break;
                #if m1data.loc[step,'High'] >= buyprice + buyprice*stoplossprofit:
                #    stoploss = -stoplossprofit
           if traded == False: profit[i,ii] += shares*(m1data.loc[day.strftime("%Y-%m-%d"),'Close'][-1]\
                                               -m1data.loc[day.strftime("%Y-%m-%d"),'Open'][0])
           ii+=1
    i+=1

plt.bar(np.arange((i-1)),np.sum(profit[:i-1,:],axis=1))
print("\n")
print(np.sum(profit))

#plothist()
#price = data['Close'].values
#makeplot()
#moving_av()
#plotbar()
#plotbuysell()
#buyit()
#buysellrand()
#plt.legend()
#alpha = profit/price - (price-np.roll(price,1))/price
#print("Total profit: %d%%"%(np.sum(profit)/price[0]*100))
#print("Month chi-sq: %d"%np.sum((month-data['Close'].values)**2))
#print("Days chi-sq: %d"%np.sum((days-data['Close'].values)**2))

#plt.plot(-0.5*(np.arange(10)),np.sum(profit,axis=1),label="stoploss = -"+str(100*stoploss)+"%")
#plt.title("Cumulative profit over %i years of trading %s, daily frequency. Avg. asset price: %i"%(int(diff),tickers[0],np.mean(data[["Close"]])))
#plt.ylabel("cumulative profit (buying and selling one share)")
#plt.xlabel("buying threshold (percentage)")