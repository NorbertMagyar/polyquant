# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:47:19 2024

@author: u0099498
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
import helpers
import modules

# Commission of buying/selling shares (in Euros)
comm = 3

def toplosers(data, ticker, day, log):
    takeprofit = 0.2
    amount = 10000
    stoplossprofit_trig = 0.03
    stoplossprofit = 0.0
    profit = 0.0
    mdata = helpers.dictdf(modules.client.get_aggregate_bars(
              ticker,day,day,multiplier=5,timespan='minute'))
    traded = False
    try :
       openprice = mdata.loc[mdata.index.strftime("%H:%M:%S") == "09:30:00",ticker]['Open'].values[0] 
       m5price = mdata.loc[mdata.index.strftime("%H:%M:%S") == "09:35:00",ticker]['Open'].values[0] 
    except IndexError:
       return
    else:
      #if (m5price-openprice)/openprice < -0.02:
      if False:
        return 
      else:
        buyprice = m5price
        stoploss =  (openprice-buyprice)/openprice - 0.02
        shares = int(amount/buyprice)
        for step in mdata.between_time("09:35", "15:55").index:
               # Prevent loss after profitable
               if mdata.loc[step,ticker]['Low'] <= buyprice + buyprice*stoploss:
                   profit = shares*buyprice*stoploss - 2*comm; traded = True; 
                   operation = 'Stop loss'; sellprice = buyprice + buyprice*stoploss
                   if traded == True: break;
               if mdata.loc[step,ticker]['High'] >= buyprice + buyprice*takeprofit:
                   profit = shares*buyprice*takeprofit - 2*comm; traded = True; 
                   operation = 'Take profit'; sellprice = buyprice + buyprice*takeprofit             
                   if traded == True: break; 
               # Dynamical adjustment of stop loss profit
               #if mdata.loc[step,ticker]['High'] >= buyprice + buyprice*stoplossprofit_trig:
               #    if stoplossprofit_trig/2 > stoplossprofit:
               #       stoplossprofit = stoplossprofit_trig/2
               #       stoplossprofit_trig = (mdata.loc[step,ticker]['High']-buyprice)/buyprice
               #       stoploss = stoplossprofit
               # Stop losses after profit more than 5%
               if mdata.loc[step,ticker]['High'] >= buyprice + buyprice*stoplossprofit_trig:
                   stoploss = 0.0
                    
        if traded == False: 
               profit = shares*(data.loc[day,ticker]['Close']\
                                              -buyprice) - 2*comm
               operation = 'End of trading day'; sellprice = data.loc[day,ticker]['Close']
             
        # Log trade 'Datetime','Ticker','Profit','Operation','Buyprice','SellPrice'
        log.loc[len(log)] = [step.date(),ticker,profit,operation,buyprice,sellprice] 
    return 
    
def buyit(profit): 
 profit = np.zeros(len(data))
 stoploss = 0.01
 for i in range(2,len(data)-1):
        if (data["Low"].iloc[i]-data["Close"].iloc[i-1])/data["Close"].iloc[i-1] >= -stoploss:
          profit[i] += (data["Close"].iloc[i]-data["Close"].iloc[i-1])*99./100.
        else:
          profit[i] += -stoploss*data["Close"].iloc[i-1]
 ax2.plot(data.index,np.cumsum(profit), label='buyit')
 return profit

def buysellrand(profit):
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
 return profit

def moving_av(profit):
    global buysell, unix_timestamp, month, days, amount, slope
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