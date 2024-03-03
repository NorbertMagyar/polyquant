"""
Created on Fri Apr 14 23:18:49 2023

@author: Norbert Magyar
"""
from matplotlib import pyplot as plt
import numpy as np
import datetime
import os.path
import yfinance as yf
import pandas as pd
import pickle
from matplotlib.dates import date2num
import modules
import strats
import helpers
import time

# Initialize global variables
modules.initialize()

# Define general date range of interest
start_date = '2010-01-28'
end_date = '2024-01-24'

if not os.path.isfile("sp500data-" + start_date +"-" + end_date +"-poly.pkl"):
    data = helpers.downloaddata(start_date,end_date,api='yfinance')
    print("\nLoading data...")
    df = helpers.dictdf(data)
else: 
    try:
       df
    except NameError:
      with open("sp500data-"+start_date+"-"+end_date+"-poly.pkl", 'rb') as f:
        data = pickle.load(f)
        print("Loading data...")
        df = helpers.dictdf(data)
        
# Download the data using yfinance
sp500 = yf.download('^GSPC', start=start_date, end=end_date, interval='1d',progress=False)

# Define date range of interest for performing strategy analysis
# TIPS: Make sure starting day is not Saturday-Monday (otherwise there is no prevday in data)

start_day =  datetime.datetime.strptime('2024-01-20',"%Y-%m-%d").date()
end_day = datetime.datetime.strptime('2024-01-24',"%Y-%m-%d").date() 

# Number of top movers to consider
numtop = 3

# DataFrame log contains informaton about the executed trades
log = pd.DataFrame(columns=['Datetime','Ticker','Profit','Operation','Buyprice',
                            'SellPrice'])
log['Datetime'] = pd.to_datetime(log['Datetime'])

# Main loop of days
start_time = time.time()

datals = []; datagn =[]

for day in df.loc[start_day:end_day].index.strftime("%Y-%m-%d"):
    print('\r'+day,end="")  
    TopChg = helpers.findtopmovers(df,day,numtop,False)
    for ticker in TopChg.index:  
         print(ticker)
         #strats.toplosers(df, ticker, day, log)   
  
log.set_index('Datetime', inplace=True)
# For visualization of daily profits and tickers on mouse click 
cl = ['r' if p<0 else 'g' for p in log['Profit']] 

plt.figure()
xvals = [x-0.2*(-len(np.where(x == date2num(log.index))[0])+len(np.where(x ==\
        date2num(log.index)[item:])[0])) for item, x in enumerate(date2num(log.index))]
bars = plt.bar(xvals,log['Profit'],width=0.2,color=cl,edgecolor='black', picker=True)  
plt.gca().xaxis_date(); plt.tight_layout()

# Plot candleplot for clicked trade
def on_pick(evt):
    ind = bars.index(evt.artist)
    day = log.index[ind].strftime("%Y-%m-%d")
    mdata = helpers.dictdf(modules.client.get_aggregate_bars(
            log['Ticker'][ind],helpers.prevday(day),
            day,multiplier=5,timespan='minute'))
    helpers.plotcandlenow_log(mdata, log.iloc[ind])
    
plt.gcf().canvas.mpl_connect("pick_event", on_pick)

print("\n")
print('Final P&L: '+ str(int(np.sum(log['Profit']))) + '€ over ' +\
      str(round((end_day - start_day).days/30,2)) +' months, a total of '+\
      str(len(log))+' trades in ' +str(len(df.loc[start_day:end_day].index))+ ' trading days')
print('Average profit per month is: ' + str(int(np.sum(log['Profit'])/((end_day - start_day).days/30)))+'€')
print("Loop execution took %s seconds " % round((time.time() - start_time),2))