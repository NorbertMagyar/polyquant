# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:28:12 2024
Find the S&P500 companies and then download historical daily candle data for a specified amount of days.
@author: u0099498
"""
import polygon
from polygon import StocksClient
import bs4 as bs
import requests
import datetime
import pandas as pd
import sys
import pickle
from datetime import date, timedelta, datetime

client = polygon.StocksClient('JvxuTJB5r_gPuVKnV9fFhwFGjteLQHBo', connect_timeout=15)

def downloaddata(start_date,end_date):
 global z
 resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
 soup = bs.BeautifulSoup(resp.text, 'lxml')
 table = soup.find('table', {'class': 'wikitable sortable'})
 tickers = []
 for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)
 
 tickers = [s.replace('\n', '') for s in tickers]
 
 start = datetime.strptime(start_date,"%Y-%m-%d").date()
 end = datetime.strptime(end_date,"%Y-%m-%d").date()
 data = [{}]*len(tickers)
 i = 0
 for ticker in tickers:
  data[i] = client.get_aggregate_bars(ticker,start,end,multiplier=1,timespan='day')
  i += 1
  print('\rNo. '+str(i)+': '+ticker,end="")
  sys.stdout.write("\033[K")
  
 with open("sp500data-"+start_date+"-"+end_date+"-poly.pkl", 'wb') as f:
   pickle.dump(data, f)
  
 return data
