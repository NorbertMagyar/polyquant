# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:53:05 2024

@author: u0099498
"""
from matplotlib import pyplot as plt
import plotly.io as pio
import polygon
import datetime

# Initializes the system, including client, timezone, and other global settings
def initialize():
 #globally relevant variables
 global client, delta

 client = polygon.StocksClient('',read_timeout=60)
 
 delta = datetime.timedelta(days=1)

 pio.renderers.default='browser'
 plt.rcParams["figure.figsize"] = [14, 7]
 plt.rcParams.update({'font.size': 16})
 
 return
