

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from pathlib import Path
from pandas_datareader import data
from typing import List
from datetime import datetime


def get_dividend_data(symbol: str, file_name:str) -> pd.Series:
    temp_root: str = tempfile.gettempdir() + '/'
    file_path: str = temp_root + file_name
    temp_file_path = Path(file_path)
    file_size = 0
    if temp_file_path.exists():
        file_size = temp_file_path.stat().st_size

    if file_size > 0:
        dividend_data = pd.read_csv(file_path, index_col='Date')
        dividend_data = pd.Series(dividend_data[dividend_data.columns[0]])
    else:
        yfData = yf.Ticker(symbol)
        dividend_data: pd.Series = pd.Series(yfData.dividends)
        dividend_data.to_csv(file_path)
    return dividend_data


def get_market_data(file_name: str,
                    data_col: str,
                    symbols: str,
                    data_source: str,
                    start_date: datetime,
                    end_date: datetime) -> pd.Series:
    """
      file_name: the file name in the temp directory that will be used to store the data
      data_col: the type of data - 'Adj Close', 'Close', 'High', 'Low', 'Open', Volume'
      symbols: a list of symbols to fetch data for
      data_source: yahoo, etc...
      start_date: the start date for the time series
      end_date: the end data for the time series
      Returns: a Pandas DataFrame containing the data.

      If a file of market data does not already exist in the temporary directory, fetch it from the
      data_source.
    """
    temp_root: str = tempfile.gettempdir() + '/'
    file_path: str = temp_root + file_name
    temp_file_path = Path(file_path)
    file_size = 0
    if temp_file_path.exists():
        file_size = temp_file_path.stat().st_size

    if file_size > 0:
        close_data_df = pd.read_csv(file_path, index_col='Date')
        close_data: pd.Series = close_data_df[close_data_df.columns[0]]
    else:
        panel_data: pd.DataFrame = data.DataReader(symbols, data_source, start_date, end_date)
        close_data: pd.Series = pd.Series(panel_data[data_col])
        close_data.to_csv(file_path)
    return close_data


def get_dividend(symbol: str) -> pd.Series:
    dividend_file = f'{symbol}_dividends'
    dividends = get_dividend_data(symbol, dividend_file)
    if (len(dividends) > 0):
        div_date_series = pd.to_datetime( dividends.index )
        dividends.index = div_date_series
    return dividends


def get_dividend_return(symbol: str, dividends: pd.Series) -> pd.Series:
    if len(dividends) > 0:
        div_date_series = pd.to_datetime(dividends.index)
        end_date = div_date_series[len(div_date_series) - 1]
        end_year = end_date.year
        end_month = end_date.month
        start_date = datetime(end_year - 10, end_month, 1)
        start_date = max(start_date, div_date_series[0])
        data_source = 'yahoo'
        close_file = f'{symbol}_close'
        close_values = get_market_data(file_name=close_file,
                                       data_col='Close',
                                       symbols=symbol,
                                       data_source=data_source,
                                       start_date=start_date,
                                       end_date=end_date)
        close_date_series = pd.to_datetime(close_values.index)
        close_ix = close_date_series.isin(div_date_series)
        dividend_close = close_values[close_ix]
        dividend_ix = div_date_series.isin(close_date_series)
        dividends_adj = dividends[dividend_ix]
        percent = (dividends_adj.values / dividend_close.values).flatten().round(2)
        percent_series = pd.Series(percent)
        dividends_adj_index = dividends_adj.index
        percent_series.index = dividends_adj_index
    else:
        percent_series = pd.Series([])
    return percent_series


def get_yearly_return(dividend_ret: pd.Series) -> pd.Series:
    if len(dividend_ret) > 0:
        yearly_div: List = []
        div_dates: List = []
        div_value = dividend_ret[0]
        div_year = dividend_ret.index[0]
        for i in range(1, dividend_ret.shape[0]):
            if dividend_ret.index[i].year == div_year.year:
                div_value = div_value + dividend_ret[i]
                div_year = dividend_ret.index[i]
            else:
                yearly_div.append(div_value)
                div_dates.append(div_year)
                div_value = dividend_ret[i]
                div_year = dividend_ret.index[i]
        yearly_div.append(div_value)
        div_dates.append(div_year)
        yearly_div_series = pd.Series(yearly_div)
        yearly_div_series.index = div_dates
    else:
        yearly_div_series = pd.Series([])
    return yearly_div_series


def get_asset_yearly_ret(symbol: str) -> pd.Series:
    dividends = get_dividend(symbol)
    div_return = get_dividend_return(symbol, dividends)
    yearly_ret = get_yearly_return(div_return)
    return yearly_ret

symbols = ['FMG.AX', 'BHP', 'RIO', 'CEQP', 'ARCC', 'EVV', 'PTY',
           'NUSI', 'SUN.AX', 'WBK', 'WMB', 'XOM', 'WPC']

expense = {'FMG.AX': 0, 'BHP': 0, 'RIO': 0, 'CEQP': 0, 'ARCC': 0,
           'EVV': 0.0191, 'PTY': 0.0109, 'NUSI': 0.0068, 'SUN.AX': 0,
           'WBK': 0, 'WMB': 0, 'XOM': 0, 'WPC': 0, 'BHK': 0.0092}

dividend_dict: dict = dict()
for sym in symbols:
    yearly_ret = get_asset_yearly_ret(sym)
    dividend_dict[sym] = yearly_ret

dividend_dict_adj = dict()
for sym in symbols:
    dividend_dict_adj[sym] = dividend_dict[sym] - expense[sym]


def plot_bar(subplot, symbol, data, width) -> None:
    title = f'{symbol} Yearly Dividend Yield'
    subplot.set_ylabel('Percent')
    subplot.set_title(title)
    subplot.bar(data.index, data.values, width=width, color='blue')


trading_days = 253
num_assets = len(dividend_dict_adj)
keys = list(dividend_dict_adj)
i = 0
while i < num_assets - (num_assets % 2):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    sym1 = keys[i]
    sym2 = keys[1+1]
    data1 = dividend_dict_adj[sym1] * 100
    data2 = dividend_dict_adj[sym2] * 100
    plot_bar(ax1, sym1, data1, trading_days)
    plot_bar(ax2, sym2, data2, trading_days)
    plt.show()
    i = i + 2

if num_assets % 2 > 0:
    i = num_assets - 1
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    sym = keys[i]
    data = dividend_dict_adj[sym] * 100
    plot_bar(ax, sym, data, trading_days)
    plt.show()

# title = f'{sym} Dividend Yield'
# (yearly_ret * 100).plot(grid=True, figsize=(10,6), ylabel="Percent", title=title, kind='bar')

print("hi there")