

import yfinance as yf
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from pandas_datareader import data
from typing import List
from datetime import datetime
from forex_python.converter import CurrencyRates


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
        percent = (dividends_adj.values / dividend_close.values).flatten().round(5)
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
           'NUSI', 'SUN.AX', 'WBK', 'WMB', 'XOM', 'WPC', 'BHK']

expense = {'FMG.AX': 0, 'BHP': 0, 'RIO': 0, 'CEQP': 0, 'ARCC': 0,
           'EVV': 0.0191, 'PTY': 0.0109, 'NUSI': 0.0068, 'SUN.AX': 0,
           'WBK': 0, 'WMB': 0, 'XOM': 0, 'WPC': 0, 'BHK': 0.0092}

c = CurrencyRates()
aux_to_dollar = c.get_rate('AUD', 'USD')

exchange_adj = {'FMG.AX': aux_to_dollar, 'BHP': 1, 'RIO': 1, 'CEQP': 1, 'ARCC': 1,
                'EVV': 1, 'PTY': 1, 'NUSI': 1, 'SUN.AX': aux_to_dollar,
                 'WBK': 1, 'WMB': 1, 'XOM': 1, 'WPC': 1, 'BHK': 1}

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
    sym2 = keys[i+1]
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

mean_return = dict()
for sym in list(dividend_dict_adj):
    m = dividend_dict_adj[sym].mean()
    mean_return[sym] = m

percent_sum = sum(list(mean_return.values()))
portfolio_percent = dict()
for sym in list(mean_return):
    percent = round(mean_return[sym] / percent_sum, 4)
    portfolio_percent[sym] = percent

portfolio_percent_df = pd.DataFrame( list(portfolio_percent.values()) )
portfolio_percent_df.index = list(portfolio_percent.keys())

print("Portfolio Percentage")
print(tabulate(portfolio_percent_df * 100, headers=['Symbol', 'Portfolio Percent'], tablefmt='fancy_grid'))

capital = 220000
allocation_df = portfolio_percent_df * capital
print(tabulate(allocation_df, headers=['Symbol', 'Dollar Allocation'], tablefmt='fancy_grid'))

print("Fetching current stock prices...")
prices = yf.download(symbols, period='1d', interval='1d')
print()

prices_low = prices['Low']
prices_high = prices['High']
prices_mid = round((prices_low + prices_high) / 2, 2)

prices_mid_adj = pd.DataFrame()
for sym in prices_mid.columns:
    prices_mid_adj[sym] = prices_mid[sym] * exchange_adj[sym]

print(f"Share prices as of {prices_mid_adj.index[0]}")
print(tabulate(prices_mid_adj.transpose(), headers=['Symbol', 'Share Price'], tablefmt='fancy_grid'))

allocation_df_t = allocation_df.transpose()
shares_df = pd.DataFrame()
for sym in allocation_df_t.columns:
    shares_df[sym] = (allocation_df_t[sym].values // prices_mid_adj[sym].values).flatten()

print(tabulate(shares_df.transpose(), headers=['Symbol', 'Number of Shares'], tablefmt='fancy_grid'))

invested_total = 0
for sym in shares_df.columns:
    asset_val = shares_df[sym].values * prices_mid_adj[sym]
    invested_total = invested_total + asset_val

invested_total_df = pd.DataFrame(invested_total)
print(tabulate(invested_total_df, headers=['', 'Total Invested'], tablefmt='fancy_grid'))

portfolio_return_df = pd.DataFrame()
portfolio_percent_df_t = portfolio_percent_df.transpose()
for sym in portfolio_percent_df.index:
    adj_return = mean_return[sym] * portfolio_percent_df_t[sym]
    portfolio_return_df[sym] = adj_return

total_return = round(portfolio_return_df.values.sum(), 4) * 100
total_return_df = pd.DataFrame([total_return])
print(tabulate(total_return_df, headers=['', 'Estimated Dividend Return'], tablefmt='fancy_grid'))

print("hi there")