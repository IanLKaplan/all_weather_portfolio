
from datetime import datetime, timedelta

import matplotlib
from tabulate import tabulate
from typing import List, Tuple
from pandas_datareader import data
import pypfopt as pyopt
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt import plotting, CLA
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import quantstats as qs



plt.style.use('seaborn-whitegrid')


"""
  Ideally this function would go in a local package. However, I want this Jupyter notebook
  to display on github and I don't know of a way to get the local package import install
  and import to work.
"""

def get_market_data(file_name: str,
                    data_col: str,
                    symbols: List,
                    data_source: str,
                    start_date: datetime,
                    end_date: datetime) -> pd.DataFrame:
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
        close_data = pd.read_csv(file_path, index_col='Date')
    else:
        panel_data: pd.DataFrame = data.DataReader(symbols, data_source, start_date, end_date)
        close_data: pd.DataFrame = panel_data[data_col]
        close_data.to_csv(file_path)
    return close_data



data_source = 'yahoo'
# yyyy-mm-dd
start_date_str = '2010-01-01'
start_date: datetime = datetime.fromisoformat(start_date_str)
end_date: datetime = datetime.today() - timedelta(days=1)




etf_symbols = ["VTI", "VGLT", "VGIT", "VPU", "IAU"]

# Fetch the close prices for the entire time period
#
etf_close_file = 'etf_close'
# Fetch all of the close prices: this is faster than fetching only the dates needed.
etf_close: pd.DataFrame = get_market_data(file_name=etf_close_file,
                                          data_col='Close',
                                          symbols=etf_symbols,
                                          data_source=data_source,
                                          start_date=start_date,
                                          end_date=end_date)

etf_weights = {"VTI": 0.30, "VGLT": 0.40, "VGIT": 0.15, "VPU": 0.08, "IAU": 0.07}

prices = etf_close[0:1]


def calc_portfolio_holdings(initial_investment: int, weights: pd.DataFrame, prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the initial portfolio holdings given am amount of cash to invest.
    :param initial_investment: The initial investment used to purchase the portfolio (no partial shares)
    :param weights: a data frame containing the the weights for each asset as symbol: fraction
    :param prices: the share prices
    :return: the dollar value of the share holdings and the number of shares
    """
    weights_np: np.array = np.zeros(weights.shape[1])
    prices_np: np.array = np.zeros(weights.shape[1])
    for ix, col in enumerate(weights.columns):
        weights_np[ix] = weights[col]
        prices_np[ix] = prices[col]
    budget_np = weights_np * float(initial_investment)
    shares = budget_np // prices_np
    holdings = shares * prices_np
    holdings_df: pd.DataFrame = pd.DataFrame(holdings).transpose()
    holdings_df.columns = weights.columns
    shares_df: pd.DataFrame = pd.DataFrame(shares).transpose()
    shares_df.columns = weights.columns
    return holdings_df, shares_df


def simple_return(time_series: List, period: int) -> List :
    return list(((time_series[i]/time_series[i-period]) - 1.0 for i in range(period, len(time_series), period)))


def return_df(time_series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a data frame consisting of price time series, return a data frame
    that consists of the simple returns for the time series.  The returned data
    frame will have the same columns, but the time index will be one time period
    less.
    """
    r_df: pd.DataFrame = pd.DataFrame()
    col_names = time_series_df.columns
    for col in col_names:
        col_vals = time_series_df[col]
        col_ret = simple_return(col_vals, 1)
        r_df[col] = col_ret
    index = time_series_df.index
    return r_df.set_index(index[1:len(index)])


# initial investment, in dollars
initial_investment = 10000

weights_df: pd.DataFrame = pd.DataFrame(etf_weights.values()).transpose()
weights_df.columns = etf_weights.keys()

holdings, shares = calc_portfolio_holdings(initial_investment=initial_investment,
                                           weights=weights_df,
                                           prices=prices)

print("Portfolio weights:")
print(tabulate(weights_df, headers=['', *weights_df.columns], tablefmt='fancy_grid'))
print("Number of Shares:")
print(tabulate(shares, headers=['As of Date', *shares.columns], tablefmt='fancy_grid'))
print(f'Total invested from {initial_investment} is {int(holdings.sum(axis=1))}')

print("Value of share holdings:")
print(tabulate(holdings, headers=['As of Date', *holdings.columns], tablefmt='fancy_grid'))

trading_days = 253
days_in_quarter = trading_days // 4
half_year = trading_days // 2

def do_rebalance(cash_holdings: pd.DataFrame, weights_df: pd.DataFrame, portfolio_range: float) -> bool:
    """
    cash_holdings: a data frame with the amount of cash for each holding.
    weights: a data frame containing the portfolio target weights
    range: the portfolio should be rebalanced if an asset is not +/- percent of the target weight
    """
    total_cash: int = int(cash_holdings.sum(axis=1))
    current_percent = round(cash_holdings.div(total_cash), 3)
    # Calculate the allowed weight range for the portfolio
    weight_range = weights_df.apply(lambda x: (x - (x * portfolio_range), x * (1 + portfolio_range)), axis=0)
    # are any of the current portfolio weights outside of the allowed portfolio range
    rebalance = list(float(current_percent[col]) < float(weight_range[col][0]) or float(current_percent[col]) > float(weight_range[col][1]) for col in current_percent.columns)
    return any(rebalance)



def portfolio_rebalance(cash_holdings: pd.DataFrame,
                        weights: pd.DataFrame,
                        portfolio_range: float,
                        prices: pd.DataFrame) -> pd.DataFrame:
    """
    cash_holdings: a data frame containing the current portfolio cash holdings by stock
    weights: a data frame containing the portfolio weights
    range: the +/- percentage range for rebalancing (e.g., +/- 0.05)
    prices: the market prices on the day the portfolio will be rebalanced.
    Return: rebalanced cash holdings
    """
    new_holdings = cash_holdings
    if do_rebalance(cash_holdings=cash_holdings, weights_df=weights, portfolio_range=portfolio_range):
        total_cash = cash_holdings.sum(axis=1)
        new_holdings, shares = calc_portfolio_holdings(initial_investment=total_cash,
                                                       weights=weights,
                                                       prices=prices)
    return new_holdings


etf_adj_close_file = 'etf_adj_close'
# Fetch the adjusted close price for the unleveraged "all weather" set of ETFs'
# VTI, VGLT, VGIT, VPU and IAU
etf_adj_close: pd.DataFrame = get_market_data(file_name=etf_adj_close_file,
                                          data_col="Adj Close",
                                          symbols=etf_symbols,
                                          data_source=data_source,
                                          start_date=start_date,
                                          end_date=end_date)

# +/- for each asset for portfolio rebalancing
portfolio_range = 0.05

returns = return_df(etf_adj_close)

portfolio_np = np.zeros(etf_close.shape, dtype=np.float64)
portfolio_total_np = np.zeros(etf_close.shape[0])
portfolio_total_np[0] = holdings.sum(axis=1)
# initialize the first row with the dollar value of the portfolio holdings.
portfolio_np[0,] = holdings

for t in range(1, portfolio_np.shape[0]):
    for col, stock in enumerate(holdings.columns):
        portfolio_np[t, col] = portfolio_np[t-1, col] + (portfolio_np[t-1, col] * returns[stock][t-1])
        portfolio_total_np[t] = portfolio_total_np[t] + portfolio_np[t,col]


portfolio_total_df: pd.DataFrame = pd.DataFrame(portfolio_total_np, index=etf_close.index, columns=['Portfolio'])
# portfolio_total_df.plot(title="Portfolio Value (without rebalancing)", grid=True, figsize=(10,8))

portfolio_total_np = portfolio_np.sum(axis=1)
portfolio_percent_np = portfolio_np / portfolio_total_np[:,None]
market_percent = portfolio_percent_np[:,0]
market_percent_df: pd.DataFrame = pd.DataFrame(market_percent, index=etf_close.index, columns=['Market'])
# market_percent_df.plot(title="Percentage of Portfolio invested in the Market", grid=True, figsize=(10,8))


def calc_rebalanced_portfolio(holdings: pd.DataFrame,
                            etf_close: pd.DataFrame,
                            returns: pd.DataFrame,
                            weights: pd.DataFrame,
                            rebalance_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    portfolio_np = np.zeros(etf_close.shape, dtype=np.float64)
    portfolio_total_np = np.zeros(etf_close.shape[0])
    portfolio_total_np[0] = holdings.sum(axis=1)
    portfolio_np[0,] = holdings

    for t in range(1, portfolio_np.shape[0]):
        for col, stock in enumerate(holdings.columns):
            portfolio_np[t, col] = portfolio_np[t-1, col] + (portfolio_np[t-1, col] * returns[stock][t-1])
            portfolio_total_np[t] = portfolio_total_np[t] + portfolio_np[t,col]
        if (t % rebalance_days) == 0:
            current_holdings: pd.DataFrame = pd.DataFrame( portfolio_np[t,]).transpose()
            current_holdings.columns = holdings.columns
            date = etf_close.index[t]
            current_holdings.index = [date]
            close_prices_t = etf_close[t:t+1]
            portfolio_np[t, ] = portfolio_rebalance(cash_holdings=current_holdings,
                                                    weights=weights,
                                                    portfolio_range=portfolio_range,
                                                    prices=close_prices_t)
    portfolio_df: pd.DataFrame = pd.DataFrame(portfolio_np, index=etf_close.index, columns=etf_close.columns)
    portfolio_total_df: pd.DataFrame = pd.DataFrame(portfolio_total_np, index=etf_close.index)
    portfolio_total_df.columns = ['Portfolio Value']
    return portfolio_df, portfolio_total_df


def plot_portfolio_weights(asset_values_df: pd.DataFrame, portfolio_total_df: pd.DataFrame) -> None:
    portfolio_total_np = np.array(portfolio_total_df)
    asset_values_np = np.array(asset_values_df)
    portfolio_percent_np = asset_values_np / portfolio_total_np

    col_names = asset_values_df.columns
    fig, ax = plt.subplots(asset_values_np.shape[1], figsize=(10,8))
    for col in range(0, asset_values_np.shape[1]):
        asset_percent = portfolio_percent_np[:,col]
        label = f'Portfolio Percent for {col_names[col]}'
        ax[col].set_xlabel(label)
        ax[col].grid(True)
        ax[col].plot(asset_percent)
    fig.tight_layout()
    plt.show()


portfolio_quarterly_df, portfolio_total_quarterly_df = calc_rebalanced_portfolio(holdings=holdings,
                                                             etf_close=etf_close,
                                                             returns=returns,
                                                             weights=weights_df,
                                                             rebalance_days=days_in_quarter)

# portfolio_total_quarterly_df.plot(title="Portfolio Value (quarterly rebalanced)", grid=True, figsize=(10,8))

print(tabulate(weights_df, headers=['', *weights_df.columns], tablefmt='fancy_grid'))

# plot_portfolio_weights(portfolio_quarterly_df, portfolio_total_quarterly_df)

portfolio_biannual_df, portfolio_total_biannual_df = calc_rebalanced_portfolio(holdings=holdings,
                                                             etf_close=etf_close,
                                                             returns=returns,
                                                             weights=weights_df,
                                                             rebalance_days=half_year)

# portfolio_total_biannual_df.plot(title="Portfolio Value (rebalanced twice a year)", grid=True, figsize=(10,8))

# plot_portfolio_weights(portfolio_biannual_df, portfolio_total_biannual_df)

portfolio_yearly_df, portfolio_total_yearly_df = calc_rebalanced_portfolio(holdings=holdings,
                                                             etf_close=etf_close,
                                                             returns=returns,
                                                             weights=weights_df,
                                                             rebalance_days=trading_days)

# portfolio_total_yearly_df.plot(title="Portfolio Value (rebalanced once a year)", grid=True, figsize=(10,8))

# plot_portfolio_weights(portfolio_yearly_df, portfolio_total_yearly_df)

portfolio_quarterly_return = return_df(portfolio_total_quarterly_df)
portfolio_biannual_return = return_df(portfolio_total_biannual_df)
portfolio_yearly_return = return_df(portfolio_total_yearly_df)

portfolio_quarterly_sd = np.std( portfolio_quarterly_return ) * np.sqrt(portfolio_quarterly_return.shape[0])
portfolio_biannual_sd = np.std( portfolio_biannual_return ) * np.sqrt(portfolio_biannual_return.shape[0])
portfolio_yearly_sd = np.std( portfolio_yearly_return ) * np.sqrt(portfolio_yearly_return.shape[0])

sd_df = pd.DataFrame([portfolio_quarterly_sd, portfolio_biannual_sd, portfolio_yearly_sd]).transpose()


print(tabulate(sd_df, headers=['stddev', 'quarterly', 'bi-annual', 'yearly'], tablefmt='fancy_grid'))


market = 'SPY'
spy_close_file = 'spy_adj_close'
spy_adj_close = get_market_data(file_name=spy_close_file,
                                data_col='Adj Close',
                                symbols=[market],
                                data_source=data_source,
                                start_date=start_date,
                                end_date=end_date)

spy_close_start_file = 'spy_close'
spy_close = get_market_data(file_name=spy_close_start_file,
                                data_col='Close',
                                symbols=[market],
                                data_source=data_source,
                                start_date=start_date,
                                end_date=end_date)

spy_initial_price = spy_close[market][0]

market_return = return_df(spy_adj_close)

def calc_market_portfolio(market_return_df: pd.DataFrame,
                          date_index: pd.Index,
                          initial_investment: int,
                          initial_market_price: float ) -> pd.DataFrame:
    market_portfolio_np: np.array = np.zeros(len(date_index))
    market_portfolio_np[0] = (initial_investment // initial_market_price) * initial_market_price
    for i in range(1, market_portfolio_np.shape[0]):
        market_portfolio_np[i] = market_portfolio_np[i-1] + (market_portfolio_np[i-1] * market_return_df[market][i-1])
    market_portfolio_df: pd.DataFrame = pd.DataFrame(market_portfolio_np, index=date_index, columns=[market])
    return market_portfolio_df


market_portfolio_df = calc_market_portfolio(market_return_df=market_return,
                                            date_index=spy_adj_close.index,
                                            initial_investment=initial_investment,
                                            initial_market_price=spy_initial_price)


portfolios_df: pd.DataFrame = pd.concat([portfolio_total_yearly_df, market_portfolio_df], axis=1)

# portfolios_df.plot(title="Portfolio Value (rebalanced once a year) + SPY", grid=True, figsize=(10,8))

forty_sixty_weights_df = pd.DataFrame([0.40, 0.44, 0.16]).transpose()
forty_sixty_weights_df.columns = ['VTI', 'VGLT', 'VGIT']
forty_sixty_holdings, forty_sixty_shares = calc_portfolio_holdings(initial_investment=initial_investment,
                                           weights=forty_sixty_weights_df,
                                           prices=prices)

portfolio_forty_sixty_df, portfolio_total_forty_sixty_df = calc_rebalanced_portfolio(holdings=forty_sixty_holdings,
                                                             etf_close=etf_close[forty_sixty_weights_df.columns],
                                                             returns=returns[forty_sixty_weights_df.columns],
                                                             weights=forty_sixty_weights_df,
                                                             rebalance_days=trading_days)

portfolio_total_forty_sixty_df.columns = ['40/60 Portfolio']
portfolio_total_yearly_df.columns = ['All Weather']
portfolios_df: pd.DataFrame = pd.concat([portfolio_total_forty_sixty_df, portfolio_total_yearly_df], axis=1)

print("40/60 Portfolio weights:")
print(tabulate(forty_sixty_weights_df, headers=['', *forty_sixty_weights_df.columns], tablefmt='fancy_grid'))

# portfolios_df.plot(title="40/60 Portfolio + All Weather Portfolio", grid=True, figsize=(10,8))
portfolio_sixty_forty_return = return_df(portfolio_total_forty_sixty_df)
portfolio_sixty_forty_sd = np.std( portfolio_sixty_forty_return ) * np.sqrt(portfolio_sixty_forty_return.shape[0])
sd_df = pd.DataFrame([np.array(portfolio_sixty_forty_sd), np.array(portfolio_yearly_sd)]).transpose()
print("Daily Return Portfolio volatility")
print(tabulate(sd_df, headers=['stddev', '60/40', 'All Weather'], tablefmt='fancy_grid'))

aom_adj_close_file = 'aom_adj_close'
aom_adj_close: pd.DataFrame = get_market_data(file_name=aom_adj_close_file,
                                          data_col="Adj Close",
                                          symbols=['AOM'],
                                          data_source=data_source,
                                          start_date=start_date,
                                          end_date=end_date)

aom_returns = np.array(return_df(aom_adj_close))

aom_total_np = np.zeros(aom_adj_close.shape[0], dtype=np.float64)
aom_total_np[0] = initial_investment
for t in range(1, aom_total_np.shape[0]):
    aom_total_np[t] = aom_total_np[t-1] + (aom_total_np[t-1] * aom_returns[t-1])

aom_total_df: pd.DataFrame = pd.DataFrame( aom_total_np )
aom_total_df.index = aom_adj_close.index
aom_total_df.columns = ['AOM']

aom_sd = np.std( aom_returns ) * np.sqrt(aom_returns.shape[0])
sd_df = pd.DataFrame([aom_sd, portfolio_sixty_forty_sd]).transpose()
print("Daily Return Portfolio volatility")
print(tabulate(sd_df, headers=['stddev', 'AOM', '40/60'], tablefmt='fancy_grid'))

portfolios_df: pd.DataFrame = pd.concat([aom_total_df, portfolio_total_forty_sixty_df], axis=1)
# portfolios_df.plot(title="AOM + 40/60 Portfolio", grid=True, figsize=(10,8))


def period_return(portfolio_total_df: pd.DataFrame, period: int) -> pd.DataFrame:
    period_range: list = list(t for t in range(portfolio_total_df.shape[0]-1, -1, -period))
    period_range.reverse()
    portfolio_total_np: np.array = np.array(portfolio_total_df)
    period_ret_l: list = []
    period_dates_l: list = []
    for ix in range(1, len(period_range)):
        ret = (portfolio_total_np[ period_range[ix] ] / portfolio_total_np[ period_range[ix-1] ]) - 1
        ret = float((ret * 100).round(2))
        period_ret_l.append(ret)
        period_dates_l.append(portfolio_total_df.index[period_range[ix]])

    period_ret_df: pd.DataFrame = pd.DataFrame(period_ret_l, index=period_dates_l)
    period_ret_df.columns = ['Return']
    return period_ret_df


def plot_return(ret_df: pd.DataFrame, title:str) -> None:
    # Point and line plot of return values
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    fig.suptitle(title)
    fig.autofmt_xdate()
    plt.ylabel('Percent')
    ax.plot(ret_df, '-bo')
    plt.show()
    # Plot a table return values
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    the_table = plt.table(cellText=ret_df.values, rowLabels=ret_df.index, loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(14)
    plt.show()

annual_ret_df = period_return(portfolio_total_forty_sixty_df, trading_days)
# plot_return(annual_ret_df, '40/60 Portfolio Annual Return')

mean_return_df = pd.DataFrame(annual_ret_df.mean(axis=0))
print(tabulate(mean_return_df, headers=['', 'Mean Annual Return'], tablefmt='fancy_grid'))


def calc_asset_beta(asset_df: pd.DataFrame, market_df: pd.DataFrame) -> float:
    asset_np = np.array(asset_df).flatten()
    market_np = np.array(market_df).flatten()
    sd_asset = np.std(asset_np)
    sd_market = np.std(market_np)
    # cor_tuple: correlation and p-value
    cor_tuple = stats.pearsonr(asset_np, market_np)
    cor = cor_tuple[0]
    beta = cor * (sd_asset/sd_market)
    return beta


portfolio_beta = calc_asset_beta(portfolio_sixty_forty_return, market_return)
vti_beta = calc_asset_beta(returns['VTI'], market_return)
vglt_beta = calc_asset_beta(returns['VGLT'], market_return)
beta_np = np.array([vti_beta, vglt_beta, portfolio_beta])
beta_df: pd.DataFrame = pd.DataFrame(beta_np).transpose()

print(tabulate(beta_df, headers=['', 'VTI Beta', 'VGLT Beta', '40/60 Portfolio'], tablefmt='fancy_grid'))


# 13-week yearly treasury bond quote
risk_free_asset = '^IRX'

rf_file_name = 'rf_adj_close'
# The bond return is reported as a yearly return percentage
rf_adj_close = get_market_data(file_name=rf_file_name,
                                data_col='Adj Close',
                                symbols=[risk_free_asset],
                                data_source=data_source,
                                start_date=start_date,
                                end_date=end_date)



# rf_adj_close.plot(title="Yield on the 13-week T-bill", grid=True, figsize=(10,8), ylabel='yearly percent')

rf_adj_rate_np: np.array = np.array( rf_adj_close.values ) / 100
rf_daily_np = ((1 + rf_adj_rate_np) ** (1/360)) - 1
rf_daily_df: pd.DataFrame = pd.DataFrame( rf_daily_np, index=rf_adj_close.index, columns=['^IRX'])
# rf_daily_df.plot(title="Daily yield on the 13-week T-bill", grid=True, figsize=(10,8), ylabel='daily yield x $10^{-5}$')


def adjust_time_series(ts_one_df: pd.DataFrame, ts_two_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adjust two DataFrame time series with overlapping date indices so that they
    are the same length with the same date indices.
    """
    ts_one_index = pd.to_datetime(ts_one_df.index)
    ts_two_index = pd.to_datetime(ts_two_df.index)
        # filter the close prices
    matching_dates = ts_one_index.isin( ts_two_index )
    ts_one_adj = ts_one_df[matching_dates]
    # filter the rf_prices
    ts_one_index = pd.to_datetime(ts_one_adj.index)
    matching_dates = ts_two_index.isin(ts_one_index)
    ts_two_adj = ts_two_df[matching_dates]
    return ts_one_adj, ts_two_adj

def excess_return_series(asset_return: pd.Series, risk_free: pd.Series) -> pd.DataFrame:
    excess_ret = asset_return.values.flatten() - risk_free.values.flatten()
    excess_ret_df = pd.DataFrame(excess_ret, index=asset_return.index)
    return excess_ret_df


def excess_return_df(asset_return: pd.DataFrame, risk_free: pd.Series) -> pd.DataFrame:
    excess_df: pd.DataFrame = pd.DataFrame()
    for col in asset_return.columns:
        e_df = excess_return_series(asset_return[col], risk_free)
        e_df.columns = [col]
        excess_df[col] = e_df
    return excess_df

def calc_sharpe_ratio(asset_return: pd.DataFrame, risk_free: pd.Series, period: int) -> pd.DataFrame:
    excess_return = excess_return_df(asset_return, risk_free)
    return_mean: List = []
    return_stddev: List = []
    for col in excess_return.columns:
        mu = np.mean(excess_return[col])
        std = np.std(excess_return[col])
        return_mean.append(mu)
        return_stddev.append(std)
    # daily Sharpe ratio
    # https://quant.stackexchange.com/questions/2260/how-to-annualize-sharpe-ratio
    sharpe_ratio = (np.asarray(return_mean) / np.asarray(return_stddev)) * np.sqrt(period)
    result_df: pd.DataFrame = pd.DataFrame(sharpe_ratio).transpose()
    result_df.columns = asset_return.columns
    ix = asset_return.index
    dateformat = '%Y-%m-%d'
    ix_start = datetime.strptime(ix[0], dateformat).date()
    ix_end = datetime.strptime(ix[len(ix)-1], dateformat).date()
    index_str = f'{ix_start} : {ix_end}'
    result_df.index = [ index_str ]
    return result_df


ret_adj_df, rf_daily_adj = adjust_time_series(portfolio_sixty_forty_return, rf_daily_df)
sharpe_ratio_sixty_forty = calc_sharpe_ratio(ret_adj_df, rf_daily_adj, trading_days)
ret_adj_df, rf_daily_adj = adjust_time_series(market_return, rf_daily_df)
sharpe_ratio_market = calc_sharpe_ratio(ret_adj_df, rf_daily_adj, trading_days)

sharpe_df = pd.concat([sharpe_ratio_sixty_forty, sharpe_ratio_market], axis=1)

print("Sharpe Ratios")
print(tabulate(sharpe_df, headers=['', '40/60 portfolio', 'SPY'], tablefmt='fancy_grid'))

leveraged_etf_symbols = [ 'SSO', 'UBT', 'UST']
leveraged_etf_weights = {"SSO": 0.40, "UBT": 0.44, "UST": 0.16 }
leveraged_etf_weights_df: pd.DataFrame = pd.DataFrame( leveraged_etf_weights.values()).transpose()
leveraged_etf_weights_df.columns = leveraged_etf_weights.keys()

# Start dates on finance.yahoo.com are completely available on this date
leveraged_etf_start_date_str = '2011-01-01'
leveraged_start_date: datetime = datetime.fromisoformat(leveraged_etf_start_date_str)

leveraged_etf_close_file = 'leveraged_etf_close'
# Fetch the adjusted close price for the unleveraged "all weather" set of ETFs'
# VTI, VGLT, VGIT, VPU and IAU
leveraged_etf_close: pd.DataFrame = get_market_data(file_name=leveraged_etf_close_file,
                                          data_col='Close',
                                          symbols=leveraged_etf_symbols,
                                          data_source=data_source,
                                          start_date=leveraged_start_date,
                                          end_date=end_date)

leveraged_etf_adj_close_file = 'leveraged_etf_adj_close'
# Fetch the adjusted close price for the unleveraged "all weather" set of ETFs'
# VTI, VGLT, VGIT, VPU and IAU
leveraged_etf_adj_close: pd.DataFrame = get_market_data(file_name=leveraged_etf_adj_close_file,
                                          data_col="Adj Close",
                                          symbols=leveraged_etf_symbols,
                                          data_source=data_source,
                                          start_date=leveraged_start_date,
                                          end_date=end_date)

leveraged_prices = leveraged_etf_close[0:1]
leveraged_returns = return_df(leveraged_etf_adj_close)

leveraged_holdings, shares = calc_portfolio_holdings(initial_investment=initial_investment,
                                           weights=leveraged_etf_weights_df,
                                           prices=leveraged_prices)

leveraged_portfolio_df, leveraged_portfolio_total_df = calc_rebalanced_portfolio(holdings=leveraged_holdings,
                                                             etf_close=leveraged_etf_close,
                                                             returns=leveraged_returns,
                                                             weights=leveraged_etf_weights_df,
                                                             rebalance_days=trading_days)

market_return_index = market_return.index
leveraged_return_index = leveraged_returns.index
market_filter = market_return_index.isin(leveraged_return_index)
trunc_market_return = market_return[market_filter]

initial_start_date = leveraged_return_index[0]
spy_close_index = spy_close.index.isin([initial_start_date])
leveraged_spy_price = float(spy_close[spy_close_index].values[0])

market_portfolio_df = calc_market_portfolio(market_return_df=trunc_market_return,
                                            date_index=leveraged_etf_close.index,
                                            initial_investment=initial_investment,
                                            initial_market_price=leveraged_spy_price)

leveraged_portfolio_plus_spy = pd.concat([leveraged_portfolio_total_df, market_portfolio_df], axis=1)
# leveraged_portfolio_plus_spy.plot(title="40/60 stock/bond, 2X leverage + SPY", grid=True, figsize=(10,8))

leveraged_portfolio_total_return = return_df(leveraged_portfolio_total_df)
stat_values = [calc_asset_beta(leveraged_portfolio_total_return, trunc_market_return),
         np.std(leveraged_portfolio_total_return),
         np.std(trunc_market_return)]

stats_df = pd.DataFrame(stat_values).transpose()
stats_df.columns = ['2X Beta', '2X StdDev', 'Market StdDev']

print("2X portfolio beta and standard deviation of the daily return")
print(tabulate(stats_df, headers=['', *stats_df.columns], tablefmt='fancy_grid'))


annual_ret_df = period_return(leveraged_portfolio_total_df, trading_days)
# plot_return(annual_ret_df, '2X Portfolio Annual Return')

mean_return_df = pd.DataFrame(annual_ret_df.mean(axis=0))
print(tabulate(mean_return_df, headers=['', 'Mean Annual Return'], tablefmt='fancy_grid'))

dividend_symbols = ['PHK', 'RCS', 'PTY', 'PMF', 'SCHP']
# the Schwab TIPS fund was started in August 2010
div_start_date_str = start_date_str = '2011-01-01'
div_start_date: datetime = datetime.fromisoformat(div_start_date_str)
# end date is the previously defined end_date
dividend_adj_close_file = "dividend_adj_close"

dividend_adj_close = get_market_data(file_name=dividend_adj_close_file,
                                     symbols=dividend_symbols,
                                     data_col='Adj Close',
                                     data_source=data_source,
                                     start_date=div_start_date,
                                     end_date=end_date)

dividend_returns = return_df(dividend_adj_close)


def calc_asset_value(initial_value: int, returns: pd.DataFrame) -> pd.DataFrame:
    length = returns.shape[0] + 1
    portfolio_value_np: np.array = np.zeros(length)
    portfolio_value_np[0] = initial_value
    for t in range(1, length):
        portfolio_value_np[t] = portfolio_value_np[t-1] + (portfolio_value_np[t-1] * returns[t-1])
    return pd.DataFrame(portfolio_value_np)


def calc_basket_value(initial_value: int, date_index: pd.Index, asset_returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the asset value for a basket of stocks.
    :param initial_value:
    :param date_index:
    :param asset_returns_df:
    :return:
    """
    portfolio_df: pd.DataFrame = pd.DataFrame()
    index = [start_date, asset_returns_df.index]
    for col in asset_returns_df.columns:
        returns = asset_returns_df[col]
        asset_value = calc_asset_value(initial_value, returns)
        asset_value.columns = [col]
        asset_value.index = date_index
        portfolio_df[col] = asset_value
    return portfolio_df


asset_returns_df = calc_basket_value(initial_value=initial_investment, date_index=dividend_adj_close.index, asset_returns_df=dividend_returns)

# asset_returns_df.plot(title="PIMCO CEFs + SCHP", grid=True, figsize=(10,8))

def calc_basket_beta(asset_returns_df: pd.DataFrame, market_return_df: pd.DataFrame) -> pd.DataFrame:
    asset_beta_l: list = []
    for col in asset_returns_df.columns:
        asset_beta = calc_asset_beta(asset_returns_df[col], market_return_df)
        asset_beta_l.append(asset_beta)
    basket_beta_df: pd.DataFrame = pd.DataFrame(asset_beta_l).transpose()
    basket_beta_df.columns = asset_returns_df.columns
    return basket_beta_df

def calc_basket_std(asset_returns_df: pd.DataFrame) -> pd.DataFrame:
    asset_std: list = []
    for col in asset_returns_df.columns:
        asset_std.append( np.std(asset_returns_df[col]) )
    basket_std: pd.DataFrame = pd.DataFrame(asset_std).transpose()
    basket_std.columns = asset_returns_df.columns
    return basket_std

market_return_index = market_return.index
dividend_return_index = dividend_returns.index
market_filter = market_return_index.isin(dividend_return_index)

trunc_market_return = market_return[market_filter]
asset_beta_df = calc_basket_beta(dividend_returns, trunc_market_return)
asset_beta_df.index = ['Beta']
print(tabulate(asset_beta_df, headers=['', *asset_beta_df.columns], tablefmt='fancy_grid'))

asset_std_df = calc_basket_std(dividend_returns)
asset_std_df['Market'] = pd.DataFrame([ np.std(trunc_market_return)])
asset_std_df.index = ['StdDev']
print("Daily Return Standard Deviation")
print(tabulate(asset_std_df, headers=['', *asset_std_df.columns], tablefmt='fancy_grid'))

dividend_returns_adj, rf_adj = adjust_time_series(dividend_returns, rf_daily_df)
pimco_asset_sharpe_ratio = calc_sharpe_ratio(dividend_returns_adj, pd.Series(rf_adj.values.flatten()), trading_days)
print("Pimco fund Sharpe Ratios")
print(tabulate(pimco_asset_sharpe_ratio, headers=['', *pimco_asset_sharpe_ratio.columns], tablefmt='fancy_grid'))

simple_port_symbols = [ 'VTI', 'SCHP']
simple_port_weights = {"VTI": 0.40, "SCHP": 0.60 }
simple_port_weights_df: pd.DataFrame = pd.DataFrame( simple_port_weights.values()).transpose()
simple_port_weights_df.columns = simple_port_weights.keys()

# The SCHP adjusted close time series has been previously fetched. See dividend_adj_close.
# The close prices has not been previously fetched.
schp_close_file = 'schp_close'
schp_close = get_market_data(file_name=schp_close_file,
                                data_col='Close',
                                symbols=['SCHP'],
                                data_source=data_source,
                                start_date=div_start_date,
                                end_date=end_date)

etf_close_forty_sixty = etf_close[forty_sixty_weights_df.columns]
# Truncate the etf_close time series so that the match the schp_close time series
etf_close_trunc: pd.DataFrame = pd.DataFrame()
for col in etf_close_forty_sixty.columns:
    ts_close_trunc, temp = adjust_time_series(etf_close_forty_sixty[col], schp_close)
    etf_close_trunc[col] = ts_close_trunc

# dividend_returns contains the returns for the PIMCO funds plus SCHP
schp_returns = dividend_returns['SCHP']
etf_returns_trunc: pd.DataFrame = pd.DataFrame()
# forty_sixty_weights_df has weights for VTI, VGLT and VGIT
# etf_returns_trunc will be the returns for VTI, VGLT and VGIT truncated to match the schp_returns time period
for col in forty_sixty_weights_df.columns:
    ret_trunc, temp = adjust_time_series(returns[col], schp_returns)
    etf_returns_trunc[col] = ret_trunc

sixty_forty_trunc_close = etf_close_trunc[0:1]
sixty_forty_trunc_holdings, shares = calc_portfolio_holdings(initial_investment=initial_investment,
                                                     weights=forty_sixty_weights_df,
                                                     prices=sixty_forty_trunc_close)

sixty_forty_port_trunc_df, sixty_forty_port_total_df = calc_rebalanced_portfolio(holdings=sixty_forty_trunc_holdings,
                                                                                 etf_close=etf_close_trunc,
                                                                                 returns=etf_returns_trunc,
                                                                                 weights=forty_sixty_weights_df,
                                                                                 rebalance_days=trading_days)

simple_port_returns = pd.concat([etf_returns_trunc['VTI'], schp_returns], axis=1)
simple_port_close = pd.concat([etf_close_trunc['VTI'], schp_close], axis=1)

simple_port_prices = simple_port_close[0:1]

simple_port_holdings, shares = calc_portfolio_holdings(initial_investment=initial_investment,
                                           weights=simple_port_weights_df,
                                           prices=simple_port_prices)

simple_port_df, simple_port_total_df = calc_rebalanced_portfolio(holdings=simple_port_holdings,
                                                             etf_close=simple_port_close,
                                                             returns=simple_port_returns,
                                                             weights=simple_port_weights_df,
                                                             rebalance_days=trading_days)

portfolios_df: pd.DataFrame = pd.concat([simple_port_total_df, sixty_forty_port_total_df], axis=1)

portfolios_df.columns = ['VTI/SCHP', 'VTI,(VGLT, VGIT)']
portfolios_df.plot(title="40/60 VTI/SCHP, 40/60 VTI,(VGLT, VGIT)", grid=True, figsize=(10,8))

portfolio_ret = return_df(portfolios_df)

"""
portfolio_ret_adj: pd.DataFrame = pd.DataFrame()
rf_adj: pd.DataFrame = pd.DataFrame()
for col in portfolio_ret.columns:
    port_adj, rf_adj = adjust_time_series(portfolio_ret[col], rf_daily_df)
    portfolio_ret_adj[col] = port_adj
"""

def portfolio_adj_timeseries(portfolio_df: pd.DataFrame, time_series_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    portfolio_adj: pd.DataFrame = pd.DataFrame()
    ts_adj: pd.DataFrame = pd.DataFrame()
    for col in portfolio_df.columns:
        port_adj, ts_adj = adjust_time_series(portfolio_df[col], time_series_df)
        portfolio_adj[col] = port_adj
    return portfolio_adj, ts_adj

portfolio_ret_adj, rf_adj = portfolio_adj_timeseries(portfolio_ret, rf_daily_df)
    
port_sharpe_ratio = calc_sharpe_ratio(portfolio_ret_adj, pd.Series(rf_adj.values.flatten()), trading_days)

port_vol = pd.DataFrame(list( portfolio_ret_adj.std())).transpose()
port_vol.columns = portfolio_ret_adj.columns

print(tabulate(port_vol, headers=['', *port_vol.columns], tablefmt='fancy_grid'))
print(tabulate(port_sharpe_ratio, headers=['', *port_sharpe_ratio.columns], tablefmt='fancy_grid'))

sixty_forty_port_ret = portfolio_ret[portfolio_ret.columns[0]]
sixty_forty_port_ret.index = pd.to_datetime(portfolio_ret.index)
# qs.plots.drawdown( sixty_forty_port_ret )

# dividend_adj_close['SCHP'] contains the adjusted close prices for SCHP from it's inception.
# etf_adj_close['VTI'] contains the VTI adjusted close prices. This is a longer period that dividend_adj_close so the
# period must be adjusted so that it is the same.

vti_adj_close = etf_adj_close['VTI']
schp_adj_close = dividend_adj_close['SCHP']
vti_adj_close, schp_adj_close = adjust_time_series(vti_adj_close, schp_adj_close)
sixty_forty_adj_close = pd.concat([vti_adj_close, schp_adj_close], axis=1)

def calc_mv_opt_weights(adj_close: pd.DataFrame, default_df:pd.DataFrame) -> pd.DataFrame:
    mu = expected_returns.capm_return(adj_close)
    if all(mu > 0):
        S = risk_models.CovarianceShrinkage(adj_close).ledoit_wolf()
        ef = pyopt.EfficientFrontier(mu, S)
        ef.max_sharpe()
        opt_weights = ef.clean_weights()
        opt_weights_df: pd.DataFrame = pd.DataFrame( opt_weights.values()).transpose()
        opt_weights_df.columns = opt_weights.keys()
    else:
        opt_weights_df = default_df
    return opt_weights_df


ix_l = list(ix for ix in range((sixty_forty_adj_close.shape[0]-1), -1, -trading_days))
ix_l.reverse()
if ix_l[0] > 0:
    ix_l[0] = 0

vti_schp_weights = pd.DataFrame()
for i in range(1, len(ix_l)):
    start_ix = ix_l[i-1]
    end_ix = ix_l[i]
    sec = sixty_forty_adj_close[start_ix:end_ix]
    opt_weights_df = round(calc_mv_opt_weights(sec, simple_port_weights_df), 2)
    opt_weights_df.index = [sixty_forty_adj_close.index[end_ix]]
    vti_schp_weights = vti_schp_weights.append(opt_weights_df)

print(tabulate(vti_schp_weights, headers=['', *vti_schp_weights.columns], tablefmt='fancy_grid'))

opt_port_weights_df = round(calc_mv_opt_weights(sixty_forty_adj_close, simple_port_weights_df), 2)
print(tabulate(opt_port_weights_df, headers=['', *opt_port_weights_df.columns], tablefmt='fancy_grid'))

opt_port_holdings, shares = calc_portfolio_holdings(initial_investment=initial_investment,
                                           weights=opt_port_weights_df,
                                           prices=simple_port_prices)

opt_port_df, opt_port_total_df = calc_rebalanced_portfolio(holdings=opt_port_holdings,
                                                             etf_close=simple_port_close,
                                                             returns=simple_port_returns,
                                                             weights=opt_port_weights_df,
                                                             rebalance_days=trading_days)

portfolios_df: pd.DataFrame = pd.concat([simple_port_total_df, opt_port_total_df], axis=1)
portfolios_df.columns = ['60/40 VTI/SCHP', 'Optimized VTI/SCHP']

# portfolios_df.plot(title="40/60 VTI/SCHP, Optimized VTI/SCHP", grid=True, figsize=(10,8))

opt_portfolio_ret = return_df(portfolios_df)
opt_portfolio_ret_adj, rf_adj = portfolio_adj_timeseries(opt_portfolio_ret, rf_daily_df)

opt_port_sharpe_ratio = calc_sharpe_ratio(opt_portfolio_ret_adj, pd.Series(rf_adj.values.flatten()), trading_days)

print('Sharpe Ratio')
print(tabulate(opt_port_sharpe_ratio, headers=['', *opt_port_sharpe_ratio.columns], tablefmt='fancy_grid'))

bhk_close_file = 'bhk_close'
bhk_close = get_market_data(file_name=bhk_close_file,
                                data_col='Close',
                                symbols=['BHK'],
                                data_source=data_source,
                                start_date=div_start_date,
                                end_date=end_date)

bhk_adj_close_file = 'bhk_adj_close'
bhk_adj_close = get_market_data(file_name=bhk_adj_close_file,
                                data_col='Adj Close',
                                symbols=['BHK'],
                                data_source=data_source,
                                start_date=div_start_date,
                                end_date=end_date)


bhk_adj_close.plot(title="BHK Ajusted Close Price", grid=True, figsize=(10,8))

bhk_port_adj_close = pd.concat([sixty_forty_adj_close, bhk_adj_close], axis=1)
bhk_port_close = pd.concat([simple_port_close, bhk_close], axis=1)
bhk_port_prices = bhk_port_close[0:1]

bhk_port_weights_df = round(calc_mv_opt_weights(bhk_port_adj_close, simple_port_weights_df), 2)

print(tabulate(bhk_port_weights_df, headers=['', *bhk_port_weights_df.columns], tablefmt='fancy_grid'))

bhk_port_holdings, shares = calc_portfolio_holdings(initial_investment=initial_investment,
                                           weights=bhk_port_weights_df,
                                           prices=bhk_port_prices)

bhk_port_returns = return_df(bhk_port_adj_close)
bhk_port_df, bhk_port_total_df = calc_rebalanced_portfolio(holdings=bhk_port_holdings,
                                                             etf_close=bhk_port_close,
                                                             returns=bhk_port_returns,
                                                             weights=bhk_port_weights_df,
                                                             rebalance_days=trading_days)

portfolios_df: pd.DataFrame = pd.concat([simple_port_total_df, bhk_port_total_df], axis=1)
portfolios_df.columns = ['40/60 VTI/SCHP', 'Optimized VTI/SCHP/BHK']

portfolios_df.plot(title="40/60 VTI/SCHP, Optimized VTI/SCHP/BHK", grid=True, figsize=(10,8))

print("Hi there")

def main():
    print("Hello World!")

if __name__ == "__main__":
    main()
