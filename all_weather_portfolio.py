
from datetime import datetime, timedelta
from tabulate import tabulate
from typing import List, Tuple
from pandas_datareader import data
from scipy.stats import stats
from pandas import Timestamp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

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

close_price_file = "etf_start_close_prices"
close_prices: pd.DataFrame =  get_market_data(file_name=close_price_file,
                                              data_col='Close',
                                              symbols=etf_symbols,
                                              data_source=data_source,
                                              start_date=start_date,
                                              end_date=start_date+timedelta(days=10))

etf_weights = {"VTI": 0.30, "VGLT": 0.40, "VGIT": 0.15, "VPU": 0.08, "IAU": 0.07}

prices = close_prices[0:1]

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

# initial investment, in dollars
initial_investment = 10000

weights_df: pd.DataFrame = pd.DataFrame(etf_weights.values()).transpose()
weights_df.columns = etf_weights.keys()

holdings, shares = calc_portfolio_holdings(initial_investment=initial_investment,
                                           weights=weights_df,
                                           prices=prices)


trading_days = 253
days_in_quarter = trading_days // 4

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

etf_adj_close_file = 'etf_adj_close'
etf_close_file = 'etf_close'

# Fetch the adjusted close price for the unleveraged "all weather" set of ETFs'
# VTI, VGLT, VGIT, VPU and IAU
etf_adj_close: pd.DataFrame = get_market_data(file_name=etf_adj_close_file,
                                          data_col="Adj Close",
                                          symbols=etf_symbols,
                                          data_source=data_source,
                                          start_date=start_date,
                                          end_date=end_date)

# Fetch all of the close prices: this is faster than fetching only the dates needed.
etf_close: pd.DataFrame = get_market_data(file_name=etf_close_file,
                                          data_col="Close",
                                          symbols=etf_symbols,
                                          data_source=data_source,
                                          start_date=start_date,
                                          end_date=end_date)

# +/- for each asset for portfolio rebalancing
portfolio_range = 0.05

returns = return_df(etf_adj_close)

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

portfolio_df, portfolio_total_df = calc_rebalanced_portfolio(holdings=holdings,
                                                             etf_close=etf_close,
                                                             returns=returns,
                                                             weights=weights_df,
                                                             rebalance_days=days_in_quarter)

portfolio_yearly_df, portfolio_total_yearly_df = calc_rebalanced_portfolio(holdings=holdings,
                                                             etf_close=etf_close,
                                                             returns=returns,
                                                             weights=weights_df,
                                                             rebalance_days=trading_days)

portfolio_total_np = np.array(portfolio_total_df)
portfolio_np = np.array(portfolio_df)
portfolio_percent_np = portfolio_np / portfolio_total_np

for col in range(0, portfolio_percent_np.shape[1]):
    percent = portfolio_percent_np[:,col]

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

portfolios_df: pd.DataFrame = pd.concat([portfolio_total_df, market_portfolio_df], axis=1)

forty_sixty_weights_df = pd.DataFrame([0.40, 0.30, 0.30]).transpose()
forty_sixty_weights_df.columns = ['VTI', 'VGLT', 'VGIT']
forty_sixty_holdings, forty_sixty_shares = calc_portfolio_holdings(initial_investment=initial_investment,
                                           weights=forty_sixty_weights_df,
                                           prices=prices)

portfolio_forty_sixty_df, portfolio_total_forty_sixty_df = calc_rebalanced_portfolio(holdings=forty_sixty_holdings,
                                                             etf_close=etf_close[forty_sixty_weights_df.columns],
                                                             returns=returns[forty_sixty_weights_df.columns],
                                                             weights=forty_sixty_weights_df,
                                                             rebalance_days=trading_days)


portfolio_yearly_return = return_df(portfolio_total_yearly_df)

portfolio_yearly_sd = np.std( portfolio_yearly_return ) * np.sqrt(portfolio_yearly_return.shape[0])

portfolio_sixty_forty_return = return_df(portfolio_total_forty_sixty_df)
portfolio_sixty_forty_sd = np.std( portfolio_sixty_forty_return ) * np.sqrt(portfolio_sixty_forty_return.shape[0])

sd_df = pd.DataFrame([portfolio_sixty_forty_sd, portfolio_yearly_sd]).transpose()
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

def period_return(portfolio_total_df: pd.DataFrame, trading_days: int) -> pd.DataFrame:
    year_range: list = list(t for t in range(portfolio_total_df.shape[0]-1, -1, -trading_days))
    year_range.reverse()
    portfolio_total_np: np.array = np.array(portfolio_total_df)
    annual_ret_l: list = []
    annual_dates_l: list = []
    for ix in range(1, len(year_range)):
        ret = (portfolio_total_np[ year_range[ix] ] / portfolio_total_np[ year_range[ix-1] ]) - 1
        ret = float((ret * 100).round(2))
        annual_ret_l.append(ret)
        annual_dates_l.append(portfolio_total_df.index[year_range[ix]])

    annual_ret_df: pd.DataFrame = pd.DataFrame(annual_ret_l, index=annual_dates_l)
    annual_ret_df.columns = ['Annual Return']
    return annual_ret_df

annual_ret_df = period_return(portfolio_total_forty_sixty_df, trading_days)

risk_free_asset = '^IRX'

rf_file_name = 'rf_adj_close'
rf_adj_close = get_market_data(file_name=rf_file_name,
                                data_col='Adj Close',
                                symbols=[risk_free_asset],
                                data_source=data_source,
                                start_date=start_date,
                                end_date=end_date)

rf_adj_rate_np: np.array = np.array( rf_adj_close.values ) / 100
rf_daily_np = ((1 + rf_adj_rate_np) ** (1/360)) - 1
rf_daily_df: pd.DataFrame = pd.DataFrame( rf_daily_np, index=rf_adj_close.index, columns=['^IRX'])

def adjust_time_series(return_df: pd.DataFrame, rf_values: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ret_index = pd.to_datetime(return_df.index)
    rf_index = pd.to_datetime(rf_values.index)
        # filter the close prices
    matching_dates = ret_index.isin( rf_index )
    ret_adj = return_df[matching_dates]
    # filter the rf_prices
    ret_index = pd.to_datetime(ret_adj.index)
    matching_dates = rf_index.isin(ret_index)
    rf_prices_adj = rf_values[matching_dates]
    return ret_adj, rf_prices_adj

def excess_return_series(asset_return: pd.Series, risk_free: pd.Series) -> pd.DataFrame:
    excess_ret = asset_return.values - risk_free.values
    excess_ret_df = pd.DataFrame(excess_ret, index=asset_return.index)
    return excess_ret_df


def excess_return_df(asset_return: pd.DataFrame, risk_free: pd.Series) -> pd.DataFrame:
    excess_df: pd.DataFrame = pd.DataFrame()
    for i, col in enumerate(asset_return.columns):
        e_df = excess_return_series(asset_return[col], risk_free)
        excess_df.insert(i, col, e_df)
    return excess_df

def calc_sharpe_ratio(asset_return: pd.DataFrame, risk_free: pd.Series) -> pd.DataFrame:
    excess_return = excess_return_df(asset_return, risk_free)
    return_mean: List = []
    return_stddev: List = []
    for col in excess_return.columns:
        mu = np.mean(excess_return[col])
        std = np.stdev(excess_return[col])
        return_mean.append(mu)
        return_stddev.append(std)
    # daily Sharpe ratio
    # https://quant.stackexchange.com/questions/2260/how-to-annualize-sharpe-ratio
    sharpe_ratio = np.asarray(return_mean) / np.asarray(return_stddev)
    result_df: pd.DataFrame = pd.DataFrame(sharpe_ratio).transpose()
    result_df.columns = asset_return.columns
    ix = asset_return.index
    ix_start = ix[0].date()
    ix_end = ix[len(ix)-1].date()
    index_str = f'{ix_start} : {ix_end}'
    result_df.index = [ index_str ]
    return result_df


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
print(tabulate(beta_df, headers=['', 'VTI Beta', 'VGLT Beta', 'Portfolio Beta'], tablefmt='fancy_grid'))

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


def calc_portfolio_value(initial_value: int, date_index: pd.Index, asset_returns_df: pd.DataFrame) -> pd.DataFrame:
    portfolio_df: pd.DataFrame = pd.DataFrame()
    index = [start_date, asset_returns_df.index]
    for col in asset_returns_df.columns:
        returns = asset_returns_df[col]
        asset_value = calc_asset_value(initial_value, returns)
        asset_value.columns = [col]
        asset_value.index = date_index
        portfolio_df[col] = asset_value
    return portfolio_df

asset_returns_df = calc_portfolio_value(initial_value=initial_investment, date_index=dividend_adj_close.index, asset_returns_df=dividend_returns)


def calc_basket_beta(asset_returns_df: pd.DataFrame, market_return_df: pd.DataFrame) -> pd.DataFrame:
    asset_beta_l: list = []
    for col in asset_returns_df.columns:
        asset_beta = calc_asset_beta(asset_returns_df[col], market_return_df)
        asset_beta_l.append(asset_beta)
    basket_beta_df: pd.DataFrame = pd.DataFrame(asset_beta_l).transpose()
    basket_beta_df.columns = asset_returns_df.columns
    return basket_beta_df

market_return_index = market_return.index
dividend_return_index = dividend_returns.index
market_filter = market_return_index.isin(dividend_return_index)

trunc_market_return = market_return[market_filter]
asset_beta_df = calc_basket_beta(dividend_returns, trunc_market_return)
asset_beta_df.index = ['Beta']
print(tabulate(asset_beta_df, headers=['', *asset_beta_df.columns], tablefmt='fancy_grid'))

leveraged_etf_symbols = [ 'SSO', 'UBT', 'UST']
leveraged_etf_weights = {"SSO": 0.40, "UBT": 0.44, "UST": 0.16 }
leveraged_etf_weights_df: pd.DataFrame = pd.DataFrame( leveraged_etf_weights.values()).transpose()
leveraged_etf_weights_df.columns = leveraged_etf_weights.keys()

leveraged_etf_start_date_str = '2011-01-01'
leveraged_start_date: datetime = datetime.fromisoformat(leveraged_etf_start_date_str)

leveraged_etf_close_file = 'leveraged_etf_close'
# Fetch the adjusted close price for the unleveraged "all weather" set of ETFs'
# VTI, VGLT, VGIT, VPU and IAU
leveraged_etf_close: pd.DataFrame = get_market_data(file_name=leveraged_etf_close_file,
                                          data_col="Close",
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

stats = [calc_asset_beta(leveraged_returns, trunc_market_return),
         np.std(leveraged_returns),
         np.std(trunc_market_return)]

stats_df = pd.DataFrame(stats).transpose()
stats_df.columns = ['2X Beta', '2X StdDev', 'Market StdDev']

print(tabulate(stats_df, headers=['', *stats_df.columns], tablefmt='fancy_grid'))

print("hi there")

def main():
    print("Hello World!")

if __name__ == "__main__":
    main()
