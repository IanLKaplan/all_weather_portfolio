
<h2>
<center>
"All Weather" Portfolios<br/>
by<br/>
Ian Kaplan<br/>
November 2021
</center>
</h2>

<p>
This Python Jupyter notebook explores investment portfolios that have decent return
(approximately 8 percent on average over the last ten years) and lower risk than the
overall stock market.
</p>
<p>
If you are fortunate to have some cash available for investment you have to decide
how to invest this cash (not making an explicit investment
decision is a decision).
</p>
<p>
The US stock market, over the long term, has yielded good returns. Unfortunately it is an
inescapable fact that risk and return are related.  The returns provided by the stock
market come with the risk of losses in your investment portfolio, at least in the short to
medium term.  At the time this notebook was written (November 2021) the stock market
had dramatic returns after a COVID-19 inspired market crash.  In such a time it is
important to remember that there have been many periods where "the market" has had
substantial downturns and low returns.
</p>
<p>
As any good financial adviser will tell you, your investment strategy should depend
on your stage in life. Is retirement decades away, close or are you retired?
</p>
<p>
If your retirement is decades away, one of the simplest ways to achieve good investment
returns is "dollar cost averaging" where you invest a certain amount every month
in one or more low fee market index funds or ETFs.
</p>
<p>
When there is a market
downturn your share purchase cost will be lower resulting in gains when the market recovers.
Many people take advantage of dollar cost averaging by allocating a fraction of their
salary for investment in their employer's 401K retirement plan.
</p>
<p>
If you are at or near retirement age then the inevitable market downturns are much
less acceptable since you will not be able to take advantage of market cycles
that could last for years.
</p>
<p>
This notebook explores conservative investment portfolios that have lower risk and lower
correlation with the stock market (a.k.a., lower market beta). These are portfolios that
may be appropriate for people who are retired or near retirement.
</p>
<p>
The portfolio that originally inspired this notebook is based on
the Bridgewater Associates "All Weather" portfolio proposed by Bridgewater
founder Ray Dalio. This portfolio is discussed in a Bridgewater promotional white paper
<a href="https://www.bridgewater.com/research-and-insights/the-all-weather-story">The All Weather Story</a>
From the white paper:
</p>
<blockquote>
 What the average person needs is a good, reliable asset allocation they can hold for the long-run.
 Bridgewater’s answer is All Weather, the result of three decades of learning how to invest in the
 face of uncertainty.
</blockquote>
<p>
The results in this notebook show that, in the last ten years (from 2021) the
performance a portfolio that is similar to the Bridgewater "all weather" portfolio
is no better than a simple 40 percent stock, 60 percent bond portfolio that is often
recommended for those at or near retirement.  This notebook shows a portfolio consisting
of just two ETFs VTI (40%) and SCHP (60%) has less volatility than the "all weather" portfolio
with returns that are only slightly lower.
</p>
<p>
The 40% market/60% bond mix is difficult to beat when it comes to risk and return.
A number of dividend assets are examined in this notebook in an attempt to find
a portfolio with higher return at a similar risk level. This search was not successful.
</p>
<h2>
<center>
Dividend Portfolios<br/>
by<br/>
Ian Kaplan<br/>
December 2021
</center>
</h2>
<h3>
Introduction
</h3>
<p>
The classic conservative investment portfolio often consists of approximately 40 percent
stocks (perhaps a mix of stock ETFs) and 60 percent bonds (perhaps a US Treasury
bond ETF).  This type of portfolio is discussed in the
<a href="https://github.com/IanLKaplan/all_weather_portfolio/blob/master/all_weather_portfolio.ipynb">All Weather Portfolio</a>
notebook.
</p>
<p>
The portfolios investigated in the All Weather Portfolio notebook have an annual
return, on average, of approximately nine percent. These returns are highly
volatile from year to year.
</p>
<h4>
Interest Rate Risk
</h4>
<p>
In an era of low interest rates (December 2021), with a possible interest rate
increases in the future, an investment portfolio that consists mostly
of bonds has interest rate risk.
</p>
<p>
When interest rates rise, the value of current bond holdings decrease as the
bonds are discounted for the higher prevailing interest rate. The return paid by
ETF bond funds at the time this was written were around 3 percent. This level of
return does not compensate for the interest rate risk carried by a significant
position in bonds.
</p>
<p>
Considering the interest rate risk of a 40 percent stock/60 percent bond portfolio, such
portfolio is not, in fact, a conservative investment.
</p>
<h4>
Market Risk
</h4>
<p>
With the 2008 financial crisis and the COVID-19 pandemic, governments in the United
States, Europe and Japan have embarked on fiscal policies that are
referred to as "quantitative easing".
This has resulted in interest rates that have
been near zero. This has made many bond investments unattractive, causing
huge investment flows into stock market.
</p>
<p>
The result is a stock market with high valuations and less room for growth, especially
if interest rates rise.
</p>
<blockquote>
<p>
The P/E ratio is a classic measure of any security's value, indicating how many years of
profits (at the current rate) it takes to recoup an investment in the stock. The
current S&P500 10-year P/E Ratio is 37.5. This is 89% above the modern-era market
average of 19.6, putting the current P/E 2.3 standard deviations above the modern-era
average. This suggests that the market is Strongly Overvalued.
</p>
<p>
https://www.currentmarketvaluation.com/models/price-earnings.php
</p>
</blockquote>
<h4>
Dividends
</h4>
<p>
An alternative to the classic 40/60 stock/bond portfolio is a portfolio of
dividend assets. A diversified portfolio that returns approximately 7%
from dividends should have lower volatility, compared to a 40/60 stock/bond portfolio.
The focus of a dividend portfolio is not capital appreciate of the assets (as would
be the case with a stock portfolio), but dividend yield.
</p>
<p>
A dividend portfolio does have stock market risk, since all of the assets are either
stocks, ETFs are closed-end funds. A market crash or downturn will have an effect
on the face vaue of the portfolio.  The face value of the portfolio is only an issue
if the assets are sold. If the dividend portfolio continues to provide an attractive
yield, it would be held in the long term. The face value of the portfolio could be
expected to recover over time from a downturn.
</p>

