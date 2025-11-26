"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=126, gamma=100):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]
        n = len(assets)
        gamma = self.gamma
        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )
        self.portfolio_weights.fillna(0, inplace=True)
        if self.exclude in self.portfolio_weights.columns:
            self.portfolio_weights[self.exclude] = 0.0
        """
        TODO: Complete Task 4 Below
        """
        SMA_FILTER = 200    
        MOM_LOOKBACK = 126  
        VOL_LOOKBACK = 20   
        TOP_N = 3           


        spy_price = self.price['SPY']
        spy_ma200 = spy_price.rolling(window=SMA_FILTER).mean().shift(1)


        start_index = max(SMA_FILTER, MOM_LOOKBACK)

        n_assets = len(assets)
        self.portfolio_weights.iloc[:start_index, self.portfolio_weights.columns.isin(assets)] = 1.0 / n_assets

        for t in range(start_index, len(self.price.index)):
            current_date = self.price.index[t]
            

            current_spy = spy_price.iloc[t-1]
            current_ma = spy_ma200.iloc[t-1]
            
            is_bull_market = current_spy > current_ma

            price_now = self.price.iloc[t-1][assets]
            price_prev = self.price.iloc[t-MOM_LOOKBACK][assets]
            momentum = (price_now / price_prev) - 1
            
            vol_window = self.returns.iloc[t-VOL_LOOKBACK : t][assets]
            volatility = vol_window.std() * np.sqrt(252) # 年化
            volatility[volatility == 0] = 1e-6


            efficiency = momentum / volatility

            target_assets = []
            weights_series = pd.Series(0.0, index=assets)

            if is_bull_market:

                
                top_performers = efficiency.nlargest(TOP_N)

                target_assets = top_performers[top_performers > 0].index
                
                if len(target_assets) == 0:

                    target_assets = volatility.nsmallest(TOP_N).index
            
            else:

                target_assets = volatility.nsmallest(TOP_N).index

            if len(target_assets) > 0:
                if is_bull_market:
                    score = efficiency[target_assets]
                    total_score = score.sum()
                    if total_score > 0:
                        weights_series[target_assets] = score / total_score
                    else:
                        weights_series[target_assets] = 1.0 / len(target_assets)
                else:
                    inv_vol = 1.0 / volatility[target_assets]
                    weights_series[target_assets] = inv_vol / inv_vol.sum()
            else:
                weights_series[assets] = 1.0 / n_assets

            self.portfolio_weights.loc[current_date, assets] = weights_series.values
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
