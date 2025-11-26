"""
Autograding System for Assignment 3 Part 2
This file contains the grading logic and be protected in GitHub Classroom
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
from Markowitz_2 import MyPortfolio, df, Bdf


"""
Assignment Judge

The following functions will help check your solution.
"""


class AssignmentJudge:
    def __init__(self):
        self.mp = MyPortfolio(df, "SPY").get_results()
        self.Bmp = MyPortfolio(Bdf, "SPY").get_results()

    def plot_performance(self, price, strategy):
        # Plot cumulative returns
        _, ax = plt.subplots()
        returns = price.pct_change().fillna(0)
        (1 + returns["SPY"]).cumprod().plot(ax=ax, label="SPY")
        (1 + strategy[1]["Portfolio"]).cumprod().plot(ax=ax, label=f"MyPortfolio")

        ax.set_title("Cumulative Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.legend()
        plt.show()
        return None

    def plot_allocation(self, df_weights):
        df_weights = df_weights.fillna(0).ffill()

        # long only
        df_weights[df_weights < 0] = 0

        # Plotting
        _, ax = plt.subplots()
        df_weights.plot.area(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Allocation")
        ax.set_title("Asset Allocation Over Time")
        plt.show()
        return None

    def report_metrics(self, price, strategy, show=False):
        df_bl = pd.DataFrame()
        returns = price.pct_change().fillna(0)
        df_bl["SPY"] = returns["SPY"]
        df_bl["MP"] = pd.to_numeric(strategy[1]["Portfolio"], errors="coerce")
        sharpe_ratio = qs.stats.sharpe(df_bl)

        if show == True:
            qs.reports.metrics(df_bl, mode="full", display=show)

        return sharpe_ratio

    def cumulative_product(self, dataframe):
        (1 + dataframe.pct_change().fillna(0)).cumprod().plot()

    def check_sharp_ratio_greater_than_one(self):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        if self.report_metrics(df, self.mp)[1] > 1:
            print("Problem 4.1 Success - Get 15 points")
            return 15
        else:
            print("Problem 4.1 Fail")
        return 0

    def check_sharp_ratio_greater_than_spy(self):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        if (
            self.report_metrics(Bdf, self.Bmp)[1]
            > self.report_metrics(Bdf, self.Bmp)[0]
        ):
            print("Problem 4.2 Success - Get 15 points")
            return 15
        else:
            print("Problem 4.2 Fail")
        return 0

    def check_portfolio_position(self, portfolio_weights):
        if (portfolio_weights.sum(axis=1) <= 1.01).all():
            return True
        print("Portfolio Position Exceeds 1. No Leverage.")
        return False

    def check_all_answer(self):
        score = 0
        score += self.check_sharp_ratio_greater_than_one()
        score += self.check_sharp_ratio_greater_than_spy()
        return score

    def run_grading(self, args):
        """
        Main grading function with exit code logic (protected from student modification)
        """
        if args.score:
            score_list = args.score
            
            if "one" in score_list:
                # Problem 4.1 單獨評分 (15分)
                score = self.check_sharp_ratio_greater_than_one()
                if score == 15:
                    sys.exit(0)  # 正確 -> exit code 0
                else:
                    sys.exit(1)  # 錯誤 -> exit code 1
                    
            elif "spy" in score_list:
                # Problem 4.2 單獨評分 (15分)
                score = self.check_sharp_ratio_greater_than_spy()
                if score == 15:
                    sys.exit(0)  # 正確 -> exit code 0
                else:
                    sys.exit(1)  # 錯誤 -> exit code 1

            elif "all" in score_list:
                total_score = self.check_all_answer()
                print(f"==> Total Score = {total_score} / 30 <==")
                return

        if args.allocation:
            if "mp" in args.allocation:
                self.plot_allocation(self.mp[0])
            if "bmp" in args.allocation:
                self.plot_allocation(self.Bmp[0])

        if args.performance:
            if "mp" in args.performance:
                self.plot_performance(df, self.mp)
            if "bmp" in args.performance:
                self.plot_performance(Bdf, self.Bmp)

        if args.report:
            if "mp" in args.report:
                self.report_metrics(df, self.mp, show=True)
            if "bmp" in args.report:
                self.report_metrics(Bdf, self.Bmp, show=True)

        if args.cumulative:
            if "mp" in args.cumulative:
                self.cumulative_product(df)
            if "bmp" in args.cumulative:
                self.cumulative_product(Bdf)