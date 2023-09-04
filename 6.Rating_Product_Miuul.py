########################## SORTING REVIEWS #############################
# This is a project that aims to correctly order the comments given to the products.
# The data of the project belongs to an electronic product on amazon.


########################## Import Library and Edit Settings  ###########################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as st
import os


path = "C:\\Users\\hseym\\OneDrive\\Masaüstü\\Miuul\\datasets"
os.chdir(path)

pd.set_option("display.max_columns", 30)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


########################## Loading  The Date  ###########################
def load_data(dataframe, time_col):
    return pd.read_csv(dataframe, parse_dates = [time_col])


reviews = load_data("amazon_review.csv", time_col = "reviewTime")
df = reviews.copy()


########################## Exploring The Data  ###########################
def check_df(dataframe):
    if isinstance(dataframe, pd.DataFrame):
        print("########## shape #########\n", dataframe.shape)
        print("########## types #########\n", dataframe.dtypes)
        print("########## head #########\n", dataframe.head())
        print("########## tail #########\n", dataframe.tail())
        print("########## NA #########\n", dataframe.isna().sum())
        print("########## describe #########\n", dataframe.describe().T)
        print("########## nunique #########\n", dataframe.nunique())


check_df(df)


##########################  Data Analysis  ###########################
def raw_avg_rate(dataframe, rate_col):
    return dataframe[rate_col].mean()


avg_rating_raw = raw_avg_rate(df, "overall")
print(avg_rating_raw)

def day_diff_analysis(dataframe, daydiff_col):
    print("Maximum day difference:\n", dataframe[daydiff_col].max())
    print("########################################################")
    print("Minimum day difference:\n", dataframe[daydiff_col].min())
    print("########################################################")
    print("Number of months in the system :\n", dataframe[daydiff_col].max()/30)


day_diff_analysis(df, "day_diff")


def time_based_weighted_average(dataframe, w1 = 0.24, w2 = 0.22, w3 = 0.20, w4 = 0.14, w5 = 0.11, w6 = 0.09):
    """
    It is a function that calculates an average score weighted by date.
    The weighting of time intervals is subjective.

    Parameters
    ----------
    dataframe
    w1 : Coefficient for comments in the first 30 days
    w2 : Coefficient for comments between 30 and 60 days
    w3 : Coefficient for comments between 60 and 90 days
    w4 : Coefficient for comments between 90 and 180 days
    w5 : Coefficient for comments between 180 and 360 days
    w6 : Coefficient for comments made before one year

    Returns
    -------
    average score weighted by date

    """
    return dataframe.loc[dataframe["day_diff"] <= 30, "overall"].mean() * w1 + \
           dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 60), "overall"].mean() * w2 + \
           dataframe.loc[(dataframe["day_diff"] > 60) & (dataframe["day_diff"] <= 90), "overall"].mean() * w3 + \
           dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w4 + \
           dataframe.loc[(dataframe["day_diff"] > 180) & (dataframe["day_diff"] <= 360), "overall"].mean() * w5 + \
           dataframe.loc[dataframe["day_diff"] > 360, "overall"].mean() * w6


w_avg_score_time = time_based_weighted_average(df)
print(w_avg_score_time)

# Date-weighted scores are higher than raw average scores.
# This means that the average score of this product has increased over time.

def plot_time_based_score(dataframe):
    day_0_30 = dataframe.loc[dataframe["day_diff"] <= 30, "overall"].mean()
    day_30_60 = dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 60), "overall"].mean()
    day_60_90 = dataframe.loc[(dataframe["day_diff"] > 60) & (dataframe["day_diff"] <= 90), "overall"].mean()
    day_90_180 = dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean()
    day_180_360 = dataframe.loc[(dataframe["day_diff"] > 180) & (dataframe["day_diff"] <= 360), "overall"].mean()
    day_360_plus = dataframe.loc[df["day_diff"] > 360, "overall"].mean()
    days = np.array([day_360_plus, day_180_360, day_90_180, day_60_90, day_30_60, day_0_30])
    periods = ["day_360_plus", "day_180_360", "day_90_180", "day_60_90", "day_30_60", "day_0_30"]

    sns.lineplot(x = periods, y = days)
    plt.title("Average Rating Based on Time")
    plt.xticks(rotation = 20)


plot_time_based_score(df)

# The score of the product has increased from the last 1 year to 6 months ago.
# The score of the product has decreased from the last 6 months to 3 months ago.
# It started to increase again 3 months ago but the increase has reversed in the last month.


##########################  Finding Top 20 Review for Detail Page ###########################
def score_pos_neg_diff(yes, no):
    return yes - no

def score_average_rating(yes, no):
    if yes + no == 0:
        return 0
    return yes / (yes + no)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    This function calculates the Wilson Lower Bound Score.
    - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is
        considered as the WLB score.
    - The score is used for product ranking.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    Notes
    ------
    - If the scores are between 1-5, 1-3 can be marked as negative and 4-5 as positive and
     can be made Bernoulli compatible.
    - This brings some problems with it. For this reason, it is necessary to make a Bayesian average rating.

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

def scores_review(dataframe, up_col, down_col):
    """
    It is a function that generates a ranking score in three ways.
        1- Directly generated from the difference between positive and negative comments.
        2- It is produced from the ratio of positive to negative comments.
        3- Wilson Lower Bond score is generated.
    Parameters
    ----------
    dataframe
    up_col : column showing the number of positive comments
    down_col : column showing the number of negative comments

    Returns
    -------
    dataframe
    """
    dataframe[down_col] = df["total_vote"] - dataframe[up_col]
    dataframe["score_pos_neg_diff"] = dataframe.apply(lambda x: score_pos_neg_diff (x[up_col], x[down_col]), axis = 1)
    dataframe["score_average_rating"] = dataframe.apply(lambda x: score_average_rating(x[up_col], x[down_col]), axis = 1)
    dataframe["wilson_lower_bound"] = dataframe.apply(lambda x: wilson_lower_bound(x[up_col], x[down_col]), axis = 1)
    return dataframe


scores_review(df, "helpful_yes", "helpful_no")

def top_x_review(dataframe, top = 20):
    return dataframe.sort_values("wilson_lower_bound", ascending = False).head(top)


top_x_review(df)


# When data is sorted by WLB, the comments most likely to be helpful are at the top.
# Here, we've seen reviews at the top that are likely to help users, even if they have a low rating.
# Thus, we noticed the comments that are impossible to see with the rough average calculation.
# These results support social proof while ensuring that objective reviews rank high.

