########################## SORTING REVIEWS #############################
# This project is a project that aims to correctly order the comments given to the products.
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
def load_data(dataname, time_col):
    return pd.read_csv(dataname, parse_dates = [time_col])


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
    return dataframe.loc[dataframe["day_diff"] <= 30, "overall"].mean() * w1 + \
           dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 60), "overall"].mean() * w2 + \
           dataframe.loc[(dataframe["day_diff"] > 60) & (dataframe["day_diff"] <= 90), "overall"].mean() * w3 + \
           dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w4 + \
           dataframe.loc[(dataframe["day_diff"] > 180) & (dataframe["day_diff"] <= 360), "overall"].mean() * w5 + \
           dataframe.loc[dataframe["day_diff"] > 360, "overall"].mean() * w6


w_avg_score_time = time_based_weighted_average(df)
print(w_avg_score_time)

day_0_30 = df.loc[df["day_diff"] <= 30, "overall"].mean()
day_30_60 = df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 60), "overall"].mean()
day_60_90 = df.loc[(df["day_diff"] > 60) & (df["day_diff"] <= 90), "overall"].mean()
day_90_180 = df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean()
day_180_360 = df.loc[(df["day_diff"] > 180) & (df["day_diff"] <= 360), "overall"].mean()
day_360_plus = df.loc[df["day_diff"] > 360, "overall"].mean()
days = np.array([day_360_plus, day_180_360, day_90_180, day_60_90, day_30_60, day_0_30 ])

sns.lineplot(x= range(1, len(days)+1), y= days)
plt.title("Average Rating Based on Time")

# The score of the product has increased from the last 1 year to 6 months ago.
# The score of the product has decreased from the last 6 months to 3 months ago.
# It started to increase again 3 months ago but the increase has reversed in the last month.

##########################  Finding Top 20 Review for Detail Page ###########################

def score_up_down_diff(yes, no):
    return yes - no

def score_average_rating(yes, no):
    if yes + no == 0:
        return 0
    return yes / (yes + no)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

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

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

def scores_review(dataframe, up_col, down_col):
    df[down_col] = df["total_vote"] - df[up_col]
    dataframe["score_pos_neg_diff"] = dataframe.apply(lambda x: score_up_down_diff(x[up_col], x[down_col]), axis = 1)
    dataframe["score_average_rating"] = dataframe.apply(lambda x: score_average_rating(x[up_col], x[down_col]), axis = 1)
    dataframe["wilson_lower_bound"] = dataframe.apply(lambda x: wilson_lower_bound(x[up_col], x[down_col]), axis = 1)
    return dataframe


scores_review(df, "helpful_yes", "helpful_no")

def top_x_review(dataframe, top = 20):
    return dataframe.sort_values("wilson_lower_bound", ascending = False).head(top)


top_x_review(df)


