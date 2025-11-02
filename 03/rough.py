# This utils file was created for Homework 03.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import grangercausalitytests
from arch.unitroot import ADF, PhillipsPerron, KPSS
from sklearn.metrics import confusion_matrix
import warnings
# we ignore deprecation warnings and futurewarnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

############################################################
file_path = "dataUSA_MSFT_NVDA.pkl"
dataUSA_MSFT_NVDA = pd.read_pickle(file_path)
dataUSA_MSFT_NVDA

dataUSA_MSFT_NVDA = dataUSA_MSFT_NVDA.iloc[:, ~dataUSA_MSFT_NVDA.columns.str.contains("_diff_")]


############################################################

######### 3.2

# Granger Causality test
# Does NVDA granger cause MSFT?
print("We check if NVDA prices Granger-cause MSFT prices")
granger_test_M_N = grangercausalitytests(dataUSA_MSFT_NVDA[["close_MSFT", "close_NVDA"]], maxlag = [10])
print("\n")

# Does NVDA granger cause MSFT?
print("We check if NVDA prices Granger-cause MSFT prices")
granger_test_N_M = grangercausalitytests(dataUSA_MSFT_NVDA[["close_NVDA", "close_MSFT"]], maxlag = [10])

from utils import eg_adf_pvalue, eg_pp_pvalue ,eg_kpss_pvalue 
from utils import granger_pvalue

# 3.2.1 Granger Causality test Rolling 
df_M_N = dataUSA_MSFT_NVDA[["close_MSFT", "close_NVDA"]]
pvalues = (eg_adf_pvalue(df_M_N, "close_MSFT", "close_NVDA"),
           eg_pp_pvalue(df_M_N, "close_MSFT", "close_NVDA"),
           eg_kpss_pvalue(df_M_N, "close_MSFT", "close_NVDA"))

print("p-values from Engle-Granger residual tests (ADF, PP, KPSS):", pvalues)

# MSFT to NVDA
_granger1 = lambda x: granger_pvalue(df_M_N.loc[x.index], "close_MSFT", "close_NVDA", maxlag = 10)

# NVDA to MSFT
_granger2 = lambda x: granger_pvalue(df_M_N.loc[x.index], "close_NVDA", "close_MSFT",maxlag = 10)

granger_pvalues_MSFT_to_NVDA = df_M_N["close_MSFT"].rolling(window = 60, step = 15).apply(_granger1)
granger_pvalues_NVDA_to_MSFT = df_M_N["close_NVDA"].rolling(window = 60, step = 15).apply(_granger2)

granger_pvalues = pd.concat([granger_pvalues_MSFT_to_NVDA.rename("p_value_NVDA_to_MSFT"),
                             granger_pvalues_NVDA_to_MSFT.rename("p_value_MSFT_to_NVDA")],
                             axis=1).dropna()
print(granger_pvalues)

# 3.2.2
# how often do we find causality (reject the null hypothesis) at 5% significance level?
rejections_NVDA_to_MSFT = (granger_pvalues["p_value_NVDA_to_MSFT"] < 0.05).sum()/len(granger_pvalues)
rejections_MSFT_to_NVDA = (granger_pvalues["p_value_MSFT_to_NVDA"] < 0.05).sum()/len(granger_pvalues)
print("Proportion of NVDA causes MSFT:", rejections_NVDA_to_MSFT)
print("Proportion of MSFT causes NVDA:", rejections_MSFT_to_NVDA)

# 3.2.3
# check with a contingency table how many times
# we have both rejections, none or only one-sided rejection

confusion_mat = confusion_matrix(
    granger_pvalues["p_value_NVDA_to_MSFT"] < 0.05,
    granger_pvalues["p_value_MSFT_to_NVDA"] < 0.05
)
# add labels of rows and columns
confusion_mat_df = pd.DataFrame(confusion_mat/np.sum(confusion_mat),
                                index=["NVDA does NOT cause MSFT", "NVDA causes MSFT"],
                                columns=["MSFT does NOT cause NVDA", "MSFT causes NVDA"])

print("Contingency table of results of Granger causality tests:")
print(confusion_mat_df)

# 3.3.1
# Testing Cointegration 
X_price = dataUSA_MSFT_NVDA["close_MSFT"] 
y_price = dataUSA_MSFT_NVDA["close_NVDA"] 

X_price = sm.add_constant(X_price)
model_ols_p = sm.OLS(y_price, X_price).fit()
print (model_ols_p.summary())

# Are the residuals stationary? Spoiler Alert: No, we have a high p value
residuals = model_ols_p.resid
adf_test2 = ADF(residuals, lags = 5, trend = 'c')
print(adf_test2.summary().as_text())

# Markdown :They are not cointegrated


# 3.3.2 Rolling Cointegration Test
_adf = lambda x: eg_adf_pvalue(df_M_N.loc[x.index], "close_MSFT", "close_NVDA")
_pp  = lambda x: eg_pp_pvalue(df_M_N.loc[x.index], 
                              "close_MSFT", "close_NVDA")
_kpss= lambda x: eg_kpss_pvalue(df_M_N.loc[x.index], 
                                "close_MSFT", "close_NVDA")

# we apply the test every 60 minutes on a 240 minute window
adf_p = df_M_N["close_NVDA"].rolling(window = 240, step = 60).apply(_adf)
pp_p  = df_M_N["close_NVDA"].rolling(window = 240, step = 60).apply(_pp)
kpss_p= df_M_N["close_NVDA"].rolling(window = 240, step = 60).apply(_kpss)

out = pd.concat({"adf_p": adf_p, "pp_p": pp_p, "kpss_p": kpss_p}, axis=1).dropna()

# 3.3.3 
out.plot()

# 3.3.4
# lets check how many times would we identify cointegration at 5% significance level?
print("Share of times cointegration identified at 5% significance level:")
print("ADF test:", (out["adf_p"] < 0.05).sum()/out.size)
print("PP test:", (out["pp_p"] < 0.05).sum()/out.size)
print("KPSS test:", (out["kpss_p"] > 0.05).sum()/out.size)

# 3.4.1
from scipy.stats import f

def rolling_granger_pvalue(ssr_r, ssr_ur, p, n):
    F = ((ssr_r - ssr_ur) / p) / (ssr_ur / (n - 2*p - 1))
    return 1 - f.cdf(F, p, n - 2*p - 1)

# unrestricted model: includes lags of both MSFT and NVDA
rolling_ur = RollingOLS(dataUSA_MSFT_NVDA["close_NVDA"], 
                        sm.add_constant(dataUSA_MSFT_NVDA[["close_NVDA", "close_MSFT"]].shift(1)),
                        window=60).fit()

# restricted model: includes only NVDA lags (no MSFT)
rolling_r = RollingOLS(dataUSA_MSFT_NVDA["close_NVDA"], 
                       sm.add_constant(dataUSA_MSFT_NVDA["close_NVDA"].shift(1)),
                       window=60).fit()

# compute p-values
pvals = rolling_granger_pvalue(rolling_r.ssr, rolling_ur.ssr, p=1, n=60)
pvals = pd.Series(rolling_granger_pvalue(rolling_r.ssr, rolling_ur.ssr, p=1, n=60))
pvals.plot(title="Rolling Granger causality p-values (MSFT causing NVDA)")

