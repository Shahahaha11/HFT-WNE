import pandas as pd
import statsmodels.api as sm
from arch.unitroot import ADF, PhillipsPerron, KPSS
from statsmodels.tsa.stattools import grangercausalitytests

# the function to get residuals from OLS regression
def _eg_residuals(df: pd.DataFrame, col1: str, col2: str):
    X = sm.add_constant(df[col1].values, has_constant="add")
    y = df[col2].values
    model = sm.OLS(y, X).fit()
    return model.resid

# the function for ADF p-value
def eg_adf_pvalue(df: pd.DataFrame, col1: str, col2: str, 
                  trend: str = "c", adf_lags: int = 5) -> float:
    resid = _eg_residuals(df, col1, col2)
    return float(ADF(resid, lags=adf_lags, trend=trend).pvalue)

# the function for PP p-value
def eg_pp_pvalue(df: pd.DataFrame, col1: str, col2: str, 
                 trend: str = "c") -> float:
    resid = _eg_residuals(df, col1, col2)
    return float(PhillipsPerron(resid, trend=trend).pvalue)

# the function for KPSS p-value
def eg_kpss_pvalue(df: pd.DataFrame, col1: str, col2: str, 
                   trend: str = "c") -> float:
    resid = _eg_residuals(df, col1, col2)
    return float(KPSS(resid, trend=trend).pvalue)

def granger_pvalue(df: pd.DataFrame, # dataframe with data
                   col1: str, # causing variable
                   col2: str, # caused variable
                   maxlag: int) -> float: # maximum lag to test
    granger_test = grangercausalitytests(df[[col2, col1]], maxlag = [maxlag],
                                         verbose = False) # do NOT print the results
    p_value = granger_test[maxlag][0]['ssr_ftest'][1]
    return float(p_value)