import pandas as pd
import numpy as np
from scipy.stats import norm
from collections import Iterable


def discount(time, discount_rate):
    """
    Returns the discounted price of a dollar at the given discount rate
    for the given time periods.

    For a time period t, the discounted price of a dollar is given by
    1/(1 + t) ^ discount_rate.

    Parameters:
    time (Iterable) - The time periods for which discounted price is to
                        be calculated.

    discount_rate (scalar/pd.Series) - Discount rate(s) per period.     

    Return:
    (pd.DataFrame) - Returns a |t| x |r| dataframe of discounted prices.
    """

    if not isinstance(time, Iterable):
        discounts = (1 + discount_rate) ** (-time)
    else:
        discounts = pd.DataFrame(
            [(1 + discount_rate) ** (-t) for t in time], index=time
        )

    return discounts


def present_value(flows, discount_rate, periods=None):
    """
    Returns the persent discounted value of future cashflows.

    Parameters:
    flows (pd.Series) - A series of future cash flows

    discount_rate (scalar/pd.Series) - Discount rate(s) per period.

    periods (pd.Series) - The time period of flows. 
                            Considers index if as to None.

    Return:
    (float) - The present value of the set of future cash flows.
    """
    if periods is None:
        periods = flows.index
        indexed_flows = flows
    else:
        indexed_flows = pd.Series(list(flows), index=periods)

    discounts = discount(periods, discount_rate)
    pv = discounts.multiply(indexed_flows, axis="index").sum()

    return pv


def compound_interest(principal, rate, n_years, periods_per_year=12):
    """
    Calculates the compound interest.

    Total = principal * (1 + rate) ** periods

    Parameters:
    principal (float): Represent the principal amount

    rate (float): The annual rate of interest.

    n_years (float): The number of years for which amount is compounded.

    periods_per_year (float): Number of periods per year.

    Return:
    (float): Total amount compounded at the given parameters.    
    """
    n_periods = n_years * periods_per_year
    return principal * (1 + rate / periods_per_year) ** n_periods


def blackscholes_to_binomial(
    ann_risk_free_rate,
    ann_volatility,
    n_periods,
    n_years,
    dividend=0.0
):
    """
    Converts the parameters of the BlackScholesModel to those of Binomial pricing model.

    The conversion to equivalent binomial model parameters is as follows:

    risk_free_return = exp(ann_risk_free_rate * n_years / n_periods)
    upward_drift = exp(ann_volatility * sqrt(n_years / n_periods))
    downward_drift = 1 / upward_drift
    dividend_per_period = dividend * n_years / n_periods

    Parameters:
    ----------

    ann_risk_free_rate: float
        The discount rate of a dollar over n_years.

    ann_volatility: float
        The standard deviation/volatility to be considered.

    n_periods: int
        The number of periods for which the model is to be constructed.

    n_years: float
        The total time T of the derivative holding.

    Returns:
    -------

    dict: The dictionary consists of four keys:
            risk_free_return
            upward_drift
            downward_drift
            dividend_per_period
    """
    risk_free_rate = np.expm1(ann_risk_free_rate * n_years / n_periods)
    upward_drift = np.exp(ann_volatility * np.sqrt(n_years / n_periods))
    downward_drift = 1 / upward_drift
    dividend_per_period = dividend * n_years / n_periods

    binomial_params = {
        "risk_free_rate": risk_free_rate,
        "upward_drift": upward_drift,
        "downward_drift": downward_drift,
        "dividend_per_period": dividend_per_period,
    }

    return binomial_params
