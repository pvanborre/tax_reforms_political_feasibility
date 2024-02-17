import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kernel_regression import KernelReg
import seaborn as sns

import click

pd.options.display.max_columns = None


def tax_liability_difference(work_df, beginning_year, end_year, name_quantile="decile", value_quantile = 10):
    """
    Computes the average per decile of T_1(y_hat) - T_0(y)
    We could have worked on tax_difference but to be consistent with the following functions we work with : 
    t1*y0 - t0*y0 = (t1 - t0)y0

    Then plots a boxplot of this difference to see heterogeneity within deciles
    """

    work_df["difference_recomputed"] = (work_df["average_tax_rate_after_reform"] - work_df["average_tax_rate_before_reform"])*work_df["total_earning"]

    # per decile, we compute the average of tax difference, that is of T_1(y_hat) - T_0(y) 
    result = work_df.groupby(name_quantile).apply(lambda x: np.average(x['difference_recomputed'], weights=x['wprm']))

    # We fit a quadratic polynomial to the original data
    x_values = value_quantile*work_df["cum_weight"]/(work_df['wprm'].sum()) #*10 to be in [0,10]
    y_values = work_df["difference_recomputed"]
    coefficients = np.polyfit(x_values, y_values, 2) # quadratic
    poly = np.poly1d(coefficients)
    x_interpolated = np.linspace(0, value_quantile, 1000)   # 0 here, not 1
    y_interpolated = poly(x_interpolated)


    plt.figure()
    plt.scatter(result.index, result.values, color = "blue")
    plt.plot(x_interpolated, y_interpolated, color='red')
    plt.xticks(range(1, value_quantile+1))  
    plt.show()
    if name_quantile == "decile":
        plt.savefig('../outputs/tax_liability_difference/tax_liability_difference_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    else:
        plt.savefig('../outputs/tax_liability_difference_{name_quantile}/tax_liability_difference_{name_quantile}_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year, name_quantile = name_quantile))

    plt.close()

    fig, ax = plt.subplots(1, 1, figsize = (10, 6), dpi=300)
    sns.boxplot(x=name_quantile, y='difference_recomputed', data=work_df, showfliers=False)
    ax.set_ylabel('')    
    ax.set_xlabel('')
    plt.xticks(range(0, value_quantile+1))  
    plt.show()
    if name_quantile == "decile":
        plt.savefig('../outputs/tax_liability_difference_boxplot/tax_liability_difference_boxplot_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    else:
        plt.savefig('../outputs/tax_liability_difference_boxplot_{name_quantile}/tax_liability_difference_boxplot_{name_quantile}_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year, name_quantile = name_quantile))

    plt.close()

    

def beneficiary_reform(df, list_ETI, beginning_year, end_year, name_quantile="decile", value_quantile = 10):
    """
    Computes the average per decile of max{T1(y1) - T0(y1), T1(y0_hat) - T0(y0)} - R(tau,h) for 4 different ETI values
    According to appendix C we approximate T1 and T0 by average tax rates such that 
    T1(y1) - T0(y1) = t1*y1 - t0*y1
    T1(y0_hat) - T0(y0) = t1*y0 - t0*y0 (We use such an approx even though we know tax_difference = T1(y0_hat) - T0(y0) to be consistent with the other term)
    
    Then we know for everybody whether he is a winner or a loser of a reform so we can compute the percentage of winners of the population
    We then check whether median support is aligned with support in the whole population (to see whether reforms are enough monotonic)
    To do so we draw a point that is (percentage winners P45-P55, percentage winners whole population) for 4 different ETI values
    """
    df_results = pd.DataFrame()

    df["difference_before"] = (df["average_tax_rate_after_reform"] - df["average_tax_rate_before_reform"])*df["total_earning"]
    
    for ETI in list_ETI:
        df["difference_after"] = (df["average_tax_rate_after_reform"] - df["average_tax_rate_before_reform"])*df[f"total_earning_after_reform_{ETI}"]
        df['max_difference'] = np.maximum(df['difference_after'], df['difference_before'])
        
        total_revenue_effect = np.average(df[f'individual_revenue_effect_{ETI}'], weights=df['wprm'])
        df["reform_effect"] = df['max_difference'] - total_revenue_effect

        df[f"winner_{ETI}"] = (df["reform_effect"] <= 0)

        df_results[ETI] = df.groupby(name_quantile).apply(lambda x: np.average(x['reform_effect'], weights=x['wprm']))

    plt.figure()
    for col in df_results.columns:
        plt.scatter(df_results.index, df_results[col])
    plt.xticks(range(1, value_quantile+1))  
    plt.show()
    if name_quantile == "decile":
        plt.savefig('../outputs/beneficiaries_decile/beneficiaries_decile_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    else:
        plt.savefig('../outputs/beneficiaries_{name_quantile}/beneficiaries_{name_quantile}_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year, name_quantile = name_quantile))

    plt.close()

    plt.figure()
    total_weight = df['wprm'].sum()
    median_df = df[(df["cum_weight"] >= 0.45*total_weight) & (df["cum_weight"] <= 0.55*total_weight)]

    for ETI in list_ETI:
        percentage_winners = 100*np.average(df[f"winner_{ETI}"], weights=df['wprm'])
        percentage_winners_median = 100*np.average(median_df[f"winner_{ETI}"], weights=median_df['wprm'])
        plt.scatter(percentage_winners_median, percentage_winners)

    x = np.linspace(0,100,100)
    plt.plot(x, x, linestyle='--') 
    plt.hlines(y = 50, xmin = 0, xmax = 100)
    plt.vlines(x = 50, ymin = 0, ymax = 100)
    # plt.xlabel('Median support')
    # plt.ylabel('Whole population support')
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.show()
    plt.savefig('../outputs/alignment_support/alignment_support_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    plt.close()


def increased_progressivity(df, beginning_year, end_year, name_quantile="decile", value_quantile = 10):
    """
    Computes the average per decile of T'/(1-T') where T' marginal tax rate
    """

    # per decile, we compute the average of T'/(1-T')
    result_before = df.groupby(name_quantile).apply(lambda x: np.average(x['mtr_ratio_before'], weights=x['wprm']))
    result_after = df.groupby(name_quantile).apply(lambda x: np.average(x['mtr_ratio_after'], weights=x['wprm']))


    plt.figure()
    plt.scatter(result_before.index, result_before.values, c = "blue")
    plt.scatter(result_after.index, result_after.values, c = "red")
    plt.xticks(range(1, value_quantile+1))  
    plt.show()
    if name_quantile == "decile":
        plt.savefig('../outputs/increased_progressivity/increased_progressivity_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    else:
        plt.savefig('../outputs/increased_progressivity_{name_quantile}/increased_progressivity_{name_quantile}_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year, name_quantile = name_quantile))
    plt.close()



def tax_ratio_by_earning(total_earning, grid_earnings, mtr_ratio, weights):
    """
    T'/(1-T') part : for each earning average all values of this ratio
    and then fit a gaussian kernel to interpolate over a given grid of earnings
    """
    
    unique_earning = np.unique(total_earning)
    mean_tax_ratios = np.zeros_like(unique_earning, dtype=float)

    # for each unique earning we average over all values of this ratio
    for i, unique_value in enumerate(unique_earning):
        indices = np.where(total_earning == unique_value)
        mean_tax_rate = np.average(mtr_ratio[indices], weights=weights[indices])
        mean_tax_ratios[i] = mean_tax_rate

    # fit a gaussian kernel
    bandwidth = 5000
    kernel_reg = KernelReg(endog=mean_tax_ratios, exog=unique_earning, var_type='c', reg_type='ll', bw=[bandwidth], ckertype='gaussian')
    smoothed_y_primary, _ = kernel_reg.fit(grid_earnings)
    return smoothed_y_primary


def distribution_checks(grid_earnings, cdf, pdf, beginning_year, end_year):
    """
    Checks the aspect of the earnings distribution by means of : 
    - the cumulative distribution function cdf
    - the density pdf 
    - the inverse hazard rate (1-cdf)/pdf
    """

    plt.figure()
    plt.plot(grid_earnings, cdf)
    plt.show()
    plt.savefig('../outputs/cdf/cdf_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    plt.close()

    plt.figure()
    plt.plot(grid_earnings, pdf)
    plt.show()
    plt.savefig('../outputs/pdf/pdf_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    plt.close()

    plt.figure()
    plt.plot(grid_earnings, (1-cdf)/pdf)
    plt.show()
    plt.savefig('../outputs/inverse_hazard_rate/inverse_hazard_rate_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    plt.close()

def pareto_bounds(df, beginning_year, end_year):
    """
    Plots Pareto Bounds
    D_up = (1 - cdf)/y.pdf * 1/ETI
    D_low =  - cdf/y.pdf * 1/ETI

    On the same graph also plot T'/(1-T') before and after the reform 

    To do so we define a grid of earnings on which we fit a gaussian to get the density pdf, from which we deduce the cdf
    For the T'/(1-T') part : at a given earning we average this ratio among all individuals that earn this earning, and then interpolate data with a gaussian kernel.
    """

    work_df = df.copy()
    total_earning = work_df["total_earning"].values # we base computations on the year before the reform
    weights = work_df["wprm"].values
    centiles = work_df["centile"].values

    # we compute weighted centiles and store them in a dictionary values_centiles
    values_centiles = {}
    tab_values = [1, 5, 8, 10, 15, 20, 25, 35, 50, 75, 90, 95, 99]
    for value in tab_values:
        values_centiles[value] = np.mean(total_earning[centiles == value])

    grid_earnings = np.linspace(values_centiles[1], values_centiles[99], 1000)
    
    # Pareto Bounds part
    kde = gaussian_kde(total_earning, weights=weights)    
    pdf = kde(grid_earnings)
    cdf = np.cumsum(pdf) / np.sum(pdf)

    distribution_checks(grid_earnings, cdf, pdf, beginning_year, end_year)

    # we kept null earnings for the cdf, pdf, now we remove them for the bound
    pdf = pdf[grid_earnings > 0]
    cdf = cdf[grid_earnings > 0]
    grid_earnings = grid_earnings[grid_earnings > 0]

    plt.figure()
    list_ETI_upper = [0.25, 0.5, 1., 1.25]  
    base_upper_bound = (1 - cdf)/(grid_earnings * pdf) 
    
    max_earnings = 60000
    max_y = 3
    condition_threshold = (grid_earnings > values_centiles[20]) & (grid_earnings < max_earnings)
    
    for ETI in list_ETI_upper:
        upper_bound = base_upper_bound * 1/ETI
        plt.plot(grid_earnings[condition_threshold], upper_bound[condition_threshold])

    # we add vertical lines to the graph to indicate percentiles
    tab_percentile = [25, 50, 75, 90, 95]
    for percentile in tab_percentile:
        plt.vlines(x=values_centiles[percentile], ymin=0, ymax=max_y-1, colors='grey', ls='--', lw=0.5)
        plt.text(values_centiles[percentile], max_y-1, f'P{percentile}', horizontalalignment='center')

    # T'/(1-T') part
    smoothed_y_primary_before = tax_ratio_by_earning(total_earning = total_earning,
                                              grid_earnings = grid_earnings, 
                                              mtr_ratio = work_df["mtr_ratio_before"].values, 
                                              weights = weights)


    smoothed_y_primary_after = tax_ratio_by_earning(total_earning = work_df["total_earning_inflated"].values, # only inflated version (from which we computed the mtr after the reform)
                                              grid_earnings = grid_earnings, 
                                              mtr_ratio = work_df["mtr_ratio_after"].values, 
                                              weights = weights)


    plt.plot(grid_earnings[condition_threshold], smoothed_y_primary_before[condition_threshold], color = 'blue')   
    plt.plot(grid_earnings[condition_threshold], smoothed_y_primary_after[condition_threshold], color = 'black')   
    plt.ylim(0, max_y)
    plt.show()
    plt.savefig('../outputs/upper_pareto_bound/upper_pareto_bound_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    plt.close()


    plt.figure()
    list_ETI_lower = [5, 4, 3, 2]
    base_lower_bound = - cdf/(grid_earnings * pdf) 

    condition_threshold_low =  (grid_earnings > values_centiles[8]) & (grid_earnings < values_centiles[35])
    
    for ETI in list_ETI_lower:
        lower_bound = base_lower_bound * 1/ETI
        plt.plot(grid_earnings[condition_threshold_low], lower_bound[condition_threshold_low])

    max_y_lower = 0.25
    min_y_lower = -1
    
    tab_percentile_low = [10, 25]
    for percentile in tab_percentile_low:
        plt.vlines(x=values_centiles[percentile], ymin=min_y_lower, ymax=max_y_lower-0.35, colors='grey', ls='--', lw=0.5)
        plt.text(values_centiles[percentile], max_y_lower-0.35, f'P{percentile}', horizontalalignment='center')

    plt.plot(grid_earnings[condition_threshold_low], smoothed_y_primary_before[condition_threshold_low], color = 'blue')   
    plt.plot(grid_earnings[condition_threshold_low], smoothed_y_primary_after[condition_threshold_low], color = 'black')   
    plt.ylim(min_y_lower, max_y_lower) 
    plt.show()
    plt.savefig('../outputs/lower_pareto_bound/lower_pareto_bound_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    plt.close()







@click.command()
@click.option('-y', '--beginning_year', default = None, type = int, required = True)
@click.option('-e', '--end_year', default = -1, type = int, required = True)
def main_function(beginning_year = None, end_year = None):
    if end_year == -1:
        end_year = beginning_year + 1 #reform phased in over 2 years only 
    print("years under consideration", beginning_year, end_year)

    # read csv
    people_df = pd.read_csv(f'excel/{beginning_year}-{end_year}/people_adults_{beginning_year}-{end_year}.csv')

    # very few individuals have weight 0, we remove them (a bit weird)
    print("number outliers removed : null weights", len(people_df[people_df["wprm"] == 0]))
    people_df = people_df[people_df["wprm"] != 0]

    print("number outliers removed : negative earnings", len(people_df[people_df["total_earning"] < 0]))
    people_df = people_df[people_df["total_earning"] >= 0] # we remove negative earnings 


    # we sort the dataframe by earnings so that the cumulative sum of the weights gives info about deciles
    work_df = people_df.sort_values(by='earnings_rank')
    work_df['cum_weight'] = work_df['wprm'].cumsum()

    # we compute the total_weight that helps us define the deciles and centiles
    total_weight = work_df['wprm'].sum()
    tab_names_quantiles = ["decile", "centile", "vingtile"]
    tab_values_quantiles = [10, 100, 20]
    for j in range(len(tab_names_quantiles)):
        quantiles = [total_weight/tab_values_quantiles[j]*i for i in range(1,tab_values_quantiles[j]+1)]
        work_df[tab_names_quantiles[j]] = np.searchsorted(quantiles, work_df['cum_weight']) + 1
        work_df.loc[work_df['cum_weight'] > quantiles[-1], tab_names_quantiles[j]] = tab_values_quantiles[j]

    work_df["mtr_ratio_before"] = work_df["marginal_tax_rate_before_reform"]/(1 - work_df["marginal_tax_rate_before_reform"])
    work_df["mtr_ratio_after"] = work_df["marginal_tax_rate_after_reform"]/(1 - work_df["marginal_tax_rate_after_reform"])
    
    # First, plot average difference in tax liability per decile
    tax_liability_difference(work_df, beginning_year, end_year)
    tax_liability_difference(work_df, beginning_year, end_year, name_quantile="vingtile", value_quantile = 20)

    # Then look at whether people were beneficiaries or losers of a reform 
    list_ETI = [0., 0.25, 1., 1.25]
    beneficiary_reform(work_df, list_ETI, beginning_year, end_year)
    beneficiary_reform(work_df, list_ETI, beginning_year, end_year, name_quantile="vingtile", value_quantile = 20)

    # then plot T'/(1-T') to see whether increased progressivity in the middle 
    increased_progressivity(work_df, beginning_year, end_year)
    increased_progressivity(work_df, beginning_year, end_year, name_quantile="vingtile", value_quantile = 20)

    # plot pareto Bounds
    pareto_bounds(work_df, beginning_year, end_year)

        
main_function()
