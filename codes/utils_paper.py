import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kernel_regression import KernelReg
import seaborn as sns

import click

pd.options.display.max_columns = None


def tax_liability_difference(work_df, beginning_year, end_year):
    """
    Computes the average per decile of T_1(y_hat) - T_0(y)
    We could have worked on tax_difference but to be consistent with the following functions we work with : 
    t1*y0 - t0*y0 = (t1 - t0)y0

    Then plots a boxplot of this difference to see heterogeneity within deciles
    """

    work_df["difference_recomputed"] = (work_df["average_tax_rate_after_reform"] - work_df["average_tax_rate_before_reform"])*work_df["total_earning"]

    # per decile, we compute the average of tax difference, that is of T_1(y_hat) - T_0(y) 
    result = work_df.groupby('decile').apply(lambda x: np.average(x['difference_recomputed'], weights=x['wprm']))

    # We fit a quadratic polynomial to the original data
    x_values = 10*work_df["cum_weight"]/(work_df['wprm'].sum()) #*10 to be in [0,10]
    y_values = work_df["difference_recomputed"]
    coefficients = np.polyfit(x_values, y_values, 2) # quadratic
    poly = np.poly1d(coefficients)
    x_interpolated = np.linspace(0, 10, 1000)   # 0 here, not 1
    y_interpolated = poly(x_interpolated)


    plt.figure()
    plt.scatter(result.index, result.values, label = "Average per decile of the difference in tax liability", color = "blue")
    plt.plot(x_interpolated, y_interpolated, label='Quadratic Polynomial Fit', color='red')
    plt.title('Changes in Tax Liability: Average Values per Decile')
    plt.xticks(range(1, 11))  
    plt.legend()
    plt.show()
    plt.savefig('../outputs/tax_liability_difference/tax_liability_difference_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    plt.close()



    plt.figure(figsize=(10, 6))
    sns.boxplot(x='decile', y='difference_recomputed', data=work_df, showfliers=False)
    plt.title('Changes in Tax Liability: Heterogeneity within Deciles')
    plt.xticks(range(1, 11))  
    plt.show()
    plt.savefig('../outputs/tax_liability_difference_boxplot/tax_liability_difference_boxplot_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    plt.close()

    

def beneficiary_reform(df, list_ETI, beginning_year, end_year):
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

        df_results[ETI] = df.groupby('decile').apply(lambda x: np.average(x['reform_effect'], weights=x['wprm']))

    plt.figure()
    for col in df_results.columns:
        plt.scatter(df_results.index, df_results[col], label=f'Elasticity {col}')
    plt.title('Winners and Losers of Major French Tax Reforms')
    plt.legend()
    plt.xticks(range(1, 11))  
    plt.show()
    plt.savefig('../outputs/beneficiaries_decile/beneficiaries_decile_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    plt.close()

    plt.figure()
    total_weight = df['wprm'].sum()
    median_df = df[(df["cum_weight"] >= 0.45*total_weight) & (df["cum_weight"] <= 0.55*total_weight)]

    for ETI in list_ETI:
        percentage_winners = 100*np.average(df[f"winner_{ETI}"], weights=df['wprm'])
        percentage_winners_median = 100*np.average(median_df[f"winner_{ETI}"], weights=median_df['wprm'])
        plt.scatter(percentage_winners_median, percentage_winners, label=ETI)

    x = np.linspace(0,100,100)
    plt.plot(x, x, linestyle='--') 
    plt.hlines(y = 50, xmin = 0, xmax = 100)
    plt.vlines(x = 50, ymin = 0, ymax = 100)
    plt.title('Majority Support versus Support by the Median Voter')
    plt.xlabel('Median support')
    plt.ylabel('Whole population support')
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.legend() 
    plt.show()
    plt.savefig('../outputs/alignment_support/alignment_support_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    plt.close()


def increased_progressivity(df, beginning_year, end_year):
    """
    Computes the average per decile of T'/(1-T') where T' marginal tax rate
    """

    # per decile, we compute the average of T'/(1-T')
    result_before = df.groupby('decile').apply(lambda x: np.average(x['mtr_ratio_before'], weights=x['wprm']))
    result_after = df.groupby('decile').apply(lambda x: np.average(x['mtr_ratio_after'], weights=x['wprm']))


    plt.figure()
    plt.scatter(result_before.index, result_before.values, c = "blue", label = beginning_year)
    plt.scatter(result_after.index, result_after.values, c = "red", label = end_year)
    plt.title("T'/(1-T') by Decile before and after Each Reform")
    plt.xticks(range(1, 11))  
    plt.legend()
    plt.show()
    plt.savefig('../outputs/increased_progressivity/increased_progressivity_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
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


def robustness_check_pareto_bound(grid_earnings, cdf, pdf, beginning_year, end_year):
    """
    Plots components of the Pareto bound separately, that is plots :
    - the tail distribution/survival function 1 - cdf
    - the density pdf 
    - the ratio of the two that is almost the upper Pareto bound (still need to account for ETI)
    """
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    axs[0].plot(grid_earnings, 1 - cdf, label='1 - cdf')
    axs[0].set_title('1 - cdf')

    axs[1].plot(grid_earnings, pdf, label='pdf')
    axs[1].set_title('pdf')

    axs[2].plot(grid_earnings, (1 - cdf) / pdf, label='(1 - cdf) / pdf')
    axs[2].set_title('(1 - cdf) / pdf')

    for ax in axs:
        ax.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('../outputs/pareto_bound_robustness/pareto_bound_robustness_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
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
    work_df.sort_values(by="total_earning", inplace=True)
    work_df = work_df[work_df["total_earning"] > 0] # for this function we remove null earnings otherwise problem with grid and estimates

    total_earning = work_df["total_earning"].values # we base computations on the year before the reform
    weights = work_df["wprm"].values

    grid_earnings = np.linspace(np.percentile(total_earning, 1), np.percentile(total_earning, 99), 1000)
    
    # Pareto Bounds part
    kde = gaussian_kde(total_earning, weights=weights)    
    pdf = kde(grid_earnings)
    cdf = np.cumsum(pdf) / np.sum(pdf)

    robustness_check_pareto_bound(grid_earnings, cdf, pdf, beginning_year, end_year)

    plt.figure()
    list_ETI_upper = [0.25, 0.4, 0.5, 0.75, 1., 1.25]
    base_upper_bound = (1 - cdf)/(grid_earnings * pdf) 
    
    
    condition_threshold = (grid_earnings > np.percentile(total_earning, 15))
    
    for ETI in list_ETI_upper:
        upper_bound = base_upper_bound * 1/ETI
        plt.plot(grid_earnings[condition_threshold], upper_bound[condition_threshold], label=ETI)

    # T'/(1-T') part
    smoothed_y_primary_before = tax_ratio_by_earning(total_earning = total_earning,
                                              grid_earnings = grid_earnings, 
                                              mtr_ratio = work_df["mtr_ratio_before"].values, 
                                              weights = weights)


    smoothed_y_primary_after = tax_ratio_by_earning(total_earning = work_df["total_earning_inflated"].values, # only inflated version (from which we computed the mtr after the reform)
                                              grid_earnings = grid_earnings, 
                                              mtr_ratio = work_df["mtr_ratio_after"].values, 
                                              weights = weights)


    plt.plot(grid_earnings[condition_threshold], smoothed_y_primary_before[condition_threshold], label='MTR ratio before', color = 'blue')   
    plt.plot(grid_earnings[condition_threshold], smoothed_y_primary_after[condition_threshold], label='MTR ratio after', color = 'red')   

    
    plt.title('Upper Pareto Bound')  
    plt.legend()
    plt.show()
    plt.savefig('../outputs/upper_pareto_bound/upper_pareto_bound_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    plt.close()


    plt.figure()
    list_ETI_lower = [5, 4, 3, 2]
    base_lower_bound = - cdf/(grid_earnings * pdf) 
    
    for ETI in list_ETI_lower:
        lower_bound = base_lower_bound * 1/ETI
        plt.plot(grid_earnings, lower_bound, label=ETI)

    plt.plot(grid_earnings, smoothed_y_primary_before, label='MTR ratio before', color = 'blue')   
    plt.plot(grid_earnings, smoothed_y_primary_after, label='MTR ratio after', color = 'red')   

    
    plt.title('Lower Pareto Bound')  
    plt.legend()
    plt.show()
    plt.savefig('../outputs/lower_pareto_bound/lower_pareto_bound_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    plt.close()







@click.command()
@click.option('-y', '--beginning_year', default = None, type = int, required = True)
@click.option('-e', '--end_year', default = -1, type = int, required = True)
def main_function(beginning_year = None, end_year = None):
    if end_year == -1:
        end_year = beginning_year + 1 #reform phased in over 2 years only 

    # read csv
    people_df = pd.read_csv(f'excel/{beginning_year}-{end_year}/people_adults_{beginning_year}-{end_year}.csv')

    # very few individuals have weight 0, we remove them (a bit weird)
    print("number outliers removed", len(people_df[people_df["wprm"] == 0]))
    people_df = people_df[people_df["wprm"] != 0]

    # we sort the dataframe by earnings so that the cumulative sum of the weights gives info about deciles
    work_df = people_df.sort_values(by='earnings_rank')
    work_df['cum_weight'] = work_df['wprm'].cumsum()

    # we compute the total_weight that helps us define the deciles
    total_weight = work_df['wprm'].sum()
    deciles = [total_weight/10*i for i in range(1,11)]
    work_df['decile'] = np.searchsorted(deciles, work_df['cum_weight']) + 1
    work_df.loc[work_df['cum_weight'] > deciles[-1], 'decile'] = 10

    work_df["mtr_ratio_before"] = work_df["marginal_tax_rate_before_reform"]/(1 - work_df["marginal_tax_rate_before_reform"])
    work_df["mtr_ratio_after"] = work_df["marginal_tax_rate_after_reform"]/(1 - work_df["marginal_tax_rate_after_reform"])
    
    # First, plot average difference in tax liability per decile
    tax_liability_difference(work_df, beginning_year, end_year)

    # Then look at whether people were beneficiaries or losers of a reform 
    list_ETI = [0., 0.25, 1., 1.25]
    beneficiary_reform(work_df, list_ETI, beginning_year, end_year)

    # then plot T'/(1-T') to see whether increased progressivity in the middle 
    increased_progressivity(work_df, beginning_year, end_year)

    # plot pareto Bounds
    pareto_bounds(work_df, beginning_year, end_year)

        
main_function()
