import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import click



def tax_liability_difference(work_df, beginning_year, end_year):
    """
    Computes the average per decile of T_1(y_hat) - T_0(y)
    TODO here use the approach of the average as in the following function ??
    """

    # per decile, we compute the average of tax difference, that is of T_1(y_hat) - T_0(y) 
    result = work_df.groupby('quantile').apply(lambda x: np.average(x['tax_difference'], weights=x['wprm']))

    # We fit a quadratic polynomial to the original data
    x_values = 10*work_df["cum_weight"]/work_df['wprm'].sum() #*10 to be in [0,10]
    y_values = work_df["tax_difference"]
    coefficients = np.polyfit(x_values, y_values, 2)
    poly = np.poly1d(coefficients)
    x_interpolated = np.linspace(0, 10, 1000)   # 0 here, not 1
    y_interpolated = poly(x_interpolated)


    plt.figure()
    plt.scatter(result.index, result.values)
    plt.plot(x_interpolated, y_interpolated, label='Quadratic Polynomial Fit', color='red')
    plt.xlabel('Quantile')
    plt.ylabel('Weighted Mean of Difference')
    plt.title('Weighted Mean of Difference for Each Quantile')
    plt.xticks(range(1, 11))  
    plt.legend()
    plt.show()
    plt.savefig('../outputs/tax_liability_difference/tax_liability_difference_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    plt.close()



    plt.figure(figsize=(10, 6))
    sns.boxplot(x='quantile', y='tax_difference', data=work_df, showfliers=False)


    plt.title('Boxplot per Decile')
    plt.xlabel('Decile')
    plt.ylabel('Tax Difference')
    plt.xticks(range(1, 11))  
    plt.show()
    plt.savefig('../outputs/tax_liability_difference_boxplot/tax_liability_difference_boxplot_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    plt.close()

    

def beneficiary_reform(df, list_ETI, beginning_year, end_year):
    """
    Computes the average per decile of max{T1(y1) - T0(y1), T1(y0_hat) - T0(y0)} - R(tau,h)
    According to appendix C we approximate T1 and T0 by average tax rates such that 
    T1(y1) - T0(y1) = t1*y1 - t0*y1
    T1(y0_hat) - T0(y0) = t1*y0 - t0*y0 (We use such an approx even though we know tax_difference = T1(y0_hat) - T0(y0) to be consistent with the other term)
    """
    df_results = pd.DataFrame()

    df["difference_before"] = (df["average_tax_rate_after_reform"] - df["average_tax_rate_before_reform"])*df["total_earning"]
    
    for ETI in list_ETI:
        df["difference_after"] = (df["average_tax_rate_after_reform"] - df["average_tax_rate_before_reform"])*df[f"total_earning_after_reform_{ETI}"]
        df['max_difference'] = np.maximum(df['difference_after'], df['difference_before'])
        
        total_revenue_effect = np.average(df[f'individual_revenue_effect_{ETI}'], weights=df['wprm'])
        df["reform_effect"] = df['max_difference'] - total_revenue_effect


        # per decile, we compute the average of tax difference, that is of T_1(y_hat) - T_0(y) 
        df_results[ETI] = df.groupby('quantile').apply(lambda x: np.average(x['reform_effect'], weights=x['wprm']))

    plt.figure()
    for col in df_results.columns:
        plt.scatter(df_results.index, df_results[col], label=f'Elasticity {col}')
    plt.title('Evolution of Elasticity Values Across Deciles')
    plt.xlabel('Decile')
    plt.ylabel('Elasticity Value')
    plt.legend()
    plt.xticks(range(1, 11))  
    plt.show()
    plt.savefig('../outputs/beneficiaries_decile/beneficiaries_decile_{beginning_year}-{end_year}.png'.format(beginning_year = beginning_year, end_year = end_year))
    plt.close()


@click.command()
@click.option('-y', '--beginning_year', default = None, type = int, required = True)
@click.option('-e', '--end_year', default = -1, type = int, required = True)
def main_function(beginning_year = None, end_year = None):
    if end_year == -1:
        end_year = beginning_year + 1 #reform phased in over 2 years only 

    # read csv
    people_df = pd.read_csv(f'excel/{beginning_year}-{end_year}/people_adults_{beginning_year}-{end_year}.csv')

    # we sort the dataframe by earnings so that the cumulative sum of the weights gives info about deciles
    work_df = people_df.sort_values(by='earnings_rank')
    work_df['cum_weight'] = work_df['wprm'].cumsum()

    # we compute the total_weight that helps us define the quantiles
    total_weight = work_df['wprm'].sum()
    quantiles = [total_weight/10*i for i in range(1,11)]
    work_df['quantile'] = np.searchsorted(quantiles, work_df['cum_weight']) + 1
    work_df.loc[work_df['cum_weight'] > quantiles[-1], 'quantile'] = 10
    
    # First, plot average difference in tax liability per decile
    tax_liability_difference(work_df, beginning_year, end_year)

    # Then look at whether people were beneficiaries or losers of a reform 
    list_ETI = [0., 0.25, 1., 1.25]
    beneficiary_reform(work_df, list_ETI, beginning_year, end_year)

        
main_function()
