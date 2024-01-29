import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import click



def weighted_mean_tax_difference(df):
    return np.average(df['tax_difference'], weights=df['wprm'])


def tax_liability_difference(df, beginning_year, end_year):
    """
    Computes the average per decile of T_1(y_hat) - T_0(y)
    """
    # we sort the dataframe by earnings so that the cumulative sum of the weights gives info about deciles
    work_df = df.sort_values(by='earnings_rank')
    work_df['cum_weight'] = work_df['wprm'].cumsum()

    # we compute the total_weight that helps us define the quantiles
    total_weight = work_df['wprm'].sum()
    quantiles = [total_weight/10*i for i in range(1,11)]
    work_df['quantile'] = np.searchsorted(quantiles, work_df['cum_weight']) + 1
    work_df.loc[work_df['cum_weight'] > quantiles[-1], 'quantile'] = 10

    # per decile, we compute the average of tax difference, that is of T_1(y_hat) - T_0(y) 
    result = work_df.groupby('quantile').apply(weighted_mean_tax_difference)

    # We fit a quadratic polynomial to the original data
    x_values = 10*work_df["cum_weight"]/total_weight #*10 to be in [0,10]
    y_values = work_df["tax_difference"]
    coefficients = np.polyfit(x_values, y_values, 2)
    poly = np.poly1d(coefficients)
    x_interpolated = np.linspace(0, 10, 1000)   # 0 here, not 1
    y_interpolated = poly(x_interpolated)



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



    


@click.command()
@click.option('-y', '--beginning_year', default = None, type = int, required = True)
@click.option('-e', '--end_year', default = -1, type = int, required = True)
def main_function(beginning_year = None, end_year = None):
    if end_year == -1:
        end_year = beginning_year + 1 #reform phased in over 2 years only 

    # read csv
    people_df = pd.read_csv(f'excel/{beginning_year}-{end_year}/people_adults_{beginning_year}-{end_year}.csv')

    
    # First, plot average difference in tax liability per decile
    people_df_restricted = people_df[["earnings_rank", "tax_difference", "wprm"]]
    tax_liability_difference(people_df_restricted, beginning_year, end_year)
        
main_function()
