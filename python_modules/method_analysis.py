# IMPORTS
import numpy as np
import pandas as pd
import tstools.analysis as ts_analysis

import matplotlib.pyplot as plt
import scipy.fftpack as sft
import helper as helper
from statsmodels.tsa.seasonal import seasonal_decompose
import tstools.analysis as tsa
from scipy.signal import argrelextrema

def method_2():
    raise Exception('Not implemented')


def method_3():
    raise Exception('Not implemented')



def method_1(seriesList, periods):

    if not isinstance(seriesList, list):
        raise Exception('seriesList must be a list of series')
    
    if isinstance(periods, np.ndarray):
        periods = periods.tolist()

    if not isinstance(periods, list):
        periods = [periods]
    
    if len(periods) < 1:
        raise Exception('No periods givens')

    if len(seriesList) < 2:
        raise Exception('There should be at least 2 series in series list')

    # for each period
    errors = []
    for j, period in enumerate(periods):

        # cannot process the period 0 (seasonal decompose cannot accept it)
        if period == 0:
            raise Exception('the 0 period cannot be processed')

        # periods must be an integer value
        if not isinstance(period, int):
            raise Exception('period {} is not an integer value'.format(period))

        # compute the error between each series
        aux_errors = []
        for idx, series in enumerate(seriesList):
            season = seasonal_decompose(series, period=period).seasonal

            for idx in range(idx+1, len(seriesList)):
                aux_season = seasonal_decompose(seriesList[idx], period=period).seasonal
                aux_errors.append(
                    __freq_difference(
                        season,
                        aux_season
                    )
                )
        errors.append(np.mean(aux_errors))

    return np.min(errors), periods[np.argmin(errors)], errors


def __freq_difference(season_1: list, season_2: list):

    # check if the seasons are list and try to convert if not
    season_1 = helper.convert_to_list(season_1)
    season_2 = helper.convert_to_list(season_2)
    if season_1 == None or season_2 == None:
        raise Exception('Seasons must be list type, cannot cast {} to list'.format( type(season_1) if season_1 == None else type(season_2)))

    # get the min length
    min_len = 0
    if len(season_1) < len(season_2):
        min_len = len(season_1)
    else:
        min_len = len(season_2)

    # computing the Fast Fourier Trasform of the seasonality
    fft_season_1 = sft.fft(season_1, min_len)
    fft_season_2 = sft.fft(season_2, min_len)

    fft_season_1_m = np.abs(fft_season_1) / len(season_1)
    fft_season_2_m = np.abs(fft_season_2) / len(season_2)

    # difference bethween the seasonality
    fft_diff = []
    for idx in range(min_len//2):
        fft_diff.append(fft_season_1_m[idx] - fft_season_2_m[idx])
    
    # Mean Squared Error
    return np.power(fft_diff, 2).mean() 


def method_1_stepAnalyzer(errors: list, periods: list, method='m', window = 30):
    """
    Analyze the periods step errors by the difference between the error of one period and the previous one

    arguments
    ---------
        method: str
            - m     -> find the chages using the mean and std over the entire series
            - rm    -> find the chages using the rolling mean and rolling std (require window)
            - lm    -> find the local minima
            - lmb   -> find the local minima with bounds (require window)
    """
    
    if len(errors) != len(periods):
        raise Exception('The errors list length must match the periods list length')
    
    if not isinstance(errors, np.ndarray):
        errors = np.array(errors)

    diffs = errors[1:len(errors)] - errors[:len(errors)-1]

    errors_jump = []
    if method == 'm':
        upper_bound = diffs.mean() + diffs.std()
        lower_bound = diffs.mean() - diffs.std()

        for idx, diff in enumerate(diffs):
            if diff > upper_bound or diff < lower_bound:
                errors_jump.append((periods[idx+1]))
    
    elif method == 'rm':
        rolling_mean = __rolling(diffs, window, np.mean)
        rolling_std = __rolling(diffs, window, np.std)

        upper_bound = rolling_mean + rolling_std
        lower_bound = rolling_mean - rolling_std

        for idx, diff in enumerate(diffs):
            if diff > upper_bound[idx] or diff < lower_bound[idx]:
                errors_jump.append((periods[idx+1]))
    
    elif method == 'lm':
        errors_jump = periods[argrelextrema(errors, np.less)[0]]
    
    elif method == 'lmb':
        if window < 2:
            raise Exception('window ({}) must be greater or equals to 2')

        arg_locals = argrelextrema(errors, np.less)

        left = window//2
        right = window//2
        
        for local_idx in arg_locals[0]:

            mean = errors[local_idx: right if local_idx + right <= 0 else len(errors)-1].mean()

            std = errors[local_idx: right if local_idx + right <= 0 else len(errors)-1].std()

            if errors[local_idx] < (mean - std):
                errors_jump.append(periods[local_idx])

    else:
        raise Exception('method {} not supported'.format(method))
        
        

    return errors_jump


def __rolling(series: np.ndarray, window: int, rol_func):
    rol_series = []
    for idx in range(0, len(series)-window+1):
        rol_series.append(rol_func(series[idx : window+idx]))
    
    for idx in range(window-1):
        rol_series.append(rol_series[len(rol_series)-1])

    return np.array(rol_series)



def run_analysis_m1(series_list, periods, title = ""):

    series_list, periods = check_periods_and_list(series_list, periods)
            
    # run analysis method
    best_error, best_period, errors = method_1(series_list, periods)
    out_periods = method_1_stepAnalyzer(errors, periods)
    
    # plot
    plt.figure(figsize=(20,5))
    plt.plot(periods, errors)
    plt.title(title)
    
    xticks = [out_periods[0]]
    
    if len(out_periods) > 0:
        xticks = [out_periods[0]]
        for period in out_periods:
            plt.axvline(period, linewidth=1, color='r', linestyle='--')
            if xticks[len(xticks) - 1] == period or period - xticks[len(xticks) - 1] < 5:
                continue
            xticks.append(period)

    plt.xticks(xticks)
    plt.show()

def check_periods_and_list(series_list, periods):
    if not isinstance(series_list[0], list):
        first_half = series_list[:len(series_list)//2]
        second_half = series_list[len(series_list)//2:]
        series_list = [first_half, second_half]
        lengh_first_half = len(first_half)
        second_first_half = len(second_half)
        min_len = lengh_first_half if lengh_first_half < second_first_half else second_first_half
    else:
        # get the lenght of the list with min lenght
        min_len = len(series_list[0])
        for idx, series in enumerate(series_list):
            lenght = len(series)
            if lenght < min_len:
                min_len = lenght
                
    if periods[len(periods)-1] > (min_len//2)-1 :
            periods = helper.intspace(1, min_len//2+1 )
    return series_list,periods


def run_analysis_m1_rolling(series_list, periods, title = ""):

    series_list, periods = check_periods_and_list(series_list, periods)

    # run analysis method
    best_error, best_period, errors = method_1(series_list, periods)
    out_periods = method_1_stepAnalyzer(errors, periods, method='rm', window = 15)

    # plot
    plt.figure(figsize=(20,5))
    plt.plot(periods, errors)
    plt.title(title)
    
    if len(out_periods) > 0:
        xticks = [out_periods[0]]
        for period in out_periods:
            plt.axvline(period, linewidth=1, color='r', linestyle='--')
            if xticks[len(xticks) - 1] == period or period - xticks[len(xticks) - 1] < 5:
                continue
            xticks.append(period)

    plt.xticks(xticks)
    plt.show()
    
'''
if __name__ == '__main__':
    periods = helper.intspace(-3, 3)
    errors = [33, 40, 38, 50, 2, 10]
    err_j = method_1_stepAnalyzer(errors, periods, method='lmb', window=2)
    
    print(periods)
    print(errors)
    print(err_j)
'''