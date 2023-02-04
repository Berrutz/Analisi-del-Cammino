# IMPORTS
import numpy as np
import pandas as pd
import tstools.analysis as ts_analysis

import matplotlib.pyplot as plt
import scipy.fftpack as sft
import helper as helper
from statsmodels.tsa.seasonal import seasonal_decompose
import tstools.analysis as tsa


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

    # get the lenght of the list with min lenght
    min_len = len(seriesList[0])
    for idx, series in enumerate(seriesList):
        lenght = len(series)
        if lenght < min_len:
            min_len = lenght

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


def method_1_stepAnalyzer(errors: list, periods: list, rolling = False, window = 30):
    """
    Analyze the periods step errors by the difference between the error of one period and the previous one
    """

    if len(errors) != len(periods):
        raise Exception('The errors list length must match the periods list length')
    
    if not isinstance(errors, np.ndarray):
        errors = np.array(errors)

    diffs = errors[1:len(errors)] - errors[:len(errors)-1]

    errors_jump = []
    if rolling:
        rolling_mean = __rolling(diffs, window, np.mean)
        rolling_std = __rolling(diffs, window, np.std)

        upper_bound = rolling_mean + rolling_std
        lower_bound = rolling_mean - rolling_std

        for idx, diff in enumerate(diffs):
            if diff > upper_bound[idx] or diff < lower_bound[idx]:
                errors_jump.append((periods[idx+1]))
    else:
        upper_bound = diffs.mean() + diffs.std()
        lower_bound = diffs.mean() - diffs.std()

        for idx, diff in enumerate(diffs):
            if diff > upper_bound or diff < lower_bound:
                errors_jump.append((periods[idx+1]))
        

    return errors_jump


def __rolling(series: np.ndarray, window: int, rol_func):
    rol_series = []
    for idx in range(0, len(series)-window+1):
        rol_series.append(rol_func(series[idx : window+idx]))
    
    for idx in range(window-1):
        rol_series.append(rol_series[len(rol_series)-1])

    return np.array(rol_series)


if __name__ == '__main__':
    periods = helper.intspace(-3, 3)
    errors = [13, 17, 20, 50, 37, 4]
    err_j = method_1_stepAnalyzer(errors, periods, True, 3)
    
    print(periods)
    print(errors)
    print(err_j)