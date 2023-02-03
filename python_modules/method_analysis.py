# IMPORTS
import numpy as np
import pandas as pd
import tstools.analysis as ts_analysis

import matplotlib.pyplot as plt
import scipy.fftpack as sft
import helper as helper
from statsmodels.tsa.seasonal import seasonal_decompose



def method_2():
    raise Exception('Not implemented')


def method_3():
    raise Exception('Not implemented')








def method_1(seriesList, periods):

    # get the lenght of the list with min lenght
    min_len = len(seriesList[0])
    for idx, series in enumerate(seriesList):
        lenght = len(series)
        if lenght < min_len:
            min_len = lenght

    # convert periods in list if is not
    if not isinstance(periods, list):
        periods = [periods]


    # for each period
    errors = []
    for j, period in enumerate(periods):

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

    # difference bethween the seasonality
    fft_diff = []
    for idx in range(min_len):
        fft_diff.append(np.abs(fft_season_1[idx]) - np.abs(fft_season_2[idx]))
    
    # Mean Squared Error
    return np.power(fft_diff, 2).mean() 