import numpy as np
import pandas as pd
import tstools.analysis as ts_analysis

import matplotlib.pyplot as plt
import scipy.fftpack as sft
import helper as helper
from statsmodels.tsa.seasonal import seasonal_decompose


@DeprecationWarning
def low_pass_filter(X, n_order, cutoff_freq):
    """ Nostra implementazione del filtro di Buttherworth
    """

    return 1 / (1 + np.power(X/cutoff_freq, 2*n_order))

@DeprecationWarning
def filter_dataframe(dataframe: pd.DataFrame, filter_n_order, filter_cutoff_freq):
    """ Filtra un intero dataframe utilizazndo il finltro di Butterworth 
    """
    
    new_dataframe = pd.DataFrame()

    for idx, column_name in enumerate(dataframe):
        series = helper.delete_nan(dataframe[column_name])
        series_fft = sft.fft(series)
        series_fft = series_fft * low_pass_filter(helper.intspace_(len(series_fft)), filter_n_order, filter_cutoff_freq)
        series_ifft = np.abs( sft.ifft(series_fft) )
        
        new_dataframe = pd.concat([new_dataframe, pd.DataFrame({column_name: series_ifft})], axis=1)

    return new_dataframe


# TODO: ????
def filter_dataframe_shifted(dataframe: pd.DataFrame, filter_n_order, filter_cutoff_freq):
    new_dataframe = pd.DataFrame()

    for idx, column_name in enumerate(dataframe):
        series = helper.delete_nan(dataframe[column_name])
        series_fft = sft.fft(series)
        series_fft = sft.fftshift(series_fft)

        if column_name == 'y_piede_dx_1':
            plt.figure(figsize=(15, 10))
            plt.subplot(2,1,1)
            plt.title('before filtering')
            plt.plot(series_fft.real, label='real')
            plt.plot(series_fft.imag, label='imag')
            plt.legend()

        series_fft = series_fft * low_pass_filter(
            helper.intspace_( len(series_fft) )
        , filter_n_order, filter_cutoff_freq)
        
        if column_name == 'y_piede_dx_1':
            plt.subplot(2,1,2)
            plt.title('after filtering')
            plt.plot(series_fft.real, label='real')
            plt.plot(series_fft.imag, label='imag')
            plt.legend()

        

        series_ifft = np.abs( sft.ifft(series_fft) )
        
        new_dataframe = pd.concat([new_dataframe, pd.DataFrame({column_name: series_ifft})], axis=1)

    return new_dataframe


def freq_difference(season_1: list, season_2: list):

    # check if the seasons are list and try to convert if not
    season_1 = helper.convert_to_list(season_1)
    season_2 = helper.convert_to_list(season_2)
    if season_1 == None or season_2 == None:
        raise Exception('Seasons must be list type, cannot cast {} to list'.format( type(season_1) if season_1 == None else type(season_2)))

    # computing the Fast Fourier Trasform of the seasonality
    fft_season_1 = sft.fft(season_1)
    fft_season_2 = sft.fft(season_2)

    if len(fft_season_1) > len(fft_season_2):
        sub_len = len(fft_season_2)
    else:
        sub_len = len(fft_season_1)

    fft_diff = []
    for idx in range(sub_len):
        # fft_diff.append(fft_season_1[idx] - fft_season_2[idx])
        fft_diff.append(np.abs(fft_season_1[idx]) - np.abs(fft_season_2[idx]))
    
    # return np.power(np.abs(fft_diff), 2).mean() 
    return np.power(fft_diff, 2).mean() 


def best_seasonal_error(seriesList, periods):

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
                    freq_difference(
                        season,
                        aux_season
                    )
                )
        
        # Debugging logs
        # print('period: {}  |  mean: {}'.format(period, np.mean(aux_errors)))
        # print(aux_errors, "\n")
        
        errors.append(np.mean(aux_errors))
        
    return np.min(errors), periods[np.argmin(errors)]
