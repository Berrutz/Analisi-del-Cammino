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

