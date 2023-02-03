# Import the required modules
import xmltodict
import re

import pandas as pd

import numpy as np

import time

import math 

from scipy import signal

import os
from os import path
from os import walk

# import sys
# sys.path.insert(1, '{}/python_modules'.format(os.getcwd()))
# import helper as helper
# import project_analysis as analysis



# global variables setted by the parsing methods
datasets_root: str
datasets_extension : str
datasets_name_patter: str = None

filter_cutoff_freq: float
filter_order: int
filter_type: str

sampling_frequency: float




def getOrRaise(key, dictionary, errorMessage=None):
    value = dictionary.get(key)
    if value == None:
        raise Exception('{} tag not found'.format(key) if errorMessage == None else errorMessage)
    return value


def parseXML(filter_config_xml):
    parseFilter(xmltodict.parse(filter_config_xml))  # Use xmltodict to parse and convert the XML document


def parseFilter(main_dict):
    filter_dict = getOrRaise('filter', main_dict)

    parseFilterConfig(filter_dict)
    validateAndSetDatasetsRoot(getOrRaise('datasets-root', filter_dict))
    validateAndSetDatasetsExtension(getOrRaise('datasets-extension', filter_dict))
    
    for key_name in filter_dict.keys():
        if key_name == 'datasets-name-match':
            validateAndSetDatasetsNameMatch(getOrRaise(key_name, filter_dict))
            

def validateAndSetDatasetsNameMatch(pattern_value):
    global datasets_name_patter
    datasets_name_patter = pattern_value 


def parseFilterConfig(filter_dict):
    filter_conf_dict = getOrRaise('filter-config', filter_dict)
    
    validateAndSetFilterCutoff(getOrRaise('filter-cutoff', filter_conf_dict))
    validateAndSetFilterType(getOrRaise('filter-type', filter_conf_dict))
    validateAndSetFilterOrder(getOrRaise('filter-order', filter_conf_dict))
    validateAndSetSamplingFrequency(getOrRaise('sampling-frequency', filter_conf_dict))


def validateAndSetFilterCutoff(cutoff_value):
    filter_cutoff_value:float
    try:
        filter_cutoff_value = float(cutoff_value)
    except:
        raise Exception('filter cutoff is not a valid number')

    if filter_cutoff_value <= 0.01:
        raise Exception('filter cutoff cannot be a negative number or lower then 0.01')
    
    global filter_cutoff_freq 
    filter_cutoff_freq = filter_cutoff_value


def validateAndSetFilterOrder(order_value):
    filter_order_value:int
    try:
        filter_order_value = int(order_value)
    except:
        raise Exception('filter order is not a valid integer number')

    if filter_order_value <= 1:
        raise Exception('filter order cannot be a negative number or lower then 1')
    
    global filter_order
    filter_order = filter_order_value


def validateAndSetSamplingFrequency(sampling_freq_value):
    filter_sampling_freq_value:float
    try:
        filter_sampling_freq_value = float(sampling_freq_value)
    except:
        raise Exception('filter sampling frequency is not a valid number')

    if filter_sampling_freq_value <= 0:
        raise Exception('filter sampling frequency cannot be a negative number or lower or euqal to 0')

    global sampling_frequency
    sampling_frequency = filter_sampling_freq_value


def validateAndSetFilterType(type_value):
    supported_types = ['lowpass']
    if not type_value in supported_types:
        raise Exception('filter type not supported')

    global filter_type
    filter_type = type_value


def validateAndSetDatasetsRoot(datasets_root_value):
    
    if not path.exists(datasets_root_value):
        raise Exception('dataset root does not exists')
    if not path.isdir(datasets_root_value):
        raise Exception('dataset root is not a valid path')
    
    global datasets_root
    datasets_root = datasets_root_value
    

def validateAndSetDatasetsExtension(datasets_extension_value):
    
    if not datasets_extension_value.startswith('.') or len(datasets_extension_value.split('.')) != 2 or datasets_extension_value.find('/') != -1 or datasets_extension_value.find('\\') != -1:
        raise Exception('extension {} not valid'.format(datasets_extension_value))

    supported_extensions = ['.csv']
    if datasets_extension_value not in supported_extensions:
        raise Exception('extension {} not supported'.format(datasets_extension_value))
    
    global datasets_extension
    datasets_extension = datasets_extension_value


# TODO: Export
# function needed for renaming columns of the dataset  (better putting on filter_config)
def rename_columns(dataset: pd.DataFrame):

    # punti di giunto
    joints = ['naso', 'torace', 'spalla_dx', 
    'gomito_dx',    'polso_dx',     'spalla_sx', 
    'gomito_sx',    'polso_sx',     'cresta_iliaca', 
    'anca_dx',      'ginocchio_dx', 'caviglia_dx', 
    'anca_sx',      'ginocchio_sx', 'caviglia_sx', 
    'occhio_dx',    'occhio_sx',    'zigomo_dx', 'zigomo_sx', 
    'piede_sx_1',   'piede_sx_2',   'piede_sx_3', 
    'piede_dx_1',   'piede_dx_2',   'piede_dx_3']

    # create the new names for the columns
    new_columns = []
    for idx, joint in enumerate(joints):
        new_columns.insert(idx*3 + 0, 'x_'+joint)
        new_columns.insert(idx*3 + 1, 'y_'+joint)
        new_columns.insert(idx*3 + 2, 'l_'+joint)

    # chek that the number of columns of the dataset match the new number of columns
    if dataset.shape[1] != len(new_columns):
        raise Exception('Il numero delle colonne del dataset di origine non è corretto')
    
    # set the new columns for the dataset
    dataset.columns = new_columns
    return dataset


#TODO: Export
def delete_nan(series):
    new_series = []
    for idx, value in enumerate(series):
        if not math.isnan(value):
            new_series.append(value)
    return new_series


#TODO: export
def filter_dataframe(dataframe: pd.DataFrame, filter_n_order, filter_cutoff_freq):
    new_dataframe = pd.DataFrame()
    for idx, column_name in enumerate(dataframe):
        series = delete_nan(dataframe[column_name])     # get the series (the column)

        b, a = signal.butter(filter_n_order, filter_cutoff_freq, fs=sampling_frequency)   # create the filter 
        filtered_series = signal.filtfilt(b, a, series)     # applay the filter
        
        new_dataframe = pd.concat([new_dataframe, pd.DataFrame({column_name: filtered_series})], axis=1)    #create the filtered dataframe

    return new_dataframe


def filterSingleDataset(root, filename):
    time0 = time.time()     # starting time

    original_dataset = pd.read_csv('{}/{}.csv'.format(root, filename))                                                # reading the dataset
    renamed_dataset = rename_columns(original_dataset)                                          # renaming the column of the dataset
    filtered_dataset = filter_dataframe(renamed_dataset, filter_order, filter_cutoff_freq)      # filter the entire dataset
    filtered_dataset.to_csv('{}/{}-filtered.csv'.format(root, filename) )

    deltaTime = time.time() - time0     # time taken for the processing
    print('\t\t[ processed ] {}\t| time = {} ms'.format(filename, round(deltaTime * 1000, 1))  )  # report 
    

def filterDatasets(): 
    # Logs
    time0 = time.time()
    print('--> starting filtering <--')

    for (root, dirs, files) in walk(datasets_root, topdown=True):

        for file in files:
            # check if the file has the right extension 
            if not file.endswith(datasets_extension):
                continue
            
            # remove the extension from the end getting the filename without it
            filename = file[:-len(datasets_extension)]
                
            # check if doesnt match the dataset name pattern go to next file
            if datasets_name_patter != None and re.search(r"{}".format(datasets_name_patter), filename) == None:
                continue
                
            # if the dataset is the filtered one go to next file
            if filename.endswith('-filtered'):
                continue
            

            # filter the dataset
            filterSingleDataset(root, filename)
            

    # Logs
    deltaTime = time.time() - time0
    print('[ -- ] total time = {:.5} seconds [ -- ]'.format(deltaTime))
    print('-->   end filtering    <--')
  




if __name__ == '__main__':
    # Open the file and read the contents
    config_path = 'filter_config.xml'
    print(os.getcwd())
    with open(config_path, 'r', encoding='utf-8') as file:
        filter_config_xml = file.read()
    parseXML(filter_config_xml)
    filterDatasets()