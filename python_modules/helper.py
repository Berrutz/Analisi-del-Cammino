import numpy as np
import pandas as pd
import math as math


def intspace_(stop):
    """ Return an array of integer elemetns from 0 to stop not included [0,stop) 
    """

    if not isinstance(stop, int):
        raise Exception('top value must be an integer')

    if stop <= 0:
        raise Exception('stop value must be grater then 0')
        
    space = []
    for i in range(stop):
        space.append(i)

    return np.array(space)


def intspace(start, stop):
    """ Return an array of integer elemetns from start to stop not included [start, stop) 
    """
    
    if not isinstance(start, int) or not isinstance(stop, int):
        raise Exception('strat and stop value must be integer values')

    if start >= stop:
        raise Exception('strat ({}) must be lower then the stop ({}) value'.format(start, stop))

    space = []
    for i in range(start, stop):
        space.append(i)

    return np.array(space)



def rename_columns(dataset: pd.DataFrame):
    """ Rinomina i nomi delle colonne del dataframe 
    
    Project specific
    ----------------
        i dati sono suddivisi a gruppi di 3 nel seso che se suddividessimo il dataset in gruppi di 3
        la prima colonna sarebbe relativa alla posizione sell'asse x di quel giunto, la seconda
        colonna sarebbe relativa alla posizione sull'asse y di quel giunto mentre l'ultima
        colonna sarebbe relativa alla likelihood    
    """
    
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
    
    
    
def delete_nan(series):
    """ emlimina i valori nulla in una lista
    """
    new_series = []
    for idx, value in enumerate(series):
        if not math.isnan(value):
            new_series.append(value)
    return new_series



def delete_nan_from_DataFrame(dataframe: pd.DataFrame):
    """ Elimina i valori nulli da un pandas dataframe
    """
    
    new_dataframe = pd.DataFrame()

    for idx, column_name in enumerate(dataframe.keys()):
        new_dataframe = pd.concat([new_dataframe, pd.DataFrame({column_name: delete_nan(dataframe[column_name])})], axis=1)

    return new_dataframe


def convert_to_list(series):
    """ prova a chiamare la funzione .tolist() altrimenti ritorna nullo
    """
    
    if isinstance(series, list):
        return series
    try:
        return series.tolist()
    except:
        return None   