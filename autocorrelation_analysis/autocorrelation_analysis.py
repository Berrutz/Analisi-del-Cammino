# import 
import numpy as np
import pandas as pd
import os
import pprint
import shutil
import statsmodels.tsa.stattools as sts
import math as math
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

import log as log
from functionality import * 
from parse_setting import *

CURRENT_FILE_DIR = os.path.dirname(__file__)


def run(settings_path):
    # getting all the filtered datasets
    datasets = parseSettings_getDatasets(os.path.join(CURRENT_FILE_DIR, settings_path))
    if len(datasets) == 0:
        return
    
        

    # analyze each dataset
    patients: dict = dict()
    for dataset_path in datasets:
        
        dataset_file = dataset_path.split('/')[-1]
        dataset_name = dataset_file.split('.')[0]

        dataset_name_split = dataset_name.split('_')
        patient_name = dataset_name_split[0]
        wolking_type = dataset_name_split[1]
        dataset_number = dataset_name_split[2].split('-')[0]

        
        if not patient_name in patients:
            patient_dict = {
                'patient-name': patient_name,
                'datasets': [
                    {
                        'type': wolking_type,
                        'number': dataset_number,
                        'path': dataset_path,
                    }
                ]
            }
            patients[patient_name] = patient_dict
        else:
            patients[patient_name]['datasets'].append(
                {
                    'type': wolking_type,
                    'number': dataset_number,
                    'path': dataset_path,
                }
            )

    # pprint.pprint(patients)
    analyze_patients(patients)



def get_patient_folder(patient):
    patient_fodler = os.path.join(CURRENT_FILE_DIR, patient)

    if os.path.exists(patient_fodler) and os.path.isdir(patient_fodler):
        shutil.rmtree(patient_fodler)
    
    os.makedirs(patient_fodler)

    return patient_fodler

def get_folder_walking(walking_type, pateint_folder):
    dataset_w_t_folder = os.path.join(pateint_folder, walking_type)

    if not os.path.exists(dataset_w_t_folder):
        os.makedirs(dataset_w_t_folder)

    return dataset_w_t_folder

def get_folder_dataset_number(d_number, walking_folder):
    dataset_numb_folder = os.path.join(walking_folder, d_number)

    if not os.path.exists(dataset_numb_folder):
        os.makedirs(dataset_numb_folder)

    return dataset_numb_folder

def get_folder_plots(p_dataset_folder):
    plots_folder = os.path.join(p_dataset_folder, 'plots')

    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    return plots_folder


def analyze_patients(patients):
    
    for key, value in patients.items():
        p_name = value['patient-name']
        p_folder = get_patient_folder(p_name)

        datasets = value['datasets']
        
        log.info('analyzing patient {}'.format(p_name))
        for dataset in datasets:
            analyze_dataset(dataset, p_folder)    

        log.info('end \n\n')   
        # for each dataset 
            # create folder
            # do analysis


def analyze_dataset(dataset_dict, p_folder):
    walking = dataset_dict['type']
    d_number = dataset_dict['number']
    d_path = dataset_dict['path']

    walking_folder = get_folder_walking(walking, p_folder)
    dat_num_folder = get_folder_dataset_number(d_number, walking_folder)
    plots_folder = get_folder_plots(dat_num_folder)

    dataset =  read_dataset(d_path)

    log.info('--- analyzing dataset | type: {} number: {}'.format(walking, d_number))
    for joint in dataset.columns:

        if joint.startswith('l') or joint.find('piede') == -1:
            continue

        analyze_joint(dataset[joint].tolist(), joint, d_number, dat_num_folder, plots_folder)


def read_dataset(d_path):
    return pd.read_csv(d_path)


def analyze_joint(joint_series, joint_name, dat_num, dat_num_folder, plots_folder):
    log.info('       analizing joint: {}'.format(joint_name))
    
    # deleting nan
    joint_series = delete_nan(joint_series)

    # making ndarray
    if not isinstance(joint_series, np.ndarray):
        joint_series = np.array(joint_series)
    
    # making it stationary
    joint_series = make_stationary(joint_series)
    
    # acf
    acf = funzione_autocorrelazione(joint_series)
    
    # save the acf plot
    plot_and_save_acf(acf, plots_folder, dat_num, joint_name)

    # get the local maxima for the acf
    local_max = argrelextrema(acf, np.greater)[0]

    # local max diff
    local_max_diff = compute_first_diff(local_max)

    # open the report file and write
    report = open(os.path.join(dat_num_folder, 'report.txt'), 'a')
    report.write('\n-----------------------------------------------\n')
    report.write('{}:\n\n'.format(joint_name))
    report.write('acf locals max     : {}\n'.format(local_max))
    report.write('acf locals max diff: {}\n\n\n'.format(local_max_diff))


    report.write('locals max :\t- mean: {:.4}\t- std: {:.4}\n'.format(local_max.mean(), local_max.std()))
    report.write('locals max :\t- mean: {:.4} sec\t- std: {:.4} sec\n\n'.format(local_max.mean()/30, local_max.std()/30))

    report.write('locals max diff:\t- mean: {:.4}\t- std: {:.4}\n'.format(local_max_diff.mean(), local_max_diff.std()))
    report.write('locals max diff:\t- mean: {:.4} sec\t- std: {:.4} sec\n\n'.format(local_max_diff.mean()/30, local_max_diff.std()/30))
    
    report.close()


def plot_and_save_acf(acf, plot_folder, d_number, joint):

    plt.figure(figsize=(20,7))
    plt.title('acf of {}'.format(joint))
    plt.ylim([-1, 1])
    x = np.arange(len(acf))
    plt.vlines(x=x, ymin=0, ymax=acf, colors='red', ls='-', lw=1)
    plt.axhline(y = 0, color = 'black', linestyle = '-')
    plt.plot(acf, ".")

    plt.xticks(x if len(x) <= 30 else np.arange(len(x), step=len(x)//20))

    fig_name = 'd{}_{}_acf.png'.format(d_number, joint)
    figure_path = os.path.join(plot_folder, fig_name)
    plt.savefig(figure_path)
    

def delete_nan(series: pd.Series | list):
    """ emlimina i valori nulla da una serie
    """
    new_series = []
    for idx, value in enumerate(series):
        if not math.isnan(value):
            new_series.append(value)
    return new_series


def compute_first_diff(lista):
    if not isinstance(lista, np.ndarray):
        lista = np.array(lista)
    return lista[1:] - lista[:len(lista)-1]



def make_stationary(series, max_steps = 30):

    steps = 0
    s_back = series.copy()
    
    while(sts.adfuller(series)[1] > 0.05 and steps < max_steps):
        series = compute_first_diff(series)
        steps += 1

    if steps > 30:
        log.warning('\t\tcannot make the joint stationary')
        series = s_back

    elif steps > 0:
        log.warning('\t\tjoint series not stationary (diff: {})'.format(steps))

    return series


if __name__ == '__main__':
    run('./')
