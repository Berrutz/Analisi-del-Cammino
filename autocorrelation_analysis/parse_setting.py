import xmltodict
import log as log
import os
import re


def __getOrRaise(key, dictionary, errorMessage=None):
    value = dictionary.get(key)
    if value == None:
        log.error('{} tag not found'.format(key) if errorMessage == None else errorMessage)
        raise Exception('{} tag not found'.format(key) if errorMessage == None else errorMessage)
    return value


def parseSettings_getDatasets(settings_path: str, settings_filename = 'settings.xml') -> list:

    settings_file = os.path.join(settings_path, settings_filename) 
    log.info('settings file: {}'.format(settings_file))


    # read the settings file
    log.info('trying opening {}'.format(settings_file))
    try:
        with open(settings_file, 'r', encoding='utf-8') as file:
            settings_file = file.read()
    except Exception as e:
        log.error('error while trying to opening settings file:\n{}'.format(e))
        raise
    
    log.info('setting file opened ...')

    settings = __get_settings(settings_file)

    # get all the dataset path
    datasets_path_list = __get_all_datasets(
        settings['datasets-path'],
        settings['datasets-extension'],
        settings.get('datasets-pattern', None)
        )

    log.info('found {} datasets'.format(len(datasets_path_list)))
    return datasets_path_list


  
    


def __get_settings(settings_file):
    settings_dict = xmltodict.parse(settings_file)
    settings_dict = __getOrRaise('Settings', settings_dict)
    
    return_settings: dict = dict()

    # datasets extension
    return_settings['datasets-extension'] = __parse_datasets_extension(settings_dict)
    
    # set datasets path
    return_settings['datasets-path'] = os.getcwd()


    # getting other settings
    for key, value in settings_dict.items():

        # not default datasets path
        if  key == 'datasets-path':
            return_settings[key] = os.path.join(os.getcwd(), value)
            if not os.path.isdir(return_settings[key]):
                log.error('datasets path folder does not exists:\n{}'.format(return_settings[key]))
                raise Exception ()

            log.info('datasets path set to: {}'.format(return_settings[key]))

        elif key == 'datasets-pattern':
            return_settings[key] = value
            log.info('datasets pattern set to: {}'.format(value))
        
        else:
            log.warning('unexpected key: {}'.format(key))

    return return_settings


def __parse_datasets_extension(settings_dict):
    log.info('getting datasets extension')
    extension = __getOrRaise('datasets-extension', settings_dict)
    
    del settings_dict['datasets-extension']

    if not isinstance(extension, str):
        log.error('datasets extension must be a string')
        raise Exception()

    if not extension.startswith('.'):
        log.error('datasets extension incorrect')
        raise Exception()

    log.info('datasets extension set to: {}'.format(extension))
    return extension




def __get_all_datasets(datasets_path, datasets_extension, datasets_patter = None):
    log.info('getting all the datasets: \
        \n\tfolder: {}   \
        \n\textension: {}'
        .format(datasets_path, datasets_extension))

    datasets_path_list: list = []

    for root, dirs, files in os.walk(datasets_path):
        
        for file in files:
            
            # check if the file has the right extension 
            if not file.endswith(datasets_extension):
                continue
            
            # getting the filename
            filename = file[:-len(datasets_extension)]

            # check if doesnt match the dataset name pattern go to next file
            if datasets_patter != None and re.search(r"{}".format(datasets_patter), filename) == None:
                continue
                
            dataset_path = os.path.join(root, file)
            datasets_path_list.append(dataset_path)

            log.info('datasets added: {}'.format(dataset_path))


    return datasets_path_list
