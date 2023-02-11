import datetime

def __colored(color, text):
    return "\033[38;2;{};{};{}m{}\033[38;2;255;255;255m".format(color[0], color[1], color[2], text)

# colors
__YELLOW = (255, 255, 0)
__RED = (255, 0, 0)
__LIGHT_BLUE = (173, 216, 230)

# log types
__INFO = 'INFO'
__WARNING = 'WARNING'
__ERROR = 'ERROR'


def __get_time_string():
    c_time = datetime.datetime.now()
    return '{}/{}/{} | {}:{}:{}'.format(
        c_time.day,
        c_time.month,
        c_time.year,
        c_time.hour,
        c_time.minute,
        c_time.second
    ) 

def __get_log(log_type, log_color):
    return '[{}][{}]'.format(
        __colored(log_color, log_type),
        __get_time_string()
    )

def info(message:str):
    print('{} {}'.format(
        __get_log(__INFO, __LIGHT_BLUE),
        message
    ))

def warning(message:str):
    print('{} {}'.format(
        __get_log(__WARNING, __YELLOW),
        message
    ))

def error(message:str):
    print('{} {}'.format(
        __get_log(__ERROR, __RED),
        message
    ))
