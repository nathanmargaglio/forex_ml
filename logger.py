import logging
import sys

loggers = {}
def setup_custom_logger(name, instance_name=''):
    global loggers
    global instance_dir

    if loggers.get(name):
        return loggers.get(name)

    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-2s [%(name)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('data/instances/' + instance_name + '/log.txt', mode='a')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    loggers[name] = logger
    return logger
