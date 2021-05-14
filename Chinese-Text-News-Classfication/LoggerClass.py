import logging
import logging.handlers
import datetime



class Logging():
    @classmethod
    def __init__(cls, name, save_path='', logerlevel='DEBUG'):
       
        cls.logger = logging.getLogger(name)
        if hasattr(logging, logerlevel):
            print(getattr(logging, logerlevel))
            cls.logger.setLevel(getattr(logging, logerlevel))
            print(f'**************{logerlevel}*****************')
        else:
            cls.logger.setLevel(logging.DEBUG)
            print(f'***************DEBUG******************')
        all_path = save_path + name + '-all.log'
        rf_handler = logging.handlers.TimedRotatingFileHandler(all_path, when='midnight', interval=1,
                                                               backupCount=7, atTime=datetime.time(0, 0, 0, 0))
        rf_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"))

        error_path = save_path + name + '-error.log'
        f_handler = logging.FileHandler(error_path)
        f_handler.setLevel(logging.ERROR)
        f_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

        cls.logger.addHandler(rf_handler)
        cls.logger.addHandler(f_handler)

        print('*****************Start recording********************')

    @classmethod
    def add_log(cls, info, level='debug'):
       
        if hasattr(cls.logger, level):
            log = getattr(cls.logger, level)
            log(info)
        else:
            cls.debug(info)
            cls.error(f'choose wrong dairy{level}')


if __name__ == "__main__":
    Logging.__init__()
    Logging.add_log('test data', 'debug')
