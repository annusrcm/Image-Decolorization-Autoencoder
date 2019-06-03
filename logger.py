import datetime
import logging
import sys
from config import Config

config = Config()

class Logger:
    __logger = None

    @staticmethod
    def get_logger():
        if Logger.__logger is None:
            module_dir = sys.path[0] + "/logs"
            config.create_folder_if_not_present(module_dir)

            now = datetime.datetime.now()

            log_file_name = "log_" + now.strftime("%Y_%m_%d") + ".txt"

            path_log = module_dir + "/" + log_file_name

            logger = logging.getLogger('logger')
            logger.setLevel(logging.DEBUG)
            logging.basicConfig(format='%(message)s', level=logging.DEBUG)
            handler = logging.FileHandler(path_log)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logging.getLogger().addHandler(handler)
            Logger.__logger = logger

        return Logger.__logger

    @staticmethod
    def log(message):
        Logger.get_logger().info(message)

    @staticmethod
    def log_error(message):
        Logger.get_logger().info(message)