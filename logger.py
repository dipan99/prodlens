from dotenv import load_dotenv
import traceback
import logging
import os

load_dotenv()

class Logging:
    LOGGER_NAME = "ProdLens"

    @staticmethod
    def setLevel():
        logFormat = "%(asctime)-15s %(levelname)s:%(message)s"
        logging.basicConfig(format=logFormat)
        logger = logging.getLogger(Logging.LOGGER_NAME)
        logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
        return True

    @staticmethod
    def logDebug(text: str):
        logger = logging.getLogger(Logging.LOGGER_NAME)
        logger.debug("%s", text)
        return True
    
    @staticmethod
    def logInfo(text: str):
        logger = logging.getLogger(Logging.LOGGER_NAME)
        logger.info("%s", text)
        return True
    
    @staticmethod
    def logError(text: str):
        logger = logging.getLogger(Logging.LOGGER_NAME)
        traceback.print_exc()
        logger.error("%s", text)
        return True
    
    @staticmethod
    def logWarning(text: str):
        logger = logging.getLogger(Logging.LOGGER_NAME)
        logger.warning("%s", text)
        return True