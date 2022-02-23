import logging
import os
import shutil

out_dir = "log_test"
if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)


def init_logger():
    logging.basicConfig(handlers=[logging.NullHandler()], level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def do_logging(logger, i, j):
    output_folder_path = os.path.join(out_dir, f"{i}_{j}")
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)
    logger_handler = logging.FileHandler(filename=os.path.join(output_folder_path, "console.log"))
    formatter = logging.Formatter('[%(asctime)s] %(funcName)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)
    logger.info("New logger has been set up")
    logger.warning("Be aware of correctly using it!")
    return logger_handler


def dummy_process():
    logger = init_logger()
    for i in range(3):
        for j in range(3):
            logger_handler = do_logging(logger, i, j)
            logger.info(f"Message from the inner loop ({j})")
            logger.handlers = [logging.NullHandler()]
            a = 1


if __name__ == '__main__':
    dummy_process()