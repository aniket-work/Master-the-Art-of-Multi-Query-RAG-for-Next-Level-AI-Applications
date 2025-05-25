import logging
import colorlog
import yaml

def setup_logging(config):
    level = getattr(logging, config.get('logging', {}).get('level', 'INFO'))
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(asctime)s [%(levelname)s] %(message)s'))
    logging.root.handlers = [handler]
    logging.root.setLevel(level)

