import os

DATA_PATH = None
PROJECT_PATH = os.path.dirname(__file__)

try:
    from user_settings import *  # silence pyflakes
except ImportError:
    pass
