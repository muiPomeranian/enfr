import os


def get_local_path(path=None):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
