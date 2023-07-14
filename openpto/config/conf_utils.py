import ruamel.yaml as yaml
import argparse
import os


def load_conf(path:str = None, method_name:str = None, prob_name:str = None):
    '''
    Function to load config file.

    Parameters
    ----------
    path : str
        Path to load config file. Load default configuration if set to `None`.
    method : str
        Name of the used mathod. Necessary if ``path`` is set to `None`.
    dataset : str
        Name of the corresponding dataset. Necessary if ``path`` is set to `None`.

    Returns
    -------
    conf : argparse.Namespace
        The config file converted to Namespace.

    '''
    if path == None and method_name == None:
        raise KeyError
    if path == None and prob_name == None:
        raise KeyError
    if path == None:
        method_names = ['spo']
        prob_names = []

        # assert method in method_name
        assert prob_name in prob_names
        dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
        path = os.path.join(dir,  prob_name + ".yaml")

        if os.path.exists(path) == False:
            raise KeyError("The configuration file is not provided.")
    
    conf = open(path, "r").read()
    conf = yaml.safe_load(conf)
    conf = argparse.Namespace(**conf)

    return conf


def save_conf(path, conf):
    '''
    Function to save the config file.

    Parameters
    ----------
    path : str
        Path to save config file.
    conf : dict
        The config dict.

    '''
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(vars(conf), f)
