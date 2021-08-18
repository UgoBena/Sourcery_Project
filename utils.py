import os
import torch
import json
import sys
from datetime import datetime

from config import Config

def pretty_time(t):
    """
    Tranforms time t in seconds into a pretty string
    """
    return f"{int(t//60)}m{int(t%60)}s"

def get_root():
    """
    Gets the absolute path to the root of the project
    """
    return os.sep.join(os.getcwd().split(os.sep)[0 : os.getcwd().split(os.sep).index("Sourcery_Project") + 1])

def now():
    """
    Current date as a string
    """
    return datetime.now().strftime('%y-%m-%d_%Hh%Mm%Ss')

def save_json(path_result, name, x):
    """
    Saves x into path_result with the given name
    """
    with open(os.path.join(path_result, f'{name}.json'), 'w') as f:
        json.dump(x, f, indent=4)

bcolors = {
    'RESULTS': '\033[95m',
    'HEADER': '\033[94m',
    'SUCCESS': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m',
    'INFO': '\033[1m',
    'UNDERLINE': '\033[4m'
}

def printc(log, color='HEADER'):
    """
    Prints logs with color according to the dict bcolors
    """
    print(f"{bcolors[color]}{log}{bcolors['ENDC']}")

def print_args(args):
    """
    Prints argparse arguments from the command line
    """
    print(f"> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")

def create_session(args):
    """
    Initializes a script session (set seed, get the path to the result folder, ...)
    """
    torch.manual_seed(0)
    
    print_args(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    printc(f"> DEVICE:  {device}", "INFO")

    path_root = get_root()
    printc(f"> ROOT:    {path_root}", "INFO")

    path_dataset = os.path.join(path_root, args.data_folder)

    main_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    session_id = f"{main_file}_{now()}"
    path_result = os.path.join(path_root, "results", session_id)
    os.mkdir(path_result)
    printc(f"> SESSION: {path_result}", "INFO")

    args.path_result = path_result
    config = Config(args)
    save_json(path_result, 'config', vars(config))

    return path_dataset, device, config