"""
target dis
MEAN
LOG
get_dict
save_best

set global seed
save latent,
save recon_metric csv,
save loss csv,
save eval metric csv,
save recon image 
save best model

cluster
IDEC
"""


import logging
import os
import pickle
import random
import re
from collections import defaultdict
from os import listdir, makedirs
from os.path import exists, join
from shutil import rmtree
from time import sleep

import numpy as np
import pandas as pd
import torch
# import torchvision


def setup_logging(log_dir):
    makedirs(log_dir, exist_ok=True)

    logpath = join(log_dir, 'log.txt')
    filemode = 'a' if exists(logpath) else 'w'

    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=logpath,
                        filemode=filemode)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def prepare_results_dir(config):
    output_dir = join(config['results_root'], config['arch'],
                      config['experiment_name'])
    if config['clean_results_dir']:
        if exists(output_dir):
            print('Attention! Cleaning results directory in 10 seconds!')
            sleep(10)
        rmtree(output_dir, ignore_errors=True)
    makedirs(output_dir, exist_ok=True)
    makedirs(join(output_dir, 'weights'), exist_ok=True)
    makedirs(join(output_dir, 'samples'), exist_ok=True)
    makedirs(join(output_dir, 'results'), exist_ok=True)
    return output_dir


def find_latest_epoch(dirpath):
    # Files with weights are in format ddddd_{D,E,G}.pth
    epoch_regex = re.compile(r'^(?P<n_epoch>\d+)_[DEG]\.pth$')
    epochs_completed = []
    if exists(join(dirpath, 'weights')):
        dirpath = join(dirpath, 'weights')
    for f in listdir(dirpath):
        m = epoch_regex.match(f)
        if m:
            epochs_completed.append(int(m.group('n_epoch')))
    return max(epochs_completed) if epochs_completed else 0


def cuda_setup(cuda=False, gpu_idx=0):
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(gpu_idx)
    else:
        device = torch.device('cpu')
    return device


def set_seed_globally(seed_value=0,if_cuda=True, gpu_idx=0):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED']=str(seed_value)
    device = "cpu"
    if torch.cuda.is_available(): 
        torch.cuda.set_device(gpu_idx)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
        device = f"cuda:{gpu_idx}"
    
    return device





def target_distribution(q):
    # print(q.shape)
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def LOG(logger,str):
    logger.info(str)
    print(str)


def MEAN(x):
    return sum(x)/len(x)


def get_dict(glob, loc):
    if glob == None:
        glob = defaultdict(list)
        for key in loc.keys():
            glob[key] = [loc[key]]
    else:
        for key in loc.keys():
            glob[key].append(loc[key])

    return glob



def save_best(model, acc, results, best_acc, epoch, optimizer, name, z, Y):
    if best_acc == None:
        best_acc = acc
        dict = {'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'acc': acc}
        results.save_model(dict, name)
        results.save_latent("best", z, Y, name=name)

        print("saved", best_acc)

    elif acc >= best_acc:
        best_acc = acc
        dict = {'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'nmi': acc}

        results.save_model(dict, name)
        print(f"saved -> {name} -> {best_acc:.4f}")

    return best_acc

def set_seed_globally(seed_value=0,  gpu=0):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(device)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(gpu)
    return device


class save_results():
    def __init__(self, root):
        self.root = root

        self.csv_path = self.root+"/CSV/"
        self.model_path = self.root+"/Model/"

        """
        root = batch:{}_lr:{}_optim:{}_alpha:{}_beta:{}_gamma:{}_recentre:{}
        
        """

        if not os.path.exists(self.csv_path):
            os.makedirs(self.csv_path)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)


    def save_loss(self, loss):
        data = loss
        dataframe = pd.DataFrame(data=data)
        dataframe.to_csv(self.csv_path+"loss.csv")

    def save_eval_metric(self, metric, name):
        data = metric
        dataframe = pd.DataFrame(data=data)
        dataframe.to_csv(self.csv_path+f"{name}_eval_metric.csv")

    def save_model(self, dic,name):
        torch.save(dic, self.model_path+f"{name}_best.pth.tar")


class LOGGING():
    def __init__(self, root):
        super().__init__()
        self.log_file = os.path.join(root, "PointNet_.log")

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(filename=self.log_file,mode='a+')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info("-"*10+"X"+"-"*10)
        self.logger.info("experimentation")

    def LOG(self, str):
        self.logger.info(str)
        print(str)