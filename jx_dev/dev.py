'''
dev masterscript 
'''

#%% import packages

import pandas                as     pd
from   pickle                import dump
import matplotlib.pyplot     as     plt
import os 
import jsonlines
from   utils_text_clf        import utils_text_clf as utils

# Turn interactive plotting off
plt.ioff()
import warnings
warnings.filterwarnings("ignore")

#%% Enter mutable info

data_dir = os.getcwd() + '\data'
filename = 'train.jsonl';
file     = os.path.join(data_dir, filename) # .json file

#%% parse the .json file 

df       = utils.parse_json(file)

#%% 
