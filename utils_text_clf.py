"""
utils_text_clf.py

Utilities for text classification 

Author: Joe Xiao 
"""

import os 
import numpy             as     np
import pandas            as     pd 
import jsonlines

#%% init

class utils_text_clf:

    
    def __init__(self):
        '''
        Purpose: 
            init the class obj
        Input:
        Output:
            class obj
        '''

#%%
# =============================================================================
#   ____            _        __  __      _   _               _     
#  | __ )  __ _ ___(_) ___  |  \/  | ___| |_| |__   ___   __| |___ 
#  |  _ \ / _` / __| |/ __| | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
#  | |_) | (_| \__ \ | (__  | |  | |  __/ |_| | | | (_) | (_| \__ \
#  |____/ \__,_|___/_|\___| |_|  |_|\___|\__|_| |_|\___/ \__,_|___/
#                                                   
# =============================================================================
        
#%% create dir 
    
    @staticmethod
    def create_dir(dir_path):
        '''
        Purpose: 
            create a directory if it does not already exist 
        Input:
            dir_path - (str) full path to the dir
        Output:
            directory at the specified location    
        '''
        
        if not os.path.isdir(dir_path): # if directory doesn't exist
            os.mkdir(dir_path)          # create it
            

#%% parse json data into df

    @staticmethod
    def parse_json(file):
        '''
        Purpose: 
            parse json data into dataframe
        Input:
            file - (str)          full path to the jsonl file 
        Output:
            df   - (pd.DataFrame) each row is a line in the jsonl file
        '''
        
        # create df for data 
        df             = pd.DataFrame()
        
        with jsonlines.open(file) as reader:
            for obj in reader:
                
                # extract the content 
                df_obj = pd.DataFrame([obj], columns = list(obj.keys()))
                
                # append
                df     = df.append(df_obj)
                
        df.reset_index(inplace = True, drop = True)
        
        return df 