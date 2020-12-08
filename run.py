#!/usr/bin/env python
print("The code needs to be run on python3.6+. If you meet any ImportError, please try python3 run.py test")

import sys
import subprocess
import os
from src import etl
import json
sys.path.insert(0, 'src')
from src import eda
import test

def main(targets):
    eda_config = json.load(open('config/eda-params.json'))
    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        mydir = os.getcwd() # would be the MAIN folder
        mydir_tmp = mydir + "//src" # add the testA folder name
        mydir_new = os.chdir(mydir_tmp) # change the current working directory
        mydir = os.getcwd() #
        
        output=etl.run_cmd_shell(data_cfg['command'])
        return output   
    if 'test' in targets:
        
        #eda.generate_stats('test', **eda_config)
        
        test.dotest()        
        
        
if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets=sys.argv[1:]
    main(targets)
    
