import numpy as np
import pandas as pd
import scipy.stats as sp
import matplotlib.pyplot as plt
import os
import pickle

from funcs_preprocess_tobii import pupil


"""
created on: april 29 2022
last edited: may 16 2022

author: author: @zoeheez

description:
This  script preprocesses the pupil data

The script will create a nested dict with keys ['sub1'], ['sub2'], ...
The value corresponding to each key is also a dict with keys ['sess1'], ['sess2'], ...
Each item contains the behavioral info.

"""


proj_dir = os.path.normpath(os.getcwd()) #+ os.sep + os.pardir)
pupil_data_folder = 'pupil_data'
#session_type = 'behavioral'


subnums = np.arange(0,60)+1 #INPUT
#subnums = np.array([1])
nsub = len(subnums)
nblock = 8

#preprocessed_pupil_data_holder = [[] for _ in range(nsub)] # num of sub x 1
# save as a dictionary: 'sub1', 'session1': pupil() class
preprocessed_pupil_data_holder = {}

# go through each subject
for isub in range(nsub):
    print(f'preprocessing for subject {isub+1}...')
    subnum = subnums[isub]
    block_count = 0

    preprocessed_pupil_data_holder[f'sub{subnum}'] = {}

    for iblock in range(nblock):
        blocknum = iblock + 1
        # load session data
        subject_path = os.path.join(
            proj_dir, pupil_data_folder, f'sub-{subnum:02d}')

        pupil_data_path = os.path.join(subject_path, 'tobii_data',
                                    f'CBandit_TobiiData_subject_{subnum}_'
                                    + f'session_{blocknum}.txt')
        
        # preprocess pupil data using the preprocessing tools
        pupil_data = pupil()
        pupil_data.import_data_tobii(filename=pupil_data_path)
        pupil_data.preprocess_tobii(lpf=[20])

        preprocessed_pupil_data_holder[f'sub{subnum}'][f'sess{blocknum}'] = pupil_data


# save and export the preprocessed pupil data
# create a binary pickle file 
f = open("pupil_data_preprocessed_all.pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(preprocessed_pupil_data_holder,f)

# close file
f.close()
print('done preprocessing.')


