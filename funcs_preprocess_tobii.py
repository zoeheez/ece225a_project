#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define a class to import, preprocess and epoch the data saved by eyelink.

@author: Maëva L'Hôtellier, from Florent Meyniel's Eyelink version
"""
import re
import pandas as pd
import numpy as np
import os
import copy
from scipy import signal
import matplotlib.pyplot as plt

regexp_tobii = '(\d+) \t \((-?)\d*.\d*,\s(-?)\d*.\d*\) \t \((-?)\d*.\d*,' + \
               '\s(-?)\d*.\d*\) \t (\d*.\d*) \t (\d*.\d*)\t \n'


class pupil:
    def __init__(self, filename=None, directory=None):
        self.filename = filename
        self.directory = directory

    def import_data_tobii(self, filename=None, directory=None, eye='left',
                          col=[0, 1, 2],
                          col_name=['x_pos', 'y_pos', 'diameter'],
                          regexp=regexp_tobii):

        """ Import the data.
            - filename: name of file in the directory (supersedes the one in initialization)
            - directory: full path to the directory containing the file (supersedes the one in 
              initialization)
            - col: column to extract from the file (after the time stamp)
            - names of the corresponding columns
            - regexp: regular expression to parse the columns of the file. 1st element is the
              time_stamp, following elements are data points (1st data point correspond to col[0]
              and col_name[0])
        It returns:
            - the events collected by eyelink
            - the blinks (onsets and offsets) detected by eyelink
            - the time_stamp (ms) of the data (each data point)
            - time_stamp (s) relative to first data point
            - the data (as a panda dataframe)
        The datafile should be an ascii file (best) or an edf file. In the latter case, the script
        will attempt to convert it to ascii (on linux).
        """

        # Locate file
        if filename is None:
            if self.filename is None:
                raise ValueError('no filename provided: cannot import data')
        else:
            self.filename = filename
        if directory is None:
            # will look for the file in the current folder
            directory = os.getcwd()
        filename = os.path.join(directory, filename)
        if not os.path.isfile(filename):
            raise ValueError('Could not find ', filename)

        # Read file and get values
        events, time_stamp = [], []
        blinks_counter = [0]
        blink_start, blink_end = [], []
        data = [[], [], []]
        with open(filename, "r", encoding="ISO-8859-1") as file:
            lines = file.readlines()
            for line in lines:
                if 'MSG' in line:
                    # get the messag sent to the eyetracker
                    event_time = line.split(' ')[0].strip()

                    if 'response' in line:
                        info = line.split(' ')
                        if info[7] == '1\n':
                            type_trial = 'forced'
                        else:
                            type_trial = 'free'
                        event_msg = 'response' + ' ' + info[3] + ' ' + 'rt' + \
                            ' ' + info[5] + ' ' + 'type trial' + ' ' + \
                            type_trial

                    elif 'Outcome received:' in line:
                        info = line.split(' ')
                        event_msg = 'outcome'+' '+info[-1].replace('\n', '')

                    elif 'Trial' in line:
                        info = line.split(' ')
                        event_msg = 'trial'+' '+info[-1].replace('\n', '')

                    elif 'estim' in line:
                        info = line.split(' ')
                        event_msg = info[-1].replace('\n', '')

                    else:
                        info = line.split(':')[1].strip()
                        event_msg = " ".join(info.split(' ')[1:])

                    events.append({'time': event_time, 'msg': event_msg})

                if re.match(regexp, str(line)) is not None:
                    # get the values saved by the eyetracker
                    info = line.rstrip('\t...\n').split('\t')
                    time_stamp.append(int(info.pop(0)))
                    if eye == 'left':
                        del info[0], info[1]
                    if eye == 'right':
                        del info[1], info[2]

                    #  Clean info
                    info = info[0].split(',') + info[1: -1]
                    to_remove = "()"
                    pattern = '[' + to_remove + ']'

                    for k, val in enumerate(info):
                        val = re.sub(pattern, "", val)
                        try:
                            data[k].append(float(val))
                        except:
                            data[k].append('')

                    if blinks_counter[-1] == 1:
                        blink_end.append(time_stamp[-1])
                    blinks_counter.append(0)

                elif "nan" in line:
                    info = line.rstrip('\t...\n').split('\t')

                    time_stamp.append(int(info[0]))
                    if blinks_counter[-1] == 0:
                        blink_start.append(int(info.pop(0)))
                    blinks_counter.append(1)

                    if eye == 'left':
                        del info[0], info[1]
                    if eye == 'right':
                        del info[1], info[2]

                    info = [0, 0, 0]
                    for k, val in enumerate(info):
                        data[k].append(float(0))

        #  Consider the last blink as ended if this is the end of the file
        if blinks_counter[-1] == 1:
            blink_end.append(time_stamp[-1])

        #  Store info
        self.events = events
        self.blink_start = np.array(blink_start)
        self.blink_end = np.array(blink_end)

        # Convert into a panda dataframe
        self.data = pd.DataFrame({'time_stamp': time_stamp,
                                  'time_stamp_rel_s': (np.array(time_stamp)
                                                       - time_stamp[
                                                            0])/1000000})
        col = [col] if type(col) is not list else col

        col_name = [col_name] if type(col_name) is not list else col_name

        for k, name in zip(col, col_name):
            self.data[name] = data[k]

        # get sampling rate (the time_stamp is in microseconds)
        self.recording_fs = 1000000/(np.round(np.mean(np.diff(time_stamp))))

    def preprocess_tobii(self, margin=0.1, lpf=[30], plot=False):
        """
        Preprocess the data
            - margin: added before and after each blink. Data during blinks is linearly interpolated
            - lpf: cutoff (Hz) to low pass filter the data with. If [], do nothing
            - plot: plot the result if True
        The function gets the blinks (NB: currently from eyelink, but a custom function could be
        used) instead and linearly interpolate the data.
        For low pass filtering, a bi-directional 4th order Butterworth filter is used.
        The function replace previously pre-processed data, if any.
        It returns an indicator column in the panda dataframe indicating blink periods.
        """
        # INTERPOLATE BETWEEN BLINKS
        # get columns to process
        col_to_change = [
            col for col in self.data.columns if col != 'track_int']

        for col in col_to_change:
            self.data[col+'_int'] = copy.deepcopy(self.data[col])
        self.data['track_int'] = np.zeros(self.data.shape[0])
        time_stamp = np.array(self.data['time_stamp'])

        # merge blinks that are too close to one another
        is_long = (self.blink_start[1:] - self.blink_end[
            0:-1]) > margin * 2 * 1000000
        self.blink_start = self.blink_start[np.hstack((True, is_long))]

        self.blink_end = self.blink_end[np.hstack((is_long, True))]

        # interpolate the signal between blink
        first = self.data['time_stamp'].iloc[0]
        last = self.data['time_stamp'].iloc[-1]

        for start, end in zip(self.blink_start, self.blink_end):
            t = (time_stamp >= (start - margin * 1000000)) & \
                (time_stamp <= (end + margin * 1000000))
            self.data.loc[t, 'track_int'] = 1
            for col in col_to_change:
                val_start = self.data[col + '_int'][
                  np.where(t)[0][0]] if start > (first+1)\
                  else self.data[col+'_int'][np.where(t)[0][-1]]
                val_end = self.data[col+'_int'][
                    np.where(t)[0][-1]] if end < last else val_start
                self.data.loc[t, col+'_int'] = np.linspace(
                    val_start, val_end, np.sum(t)+1)[1:]

        # LOW PASS FILTER
        if lpf != []:
            w = lpf / (self.recording_fs / 2)  # Normalize the frequency
            b, a = signal.butter(4, w, 'low')  # 4-th order butterworth filter
            for col in col_to_change:
                # use a bidirectional (0-lag) filter
                self.data[col + '_int'] = signal.filtfilt(b, a, self.data[
                    col + '_int'])

        #  PLOT TO CHECK QUALITY
        if plot:
            plt.figure()
            plt.title('diameter')
            plt.plot(self.data['time_stamp_rel_s'], self.data['diameter'], '-',
                     self.data[
                         'time_stamp_rel_s'], self.data['diameter_int'], '-')
            plt.show()
            plt.figure()
            plt.title('x_pos')
            plt.plot(self.data['time_stamp_rel_s'], self.data['x_pos'], '-',
                     self.data[
                         'time_stamp_rel_s'], self.data['x_pos_int'], '-')
            plt.show()
            plt.figure()
            plt.title('y_pos')
            plt.plot(self.data['time_stamp_rel_s'], self.data['y_pos'], '-',
                     self.data[
                         'time_stamp_rel_s'], self.data['y_pos_int'], '-')
            plt.show()

    def epoch_tobii(self, onsets, before=1, after=1, variable='diameter_int',
                    decimate=1, reject=False, blink_ratio_thd=0.2,
                    conditions={}, baseline=[]):
        """
        Epoch the data
            - onset: list of points around which to epoch. Onsets are according to the
                data['time_stamp']
            - before / after: duration (in s) before and after the onset
            - variable: which element of data.data should be epoched
            - decimate: downsampling factor (int). If 1, there is no downsampling
            - reject: remove epoch with an excessive blink ratio
            - blink_ratio: threshold above which the epoch is removed
            - conditions: dictionary which define properties of the onset (it will be copied, after
                rejecting the relevant epochs, in the output)
            - baseline: [start, end] in s relative to the onset, use for subtractive basline
                correction. If [], no correction is made
        It returns:
            epochs: the curated, baseline-corrected epochs (as a numpy array, epoch x time)
            epochs_times: the peri-stimulus time vector (in s).
            epochs_info: a panda dataframe with the epoch number, the blink ratio, and conditions (
                rejected epoch do not appear there)
        """

        epochs = []
        blink_ratio = []

        index_onsets = self.data['time_stamp'].tolist()

        for onset in onsets:

            index = index_onsets.index(onset)

            indices = (self.data['time_stamp'] >= self.data['time_stamp'][
                index-int(before*60)]) & (self.data['time_stamp'] <= self.data[
                    'time_stamp'][index+int(after*60)])

            chunk = self.data[variable].loc[indices]

            if onset == onsets[0]:
                epochs.append(chunk.values)

            elif len(epochs[0]) < len(chunk.values):
                value = chunk.values[0:len(epochs[0])]
                epochs.append(value)
            elif len(epochs[0]) > len(chunk.values):
                value = np.concatenate([chunk.values,
                                        np.zeros(len(epochs[0]) - len(
                                            chunk.values))])
                epochs.append(value)
            else:
                epochs.append(chunk.values)

            blink_ratio.append(np.mean(self.data['track_int'].loc[indices]))

        epoch_blink_ratio = np.array(blink_ratio)
        epochs = np.vstack(epochs)[:, 0::decimate]

        # get peri onset times
        epoch_times = (self.data['time_stamp'].loc[indices].values[
            0::decimate] - onset)/1000000

        # baseline correct
        baseline_val = []
        if baseline != []:
            baseline_ind = (epoch_times >= baseline[0]) &\
                (epoch_times <= baseline[1])
            baseline_val = np.mean(epochs[:, baseline_ind], axis=1)
            epochs = epochs - baseline_val[:, np.newaxis]
        else:
            baseline_val = np.zeros(epochs.shape[0])

        # reject epochs with too much blinks
        epoch_number = np.arange(0, epochs.shape[0])
        if reject is True:
            to_keep = epoch_blink_ratio <= blink_ratio_thd
            epochs = epochs[to_keep, :]
            epoch_number = epoch_number[to_keep]
            epoch_blink_ratio = epoch_blink_ratio[to_keep]
            baseline_val = baseline_val[to_keep]

            for info in conditions.keys():
                conditions[info] = np.array(conditions[info])[to_keep]

        # assemble the data
        self.epochs = epochs
        self.epochs_times = epoch_times
        self.conditions = conditions
        self.epochs_info = pd.DataFrame(data=dict(
            {'epoch_number': epoch_number,
             'blink_ratio': epoch_blink_ratio,
             'baseline': baseline_val}, ** conditions))
