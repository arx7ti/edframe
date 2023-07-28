from __future__ import annotations
from ast import Not
from codecs import ignore_errors

import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime

from collections.abc import Sequence
from typing import Optional, Callable, Union, List


class ExhaustiveArgumentError(Exception):

    def __init__(self, arg_name, *args: object) -> None:
        message = 'Argument "{}" is exhaustive'.format(arg_name)
        self.message = message
        return super().__init__(message, *args[1:])


class Reader(Sequence):
    pass


def default_label(label: str) -> str:
    """
    Format a label by default

    Arguments:
        label: str
    Returns:
        str
    """
    label = label.lower().replace(' ', '_')
    return label


class PLAID(Reader):

    def __init__(
        self,
        dataset_path: str,
        metadata: Optional[dict] = None,
        metadata_path: Optional[str] = None,
        dtype=None,
        label_fn: Optional[Callable] = None,
    ):
        self._dataset_path = dataset_path
        if metadata is not None:
            if metadata_path is not None:
                raise ExhaustiveArgumentError('metadata_path')
            self._metadata = metadata
        elif metadata_path is not None:
            if metadata is not None:
                raise ExhaustiveArgumentError('metadata')
            with open(metadata_path) as j:
                self._metadata = json.load(j)
        else:
            raise ValueError('Metadata is required')
        self._metadata = list(
            sorted(self._metadata.items(), key=lambda x: int(x[0])))
        self._dtype = np.float32 if dtype is None else dtype
        self._label_fn = label_fn
        return None

    def __len__(self):
        return len(self._metadata)

    def __getitem__(
        self,
        idxs: Union[slice, int],
    ) -> tuple[str, np.ndarray, np.ndarray, int]:
        if isinstance(idxs, slice):
            iterator = self._metadata[idxs]
        elif isinstance(idxs, list):
            iterator = [self._metadata[idx] for idx in idxs]
        elif isinstance(idxs, int):
            iterator = []
        else:
            raise ValueError

        if len(iterator) > 0:
            samples = []
            for sample_idx, meta in iterator:
                sample = self._get(sample_idx, meta)
                samples.append(sample)
            return samples
        else:
            idx = idxs
            sample_idx, meta = self._metadata[idx]
            sample = self._get(sample_idx, meta)
            return sample

    def _get_label(self, app_info: dict) -> str:
        label = app_info['type']
        if self._label_fn is not None:
            label = self._label_fn(label)
        return label

    def _get_target(
        self,
        app_meta: dict,
        maxlen: Optional[int] = None,
    ) -> Union[str, tuple[str, list[tuple[int, int]]]]:
        label = self._get_label(app_meta)
        if app_meta.get('on') and app_meta.get('off'):
            assert maxlen is not None
            parse_fn = lambda x: re.findall("\d+", x)
            locs_on = list(map(int, parse_fn(app_meta["on"])))
            locs_off = list(map(int, parse_fn(app_meta["off"])))
            if len(locs_on) - len(locs_off) == 1:
                locs_off.append(maxlen - 1)
            elif len(locs_on) - len(locs_off) > 1:
                print(locs_on)
                print(locs_off)
                raise NotImplementedError
            locs = list(zip(locs_on, locs_off))
            return label, locs
        else:
            return label

    def _get_targets(
        self,
        apps_meta,
        maxlen: int,
    ) -> tuple[list[str], list[int]]:
        labels = []
        locs = []
        for app_meta in apps_meta:
            label, app_locs = self._get_target(app_meta, maxlen=maxlen)
            labels += [label] * len(app_locs)
            locs += app_locs
        if len(labels) > 1:
            ord = sorted(range(len(labels)), key=lambda idx: labels[idx])
            labels = [labels[idx] for idx in ord]
            locs = [locs[idx] for idx in ord]
        return labels, locs

    def _get(
        self,
        sample_idx: int,
        meta: dict,
    ) -> tuple[str, np.ndarray, np.ndarray]:

        # Read the waveforms
        fs = int(meta['header']['sampling_frequency'].replace('Hz', ''))
        filename = '{idx}.{ext}'.format(idx=sample_idx, ext='csv')
        filepath = os.path.join(self._dataset_path, filename)
        waveforms = pd.read_csv(filepath,
                                names=['current', 'voltage'],
                                dtype=self._dtype)
        v = waveforms.voltage.to_numpy()
        i = waveforms.current.to_numpy()

        # Read the meta information about an appliance/appliances
        keys = filter(lambda key: key.startswith('appliance'), meta.keys())
        keys = list(keys)

        if len(keys) != 1:
            raise ValueError('Given meta-data is not supported')

        key = keys[0]
        apps_meta = meta[key]

        if key.endswith('s'):
            y, locs = self._get_targets(apps_meta, len(i))
            return v, i, fs, y, locs
        else:
            app_meta = apps_meta
            y = self._get_target(app_meta)
            return v, i, fs, y

        # power_sample = PowerSample(y, fs, fs_type="high", locs=locs, v=v, i=i)

class UKDale(Reader):
    
    EXT = '.dat'
    HOUSES = [f'house_{i+1}' for i in range(5)]
    
    
    def __init__(
        self, 
        dataset_path: str,
        metadata: Optional[dict]=None,
        metadata_path: Optional[str]=None,
        dtype=None,
        label_fn: Optional[Callable]=None,
        houses: List[int, str]=[]
    ):
        self._dataset_path = dataset_path
        
        self._dtype = np.float32 if dtype is None else dtype
        self._label_fn = label_fn
        
        self._houses = self._collect_houses(houses)
        self.__build_dataset()
        self._dataset_length = self.__compute_len()
        
    def _collect_houses(self, houses):
        if len(houses) == 0:
            _houses = self.HOUSES
        else:
            _houses = []
            for house in houses:
                if type(house) == int or (type(house) == str and house.isnumeric()):
                    house = 'house_' + str(int(house))
                    if house in self.HOUSES:
                        _houses.append(house)
                    else: 
                        print(f'house `{house}` is not presented in the dataset. Skipped')
                        
                elif type(house) == str:
                    if house.lower() in self.HOUSES:
                        _houses.append(house.lower())
                    else: 
                        print(f'house: `{house}` is not presented in the dataset. Skipped')
                else:
                    print(f'house: `{house}` type is unknown. Skipped')
        
        return _houses            
                
    def __compute_len(self):
        return self._data.shape[0]
        
    def __len__(self):
        return self._dataset_length
    
    def __getitem(self, idxs:Union[slice, int]) -> tuple[str, np.ndarray, np.ndarray, int]:
        if type(idxs) == int:
            rows = self._data.loc[idxs:idxs]
        else:
            idxs = slice(idxs.start, idxs.stop-1, idxs.step) # since df.loc[] includes right index
            rows = self._data.loc[idxs]

        houses = rows.house.to_list()
        appliances = rows.appliance.to_list()
        signals = []
        
        for path in rows.metering_path.to_list():
            data = pd.read_csv(path, delimiter=' ', header=None)
            data[0] = pd.to_datetime(data[0], unit='s')          
            signals.append([
                data[0].to_numpy(),
                data[1].to_numpy(dtype=self._dtype)
            ])
          
        return (houses, appliances, signals)          
                    
    def __getitem__(
        self,
        idxs: Union[slice, int]
    ) -> tuple[str, np.ndarray, np.ndarray, int]:
        return self.__getitem(idxs)
    
    
    def __build_dataset(self):
        # parse folder and create dict (or pd.DataFrame) of 
        # year: house_id: appliance: raw_signal 
        
        # all data is to be stored in pd.DataFrame with the following structure 
        '''
        ----------------------------------------------------------------------------------------
        | house_id | appliance | channel | metering_path | button_press_path | metering_units |
        ----------------------------------------------------------------------------------------
        '''
        # columns = ['house_id', 'appliance', 'channel', 'metering_path', 'button_press_path', 'metering_units']
        
        # mapping house id to its appliances and their channes ids
        self._house_labels = {} # house_id: channel_id: appliance_name;  
        self._house_mains = {} # house_id: mains readings
   
        for house in self._houses:
            _dir = os.path.join(self._dataset_path, house)
            labels = {}
            # mains = []
            with open(os.path.join(_dir, 'labels.dat'), 'r') as f:
                for line in f.readlines():
                    line = line.replace('\n', '').split(' ') # line[0] = channel id; line[1] = appliance name
                    labels[int(line[0])] = line[1]
            
            # TODO read and store mains for each house    
            # with open(os.path.join(_dir, 'mains.dat'), 'r') as f:
            #     for line in f.readlines():
            #         line = line.replace('\n', '').split()
            #         mains
                    
        
            self._house_labels[house] = labels
            # self._house_mains[_dir] = mains
        
        # reading all folders to collect houses, appliances (channels) and related files; 
        rows = []
        for house in self._houses:
            for file in os.listdir(os.path.join(self._dataset_path, house)):
                if file.endswith(self.EXT) and len(file.split('_')) == 2:
                    # if file == 'mains.dat':
                    #     # requires separate processing
                    channel = int(file.split('_')[1].split('.')[0])
                    row = {
                        'house': house,
                        'channel': channel,
                        'appliance': self._house_labels[house][channel],
                        'metering_path': os.path.join(self._dataset_path, house, file),
                        'button_press_path': os.path.join(self._dataset_path, house, file.split('.')[0] + '_button_press.dat'),
                        'metering_units': 'W' # TODO read metering units from house metadata (depends on meter type)
                        
                    }
                    row['button_press_path'] = row['button_press_path'] if os.path.exists(row['button_press_path']) else None

                    rows.append(row)

        rows = [pd.DataFrame(row, index=[i]) for i, row in enumerate(rows)]
        self._data = pd.concat(rows)
                    
    
    def __download(self):
        # TODO impelement downloading dataset archive
        # call if not found on the disk
        
        # check and create directory to download to
        # run wget command 
        # wget -e robots=off --mirror --no-parent -r https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017
        
        # check data version -- simple, 2015, 2017 
         
        
        raise NotImplemented