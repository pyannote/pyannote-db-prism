#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr
# Claude BARRAS

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import itertools
import numpy as np
from functools import partial
import os.path as op
from pandas import DataFrame, read_table
from pyannote.database import Database
from pyannote.database.protocol import SpeakerRecognitionProtocol


DATABASES = ['FISHATD5', 'FISHE1',
             'MIX04', 'MIX05', 'MIX06', 'MIX08', 'MIX10',
             'SWCELLP1', 'SWCELLP2', 'SWPH2', 'SWPH3']

FIELDS = ['recording',
          'database',
          'speaker',
          'uri',
          'channel',
          'SESSION_ID',
          'gender',
          'YEAR_OF_BIRTH',
          'YEAR_OF_RECORDING',
          'AGE',
          'speech_type',
          'channel_type',
          'length',
          'language',
          'NATIVE_LANGUAGE',
          'vocal_effort']


class PrismSpeakerRecognitionProtocol(SpeakerRecognitionProtocol):
    """Base speaker recognition protocol for PRISM database

    This class should be inherited from, not used directly.

    Parameters
    ----------
    preprocessors : dict or (key, preprocessor) iterable
        When provided, each protocol item (dictionary) are preprocessed, such
        that item[key] = preprocessor(**item). In case 'preprocessor' is not
        callable, it should be a string containing placeholder for item keys
        (e.g. {'wav': '/path/to/{uri}.wav'})
    """

    def __init__(self, preprocessors={}, databases=None, **kwargs):
        super(PrismSpeakerRecognitionProtocol, self).__init__(
            preprocessors=preprocessors, **kwargs)

        if databases is None:
            databases = DATABASES
        self.databases = databases

        self.recordings_ = self.load_recordings(self.databases)

    def load_recordings(self, databases):
        data_dir = op.join(op.dirname(op.realpath(__file__)), 'data')

        recordings = DataFrame()
        for database in databases:
            path = op.join(data_dir, 'KEYS', '{db}.key'.format(db=database))
            local_keys = read_table(path, delim_whitespace=True, names=FIELDS)
            recordings = recordings.append(local_keys)

        # remove duplicates
        recordings = recordings[~recordings.duplicated()]

        # index using unique recording name
        recordings = recordings.set_index('recording')

        # translate channels (a --> 1, b --> 2, x --> 1)
        func = lambda channel: {'a': 1, 'b': 2, 'x': 1}[channel]
        recordings['channel'] = recordings['channel'].apply(func)

        return recordings


class SRE10(PrismSpeakerRecognitionProtocol):
    """SRE10 speaker recognition protocols

    Parameters
    ----------
    gender : {'f', 'm'}, optional
        Defaults to 'f'.
    condition : {1, 2, 3, 4, 5, 6, 7, 8, 9}, optional
        Defaults to 5.
    preprocessors : dict or (key, preprocessor) iterable
        When provided, each protocol item (dictionary) are preprocessed, such
        that item[key] = preprocessor(**item). In case 'preprocessor' is not
        callable, it should be a string containing placeholder for item keys
        (e.g. {'wav': '/path/to/{uri}.wav'})
    databases : iterable, optional
        Defaults to ['FISHATD5', 'FISHE1', 'MIX04', 'MIX05', 'MIX06', 'MIX08',
        'MIX10', 'SWCELLP1', 'SWCELLP2', 'SWPH2', 'SWPH3']
    """

    def __init__(self, preprocessors={}, condition=5, gender='f', **kwargs):

        super(SRE10, self).__init__(preprocessors=preprocessors, **kwargs)

        self.gender = gender
        self.condition = condition

        self.trn_recordings_ = self._get_trn_recordings()
        self.trn_iter.__func__.n_items = self.trn_recordings_.shape[0]

        self.dev_enroll_recordings_ = self._get_dev_recordings(trn_or_tst='trn')
        self.dev_enroll_iter.__func__.n_items = len(self.dev_enroll_recordings_)

        self.dev_test_recordings_ = self._get_dev_recordings(trn_or_tst='tst')
        self.dev_test_iter.__func__.n_items = len(self.dev_test_recordings_)

        self.dev_keys_ = self._get_dev_keys()

        self.tst_enroll_recordings_ = self._get_tst_recordings(trn_or_tst='trn')
        self.tst_enroll_iter.__func__.n_items = len(self.tst_enroll_recordings_)

        self.tst_test_recordings_ = self._get_tst_recordings(trn_or_tst='tst')
        self.tst_test_iter.__func__.n_items = len(self.tst_test_recordings_)

        self.tst_keys_ = self._get_tst_keys()

    def _filter_by_gender(self, recordings):
        return recordings[recordings['gender'] == self.gender]

    def _filter_by_language(self, recordings):
        # TODO adapt to self.condition
        return recordings[recordings['language'].isin(['ENG', 'USE'])]

    def _filter_by_length(self, recordings):
        # filter short segments as they usually are excerpt of longer segments
        # and therefore do not bring any additional information
        return recordings[recordings['length'] > 100]

    def _filter_by_speech_type(self, recordings):
        # TODO / adapt to condition (1-9)
        return recordings[recordings['speech_type'] == 'tel']

    def _filter_by_channel_type(self, recordings):
        # TODO / adapt to condition (1-9)
        return recordings[recordings['channel_type'] == 'phn']

    def _filter_by_vocal_effort(self, recordings):
        # TODO / adapt to condition (1-9)
        return recordings[~recordings['vocal_effort'].isin(['high', 'low'])]

    def filter(self, recordings):
        recordings = self._filter_by_gender(recordings)
        recordings = self._filter_by_language(recordings)
        recordings = self._filter_by_length(recordings)
        recordings = self._filter_by_speech_type(recordings)
        recordings = self._filter_by_channel_type(recordings)
        recordings = self._filter_by_vocal_effort(recordings)
        return recordings

    def _get_trn_recordings(self):

        recordings = self.filter(self.recordings_)

        # get all MIX08 and MIX10 recordings
        remove = recordings[recordings['database'].isin(['MIX08', 'MIX10'])]

        # get the list of speakers involved in this recordings
        speakers = remove['speaker'].unique()

        # remove all recordings involving those speakers
        # (including recordings from other databases than MIX08 and MIX10)
        recordings = recordings[~recordings['speaker'].isin(speakers)]

        return recordings

    # TRAIN

    def trn_iter(self):
        for recording, item in self.trn_recordings_.iterrows():
            yield recording, dict(item)

    # DEV

    def _get_dev_recordings(self, trn_or_tst='trn'):

        recordings = self.filter(self.recordings_)

        # get MIX08 recordings
        recordings = recordings[recordings['database'] == 'MIX08']

        # only keep speakers with two recordings or more
        speakers, counts = np.unique(recordings['speaker'], return_counts=True)
        keep = [s for s, c in zip(speakers, counts) if c > 1]
        recordings = recordings[recordings['speaker'].isin(keep)]

        # group recordings by speaker
        groups = recordings.groupby('speaker', as_index=False)

        # use first recording as enrollment
        if trn_or_tst == 'trn':
            return groups.nth(0)

        # use second recording as test
        elif trn_or_tst == 'tst':
            return groups.nth(1)

    def dev_enroll_iter(self):
        for recording, item in self.dev_enroll_recordings_.iterrows():
            yield recording, dict(item)

    def dev_test_iter(self):
        for recording, item in self.dev_test_recordings_.iterrows():
            yield recording, dict(item)

    def _get_dev_keys(self):

        trn = self.dev_enroll_recordings_.index
        tst = self.dev_test_recordings_.index
        n_trn, n_tst = len(trn), len(tst)
        data = np.zeros((n_trn, n_tst))

        trn_speakers = self.dev_enroll_recordings_['speaker']
        tst_speakers = self.dev_test_recordings_['speaker']

        n_non_target = 0

        for i, trn_speaker in enumerate(trn_speakers):
            for j, tst_speaker in enumerate(tst_speakers):
                if i <= j:
                    continue
                status = trn_speaker == tst_speaker
                if status:
                    data[i, j] = 1
                else:
                    # only perform one out of 100 non-target tests
                    n_non_target += 1
                    if n_non_target % 100 == 0:
                        data[i, j] = -1

        return DataFrame(data=data, index=trn, columns=tst, dtype=np.int8)

    def dev_keys(self):
        return self.dev_keys_

    # TEST

    def _get_tst_recordings(self, trn_or_tst='trn'):
        data_dir = op.join(op.dirname(op.realpath(__file__)), 'data')
        filename = 'sre10c{0:02d},{1:s}.{2:s}ids'.format(
            self.condition, self.gender, trn_or_tst)
        path = op.join(data_dir, 'TRIALS', 'sre10.conditions', filename)
        with open(path, 'r') as fp:
            recordings = [line.strip() for line in fp]
        return self.recordings_.loc[recordings]

    def tst_enroll_iter(self):
        for recording, item in self.tst_enroll_recordings_.iterrows():
            yield recording, dict(item)

    def tst_test_iter(self):
        for recording, item in self.tst_test_recordings_.iterrows():
            yield recording, dict(item)

    def _get_tst_keys(self):
        """
        Returns
        -------
        keys : pandas.DataFrame
            0: not tested, 1: target, -1: non target
        """

        data_dir = op.join(op.dirname(op.realpath(__file__)), 'data')
        filename = 'sre10c{0:02d},{1:s}.keymask'.format(
            self.condition, self.gender)
        path = op.join(data_dir, 'TRIALS', 'sre10.conditions', filename)

        tst = [rec for rec, _ in self.tst_test_recordings_.iterrows()]
        keys = read_table(path, delim_whitespace=True, names=tst)

        trn = [rec for rec, _ in self.tst_enroll_recordings_.iterrows()]
        keys['trn'] = trn
        keys = keys.set_index('trn')

        return keys

    def tst_keys(self):
        return self.tst_keys_

class Debug(SRE10):
    """Speaker recognition protocols for debugging purposes

    Parameters
    ----------
    preprocessors : dict or (key, preprocessor) iterable
        When provided, each protocol item (dictionary) are preprocessed, such
        that item[key] = preprocessor(**item). In case 'preprocessor' is not
        callable, it should be a string containing placeholder for item keys
        (e.g. {'wav': '/path/to/{uri}.wav'})
    """

    def __init__(self, preprocessors={}, **kwargs):
        super(Debug, self).__init__(
            preprocessors=preprocessors,
            gender='f', condition=5,
            databases=['MIX06', 'MIX08', 'MIX10'],
            **kwargs)

        self.trn_recordings_ = self.trn_recordings_[:100]
        self.trn_iter.__func__.n_items = 100

        self.dev_enroll_recordings_ = self.dev_enroll_recordings_[:100]
        self.dev_enroll_iter.__func__.n_items = 100

        self.dev_test_recordings_ = self.dev_test_recordings_[:100]
        self.dev_test_iter.__func__.n_items = 100

        self.dev_keys_ = self.dev_keys_.iloc[:100, :100]

        self.tst_enroll_recordings_ = self.tst_enroll_recordings_[:100]
        self.tst_enroll_iter.__func__.n_items = 100

        self.tst_test_recordings_ = self.tst_test_recordings_[:100]
        self.tst_test_iter.__func__.n_items = 100

        self.tst_keys_ = self.tst_keys_.iloc[:100, :100]


class Prism(Database):
    """PRISM corpus

Parameters
----------
preprocessors : dict or (key, preprocessor) iterable
    When provided, each protocol item (dictionary) are preprocessed, such
    that item[key] = preprocessor(**item). In case 'preprocessor' is not
    callable, it should be a string containing placeholder for item keys
    (e.g. {'wav': '/path/to/{uri}.wav'})

Reference
---------

Citation
--------

Website
-------
    """

    def __init__(self, preprocessors={}, **kwargs):
        super(Prism, self).__init__(preprocessors=preprocessors, **kwargs)

        self.register_protocol(
             'SpeakerRecognition', 'Debug', Debug)

        PROTOCOL = 'SRE10_c{condition:02d}_{gender}'
        for condition in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            for gender in ('f', 'm'):
                self.register_protocol(
                    'SpeakerRecognition',
                    PROTOCOL.format(condition=condition, gender=gender),
                    partial(SRE10, condition=condition, gender=gender))
