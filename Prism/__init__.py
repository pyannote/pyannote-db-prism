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
import os.path as op
from pandas import DataFrame, read_table
from pyannote.database import Database
from pyannote.database.protocol import SpeakerRecognitionProtocol


DATABASES = ['FISHATD5', 'FISHE1',
             'MIX04', 'MIX05', 'MIX06', 'MIX08', 'MIX10',
             'SWCELLP1', 'SWCELLP2', 'SWPH2', 'SWPH3']

FIELDS = ['unique_name',
          'database',
          'target',
          'uri',
          'channel',
          'SESSION_ID',
          'gender',
          'YEAR_OF_BIRTH',
          'YEAR_OF_RECORDING',
          'AGE',
          'SPEECH_TYPE',
          'CHANNEL_TYPE',
          'NOMINAL_LENGTH',
          'language',
          'NATIVE_LANGUAGE',
          'VOCAL_EFFORT']


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

    def __init__(self, preprocessors={}, **kwargs):
        super(PrismSpeakerRecognitionProtocol, self).__init__(
            preprocessors=preprocessors, **kwargs)

        self.databases = DATABASES
        self.keys_ = self.read_keys(self.databases)

    def read_keys(self, databases):

        data_dir = op.join(op.dirname(op.realpath(__file__)), 'data')

        keys = DataFrame()
        for database in databases:
            path = op.join(data_dir, 'KEYS', '{db}.key'.format(db=database))
            local_keys = read_table(path, delim_whitespace=True, names=FIELDS)
            keys = keys.append(local_keys)

        # remove duplicates
        keys = keys[~keys.duplicated()]

        # index using 'session' unique recording name
        keys = keys.set_index('unique_name')

        # translate channels (a --> 1, b --> 2, x --> 1)
        func = lambda channel: {'a': 1, 'b': 2, 'x': 1}[channel]
        keys['channel'] = keys['channel'].apply(func)

        return keys


class Debug(PrismSpeakerRecognitionProtocol):
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
        super(Debug, self).__init__(preprocessors=preprocessors, **kwargs)
        self.gender = 'f'
        self.condition = 5

    def _listing(self, trn_or_tst='trn'):
        data_dir = op.join(op.dirname(op.realpath(__file__)), 'data')
        filename = 'sre10c{0:02d},{1:s}.{2:s}ids'.format(
            self.condition, self.gender, trn_or_tst)
        path = op.join(data_dir, 'TRIALS', 'sre10.conditions', filename)
        with open(path, 'r') as fp:
            unique_names = [line.strip() for line in fp]
        return unique_names

    def trn_iter(self):

        keys = self.keys_

        # filter keys based on gender
        keys = keys[keys['gender'] == self.gender]

        # filter keys based on (language in ['ENG', 'USE'])
        # TODO / adapt to condition
        keys = keys[keys['language'].isin(['ENG', 'USE'])]

        # filter keys based on self.condition (NOMINAL_LENGTH > ...)
        # TODO / adapt to condition

        # filter keys based on (SPEECH_TYPE == 'tel')
        # TODO / adapt to condition
        keys = keys[keys['SPEECH_TYPE'] == 'tel']

        # filter keys based on (CHANNEL_TYPE == 'phn')
        # TODO / adapt to condition
        keys = keys[keys['CHANNEL_TYPE'] == 'phn']

        # filter keys based on (VOCAL_EFFORT not in ['high', 'low'])
        # TODO / adapt to condition
        keys = keys[~keys['VOCAL_EFFORT'].isin(['high', 'low'])]

        # filter keys that **are** part of MIX10
        keys = keys[keys['database'] == 'MIX10']

        # remove keys so that targets are **not**
        # part of {dev|tst}_{enroll|test}
        iterator = itertools.chain(self.dev_enroll_iter(),
                                   self.dev_test_iter(),
                                   self.tst_enroll_iter(),
                                   self.tst_test_iter())
        skip_targets = set(item['target'] for _, item in iterator)
        keys = keys[~keys['target'].isin(skip_targets)]

        for i, (unique_name, row) in enumerate(keys.iterrows()):
            yield unique_name, dict(row)
            if i > 20:
                break

    def dev_enroll_iter(self):
        return []

    def dev_test_iter(self):
        return []

    def tst_enroll_iter(self):
        for unique_name in self._listing(trn_or_tst='trn')[:10]:
            yield unique_name, dict(self.keys_.loc[unique_name])

    def tst_test_iter(self):
        for unique_name in self._listing(trn_or_tst='tst')[:10]:
            yield unique_name, dict(self.keys_.loc[unique_name])

    def tst_keys(self):

        data_dir = op.join(op.dirname(op.realpath(__file__)), 'data')
        filename = 'sre10c{0:02d},{1:s}.keymask'.format(
            self.condition, self.gender)
        path = op.join(data_dir, 'TRIALS', 'sre10.conditions', filename)

        trn = self._listing(trn_or_tst='trn')[:10]
        tst = self._listing(trn_or_tst='tst')[:10]
        keys = read_table(path, delim_whitespace=True, names=tst)
        keys['trn'] = trn
        keys = keys.set_index('trn')
        return keys


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
    """

    def __init__(self, preprocessors={}, gender='f', condition=5, **kwargs):
        super(SRE10, self).__init__(preprocessors=preprocessors, **kwargs)
        self.gender = gender
        self.condition = condition

    def _listing(self, trn_or_tst='trn'):
        data_dir = op.join(op.dirname(op.realpath(__file__)), 'data')
        filename = 'sre10c{0:02d},{1:s}.{2:s}ids'.format(
            self.condition, self.gender, trn_or_tst)
        path = op.join(data_dir, 'TRIALS', 'sre10.conditions', filename)
        with open(path, 'r') as fp:
            unique_names = [line.strip() for line in fp]
        return unique_names

    def trn_iter(self):

        keys = self.keys_

        # filter keys based on gender
        keys = keys[keys['gender'] == self.gender]

        # filter keys based on (language in ['ENG', 'USE'])
        # TODO / adapt to condition
        keys = keys[keys['language'].isin(['ENG', 'USE'])]

        # filter short segments as they usually are excerpt of longer segments
        # and therefore do not bring any additional information
        keys = keys[keys['NOMINAL_LENGTH'] > 100]

        # filter keys based on (SPEECH_TYPE == 'tel')
        # TODO / adapt to condition
        keys = keys[keys['SPEECH_TYPE'] == 'tel']

        # filter keys based on (CHANNEL_TYPE == 'phn')
        # TODO / adapt to condition
        keys = keys[keys['CHANNEL_TYPE'] == 'phn']

        # filter sessions based on (VOCAL_EFFORT not in ['high', 'low'])
        # TODO / adapt to condition
        keys = keys[~keys['VOCAL_EFFORT'].isin(['high', 'low'])]

        # filter targets that are part of MIX10 (used in SRE10 conditions)
        keys = keys[~keys['database'] == 'MIX10']

        for unique_name, row in keys.iterrows():
            yield unique_name, dict(row)

    def dev_enroll_iter(self):
        return []

    def dev_test_iter(self):
        return []

    def tst_enroll_iter(self):
        for unique_name in self._listing(trn_or_tst='trn'):
            yield unique_name, dict(self.keys_.loc[unique_name])

    def tst_test_iter(self):
        for unique_name in self._listing(trn_or_tst='tst'):
            yield unique_name, dict(self.keys_.loc[unique_name])

    def tst_keys(self):

        data_dir = op.join(op.dirname(op.realpath(__file__)), 'data')
        filename = 'sre10c{0:02d},{1:s}.keymask'.format(
            self.condition, self.gender)
        path = op.join(data_dir, 'TRIALS', 'sre10.conditions', filename)

        trn = self._listing(trn_or_tst='trn')
        tst = self._listing(trn_or_tst='tst')
        keys = read_table(path, delim_whitespace=True, names=tst)
        keys['trn'] = trn
        keys = keys.set_index('trn')
        return keys


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

        self.register_protocol(
             'SpeakerRecognition', 'SRE10', SRE10)
