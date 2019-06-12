# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 21:47:27 2016

@author: derrick
tests for construct module
"""
import detex
import os
import numpy as np
import obspy
import pytest

from pathlib2 import Path


test_data = Path(__file__).parent / 'test_data'
##### Tests for waveforms handling
@pytest.fixture(scope='module')
def load_gap_one_chan():
    path_to_gap_one_chan = test_data / 'Misc' / 'Trace_one_chan_gap.pkl'
    st = obspy.read(str(path_to_gap_one_chan))
    return st


@pytest.fixture(scope='module')
def load_gap_three_chan():
    path_to_gap_three_chan2 = test_data / 'Misc' / 'Trace_three_chan2.pkl'
    st = obspy.read(str(path_to_gap_three_chan2))
    return st


@pytest.fixture(scope='module')
def load_gap_all_chans():
    st = obspy.read()
    st = _make_gap(st)
    return st

def _make_gap(st, gap=.1): # makes a gap in the file to test merge
    start = st[0].stats.starttime
    stop = st[0].stats.endtime
    dur = stop - start
    mid = (start.timestamp + stop.timestamp) / 2.
    st1 = st.copy().slice(starttime=start, endtime=obspy.UTCDateTime(mid - gap*dur))
    st2 = st.copy().slice(starttime=obspy.UTCDateTime(mid+gap*dur), endtime=stop)
    return st1 + st2

class Test_merge_channels():
    def test_merge_gap_on_one_chan(self, load_gap_one_chan):
        st = load_gap_one_chan
        st_out = detex.construct._mergeChannels(st)
        nc = len(set([x.stats.channel for x in st_out]))
        assert nc == len(st_out)

    def test_merge_gap_on_three_chan2(self, load_gap_three_chan):
        st = load_gap_three_chan
        st_out = detex.construct._mergeChannels(st)
        nc = len(set([x.stats.channel for x in st_out]))
        assert nc == len(st_out)

    def test_merge_gap_on_three_chan(self, load_gap_all_chans):
        st = load_gap_all_chans
        st_out = detex.construct._mergeChannels(st)
        nc = len(set([x.stats.channel for x in st_out]))
        assert nc == len(st_out)