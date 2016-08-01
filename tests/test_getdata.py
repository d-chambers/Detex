# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 21:22:51 2015

@author: derrick
tests for get data module
"""
import pytest
import detex
import obspy
from collections import namedtuple
import pdb

##### General tests on method inputs
class TestFetcherInputs():
    def test_bad_method_type(self):
        with pytest.raises(TypeError):
            detex.getdata.DataFetcher(method=True)
        with pytest.raises(TypeError):
            detex.getdata.DataFetcher(method=Exception)
    
    def test_bad_method_string(self):
        with pytest.raises(ValueError):
            detex.getdata.DataFetcher(method='bob')
        with pytest.raises(ValueError):
            detex.getdata.DataFetcher(method='IRIR')
    
    def test_bad_client_method(self):
        with pytest.raises(ValueError):
            detex.getdata.DataFetcher(method='client')
        with pytest.raises(TypeError):
            detex.getdata.DataFetcher(method='client', client='bob')
    


############# obspy client tests
# named tuple classes for storing info
stinfo_list = ['network', 'station', 'channel']
STinfo = namedtuple('stream_info', stinfo_list)
ClientInfo = namedtuple('ClientInfo', ['ip', 'port'])


# general functions
def get_short_names(cli_info):
    """
    takes an instance of the named tuple class ClientInfo and returns shorter
    names to work with
    """
    return cli_info.network, cli_info.station, cli_info.channel
    
def get_station_inventory(fdsn_client, st_info, t1, t2):
    """
    take an fdsn_client and return response level inventories for given
    parameters
    """
    net, sta, chan = get_short_names(st_info)
    inv = fdsn_client.get_stations(starttime=t1, endtime=t2, level="response",
                                   **st_info._asdict())
    return inv

def check_remove_response(st):
    """
    take an obspy stream and test if its response has been removed for each
    chan. Return True if so, else False
    """
    response_gone = []
    for tr in st:
        proc = st[0].stats['processing']
        response_gone.append(any(['remove_response' in x for x in proc]))
    return all(response_gone)

##### Test data for FDSN clients (ie IRIS)

# client info
fdsn_inf = ClientInfo(ip="IRIS", port=None)
fdsn_net = 'TA'
fdsn_sta = 'M17A'
fdsn_chan = 'BH*'
fdsn_st_info = STinfo(fdsn_net, fdsn_sta, fdsn_chan)
fdsn_t1 = obspy.UTCDateTime('2008-01-01T00-00-00')
fdsn_t2 = obspy.UTCDateTime('2008-01-01T00-01-00')

## test datafetcher for IRIS clients
@pytest.fixture(scope="module")
def init_fdsn_client():
    import obspy.fdsn
    client = obspy.fdsn.Client(fdsn_inf.ip)
    return client

@pytest.fixture(scope="module")
def init_fdsn_fetcher(init_fdsn_client):
    client = init_fdsn_client
    fetcher = detex.getdata.DataFetcher("client", client, removeResponse=True)
    return fetcher

@pytest.fixture(scope="module")
def init_fdsn_getStream(init_fdsn_fetcher):
    fetcher = init_fdsn_fetcher
    net, sta, chan = get_short_names(fdsn_st_info)
    st = fetcher.getStream(fdsn_t1, fdsn_t2, net, sta, chan)
    return st

class TestIRISFetcher():
    def test_fdsn_fetcher(self, init_fdsn_fetcher): # make sure fetcher has correct type/attrs
        fetcher = init_fdsn_fetcher
        assert isinstance(fetcher, detex.getdata.DataFetcher)
        assert hasattr(fetcher, '_getStream')
        assert isinstance(fetcher.client, obspy.fdsn.Client)
    
    def test_fdsn_getStream(self, init_fdsn_getStream):
        st = init_fdsn_getStream
        assert isinstance(st, obspy.Stream)
        assert len(st) > 0
        assert all([len(tr) > 0 for tr in st])
    
    def test_fdsn_remove_response(self, init_fdsn_getStream):
        st = init_fdsn_getStream
        assert check_remove_response(st)


#class Test_Iris:
#    def no_data:
        

##### Test data for earthworm clients
# client info

ew_inf = ClientInfo(ip="pubavo1.wr.usgs.gov", port=16022)
ew_net = 'AV'
ew_sta = 'ACH'
ew_chan = 'EH*'
ew_st_info = STinfo(ew_net, ew_sta, ew_chan)

@pytest.fixture(scope="module")
def init_ew_client():
    import obspy.earthworm
    client = obspy.earthworm.Client(ew_inf.ip, ew_inf.port)
    return client

@pytest.fixture(scope="module")
def init_ew_fetcher(init_ew_client):
    client = init_ew_client
    fetcher = detex.getdata.DataFetcher("client", client, removeResponse=False)
    return fetcher

@pytest.fixture(scope="module")
def get_avail(init_ew_client):
    client = init_ew_client
    avail = client.availability(**ew_st_info._asdict())
    return avail

@pytest.fixture(scope="function")
def init_ew_getStream(get_avail, init_ew_fetcher):
    avail, fetcher = get_avail, init_ew_fetcher
    t1 = avail[0][4] + 100
    t2 = t1 + 4
    net, sta, chan = get_short_names(ew_st_info)
    st = fetcher.getStream(t1, t2, net, sta, chan)
    return st

class TestEWFetcher():
    def test_ew_fetcher(self, init_ew_fetcher): 
        fetcher = init_ew_fetcher
        assert isinstance(fetcher, detex.getdata.DataFetcher)
        assert hasattr(fetcher, '_getStream')
        assert isinstance(fetcher.client, obspy.earthworm.Client)
    
    def test_ew_getStream(self, init_ew_getStream):
        st = init_ew_getStream
        assert isinstance(st, obspy.Stream)
        assert len(st) > 0
        assert all([len(tr) > 0 for tr in st])
    
    def test_ew_remove_response(self, init_fdsn_client, init_ew_fetcher, get_avail):
        fdsn_client = init_fdsn_client
        fetcher = init_ew_fetcher
        avail = get_avail
        t1 = avail[0][4] + 100
        t2 = t1 + 4
        inv = get_station_inventory(fdsn_client, ew_st_info, t1, t2)
        fetcher.removeResponse = True
        fetcher.inventory = inv
        net, sta, chan = get_short_names(ew_st_info)
        st = fetcher.getStream(t1, t2, net, sta, chan)
        assert check_remove_response(st)
        fetcher.removeResponse = False
    
    def test_ew_iris_remove_response(self, init_ew_fetcher, get_avail):
        fetcher = init_ew_fetcher
        avail = get_avail
        t1 = avail[0][4] + 100
        t2 = t1 + 4
        net, sta, chan = get_short_names(ew_st_info)
        #st1 = fetcher.getStream(t1, t2, net, sta, chan)
        fetcher.removeResponse = True
        fetcher.inventoryArg = obspy.fdsn.Client('IRIS')
        st = fetcher.getStream(t1, t2, net, sta, chan)
        assert check_remove_response(st)


##### Test data for NEIC clients
# client info

neic_inf = ClientInfo(ip="137.227.224.97", port=2061)
neic_net = 'IU'
neic_sta = 'ANMO'
neic_chan = 'BH?'
neic_st_info = STinfo(neic_net, neic_sta, neic_chan)
neic_t1 = obspy.UTCDateTime() - 5 * 3600
neic_t2 = neic_t1 + 10

@pytest.fixture(scope="module")
def init_neic_client():
    import obspy.neic
    client = obspy.neic.Client(neic_inf.ip, neic_inf.port)
    return client

@pytest.fixture(scope="module")
def init_neic_fetcher(init_neic_client):
    client = init_neic_client
    fetcher = detex.getdata.DataFetcher("client", client, removeResponse=False)
    return fetcher

@pytest.fixture(scope="function")
def init_neic_getStream(init_neic_fetcher):
    fetcher = init_neic_fetcher
    t1 = neic_t1
    t2 = neic_t2
    net, sta, chan = get_short_names(neic_st_info)
    st = fetcher.getStream(t1, t2, net, sta, chan)
    return st
    
class TestNEICFetcher():
    # make sure fetcher has correct type/attrs
    def test_neic_fetcher(self, init_neic_fetcher): 
        fetcher = init_neic_fetcher
        assert isinstance(fetcher, detex.getdata.DataFetcher)
        assert hasattr(fetcher, '_getStream')
        assert isinstance(fetcher.client, obspy.neic.Client)
    
    def test_neic_getStream(self, init_neic_getStream):
        st = init_neic_getStream
        assert isinstance(st, obspy.Stream)
        assert len(st) > 0
        assert all([len(tr) > 0 for tr in st])
    
    def test_neic_remove_response(self, init_fdsn_client, init_neic_fetcher):
        fdsn_client = init_fdsn_client
        fetcher = init_neic_fetcher
        t1 = neic_t1
        t2 = neic_t2
        inv = get_station_inventory(fdsn_client, neic_st_info, t1, t2)
        fetcher.removeResponse = True
        fetcher.inventory = inv
        net, sta, chan = get_short_names(neic_st_info)
        st = fetcher.getStream(t1, t2, net, sta, chan)
        assert check_remove_response(st)
        fetcher.removeResponse = False
        
    def test_neic_iris_remove_response(self, init_neic_fetcher):
        fetcher = init_neic_fetcher
        fetcher.removeResponse = True
        fetcher.inventoryArg = 'iris'
        t1 = neic_t1
        t2 = neic_t2
        net, sta, chan = get_short_names(neic_st_info)
        st = fetcher.getStream(t1, t2, net, sta, chan)
        assert check_remove_response(st)

    
####### Genral get data tests
    
#def test_general_keys_exists(load_keys):
#    temkey, stakey, phases = load_keys
#

    
    
    
    
    
    
    
    
    
    
    
    