"""
Modified Python package based on this one:
https://github.com/miili/StreamPick
"""

import os
import pickle

import PyQt4
from PyQt4 import QtGui, QtCore
from obspy.core import event, UTCDateTime
from itertools import cycle
import sys
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

import numpy as np

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.transforms import offset_copy


#
# def streamPick(stream=None, parent=None, ap=None):
#    print 'starting PyQt4'
#    reload(PyQt4)
#    out = dummy()
#    #qApp = PyQt4.QtGui.QApplication(sys.argv)
#    pk = streamPicka(stream=stream, parent=parent, ap=ap, ou=out)
#    pk = None
#    #qApp.deleteLater()
#
#    print 'outout'
#    return out
#
# class dummy(object):
#    def __init__(self):
#        self.KeepGoing = False
#        self._picks = {}


class streamPick(PyQt4.QtGui.QMainWindow):
    def __init__(self, stream=None, parent=None, ap=None):
        # Initialising PyQt4.QtGui
        if ap is None:
            self.qApp = PyQt4.QtGui.QApplication(sys.argv)
        else:
            self.qApp = ap
        self.KeepGoing = False

        # Init vars
        if stream is None:
            msg = 'Define stream = obspy.core.Stream()'
            raise ValueError(msg)
        self.st = stream.copy()
        self.st.merge()
        self._picks = []
        self.savefile = None
        self.onset_types = ['emergent', 'impulsive', 'questionable']

        # Load filters from pickle if they exist
        try:
            self.bpfilter = pickle.load(open('.pick_filters', 'r'))
        except:
            self.bpfilter = []
        # Internal variables
        # Gui vars
        self._shortcuts = {'st_next': 'c',
                           'st_previous': 'x',
                           'filter_apply': 'f',
                           'pick_p': 'q',
                           'pick_p_end': 'a',
                           'pick_s': 'w',
                           'pick_s_end': 's',
                           'pick_custom': 't',
                           'pick_remove': 'r',
                           'gain_up': '1',
                           'gain_down': '2',
                           'stream_next': 'v'
                           }
        self._plt_drag = None
        self._current_filter = None
        # Init stations
        self._initStations()  # defines list self._stations
        self._stationCycle = cycle(self._stations)
        self._streamStation(self._stationCycle.next())
        # Init PyQt4.QtGui
        PyQt4.QtGui.QMainWindow.__init__(self)
        self.setupUI()
        # exec QtApp
        self.qApp.exec_()
        # self.qApp.deleteLater()

    def setupUI(self):
        '''
        Setup the UI
        '''
        self.main_widget = PyQt4.QtGui.QWidget(self)
        # Init parts of the UI
        self._initMenu()
        self._createStatusBar()
        self._initPlots()
        self._wadatiPlt = None

        # Define layout
        l = PyQt4.QtGui.QVBoxLayout(self.main_widget)
        # self.setLayout(l)

        l.addLayout(self.btnbar)
        l.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)
        self.setGeometry(300, 300, 1200, 800)
        self.setWindowTitle('obspy.core.Stream-Picker')
        self.show()

    def _killLayout():
        pass

    def _initPlots(self):
        self.fig = Figure(facecolor='.86', dpi=72, frameon=True)
        # Change facecolor
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(PyQt4.QtCore.Qt.StrongFocus)
        # Draw the matplotlib figure
        self._drawFig()
        # Connect the events
        self.fig.canvas.mpl_connect('scroll_event',
                                    self._pltOnScroll)
        self.fig.canvas.mpl_connect('motion_notify_event',
                                    self._pltOnDrag)
        self.fig.canvas.mpl_connect('button_release_event',
                                    self._pltOnButtonRelease)
        self.fig.canvas.mpl_connect('button_press_event',
                                    self._pltOnButtonPress)

    def _initMenu(self):
        # Next and Prev Button
        nxt = PyQt4.QtGui.QPushButton('NextSta >>',
                                      shortcut=self._shortcuts['st_next'], parent=self.main_widget)
        nxt.clicked.connect(self._pltNextStation)
        nxt.setToolTip('shortcut <b>c</d>')
        nxt.setMaximumWidth(150)
        prv = PyQt4.QtGui.QPushButton('<< Prev',
                                      shortcut=self._shortcuts['st_previous'], parent=self.main_widget)
        prv.clicked.connect(self._pltPrevStation)
        prv.setToolTip('shortcut <b>x</d>')
        prv.setMaximumWidth(150)

        # Stations drop-down
        self.stcb = PyQt4.QtGui.QComboBox(self)
        for st in self._stations:
            self.stcb.addItem(st)
        self.stcb.activated.connect(self._pltStation)
        self.stcb.setMaximumWidth(100)
        self.stcb.setMinimumWidth(80)

        # Filter buttons
        self.fltrbtn = PyQt4.QtGui.QPushButton('Filter Trace',
                                               shortcut=self._shortcuts['filter_apply'])
        self.fltrbtn.setToolTip('shortcut <b>f</b>')
        self.fltrbtn.setCheckable(True)
        # self.fltrbtn.setAutoFillBackground(True)
        # self.fltrbtn.setStyleSheet(PyQt4.QtCore.QString(
        # 'QPushButton:checked {background-color: lightgreen;}'))
        self.fltrbtn.clicked.connect(self._appFilter)

        self.fltrcb = PyQt4.QtGui.QComboBox(self)
        self.fltrcb.activated.connect(self._changeFilter)
        self.fltrcb.setMaximumWidth(170)
        self.fltrcb.setMinimumWidth(150)
        self._updateFilterCB()  # fill QComboBox

        # edit/delete filer buttons
        fltredit = PyQt4.QtGui.QPushButton('Edit')
        fltredit.resize(fltredit.sizeHint())
        fltredit.clicked.connect(self._editFilter)

        fltrdel = PyQt4.QtGui.QPushButton('Delete')
        fltrdel.resize(fltrdel.sizeHint())
        fltrdel.clicked.connect(self._deleteFilter)

        nxtstr = PyQt4.QtGui.QPushButton('NextStr >>', shortcut=self._shortcuts['stream_next'])
        nxtstr.clicked.connect(self._pltNextStream)
        nxtstr.setToolTip('shortcut <b>v</d>')
        nxtstr.setMaximumWidth(150)

        btnstyle = PyQt4.QtGui.QFrame(fltredit)
        btnstyle.setFrameStyle(PyQt4.QtGui.QFrame.Box | PyQt4.QtGui.QFrame.Plain)
        btnstyle = PyQt4.QtGui.QFrame(fltrdel)
        btnstyle.setFrameStyle(PyQt4.QtGui.QFrame.Box | PyQt4.QtGui.QFrame.Plain)

        # onset type
        _radbtn = []
        for _o in self.onset_types:
            _radbtn.append(PyQt4.QtGui.QRadioButton(str(_o[0].upper())))
            _radbtn[-1].setToolTip('Onset ' + _o)
            _radbtn[-1].clicked.connect(self._drawPicks)
            if _o == 'impulsive':
                _radbtn[-1].setChecked(True)
        self.onsetGrp = PyQt4.QtGui.QButtonGroup()
        self.onsetGrp.setExclusive(True)
        onsetbtns = PyQt4.QtGui.QHBoxLayout()

        for _i, _btn in enumerate(_radbtn):
            self.onsetGrp.addButton(_btn, _i)
            onsetbtns.addWidget(_btn)

        # Arrange buttons
        vline = PyQt4.QtGui.QFrame()
        vline.setFrameStyle(PyQt4.QtGui.QFrame.VLine | PyQt4.QtGui.QFrame.Raised)
        self.btnbar = PyQt4.QtGui.QHBoxLayout()
        self.btnbar.addWidget(prv)
        self.btnbar.addWidget(nxt)
        self.btnbar.addWidget(nxtstr)
        self.btnbar.addWidget(PyQt4.QtGui.QLabel('Station'))
        self.btnbar.addWidget(self.stcb)
        ##
        self.btnbar.addWidget(vline)
        self.btnbar.addWidget(self.fltrbtn)
        self.btnbar.addWidget(self.fltrcb)
        self.btnbar.addWidget(fltredit)
        self.btnbar.addWidget(fltrdel)
        ##
        self.btnbar.addWidget(vline)
        self.btnbar.addWidget(PyQt4.QtGui.QLabel('Pick Onset: '))
        self.btnbar.addLayout(onsetbtns)
        self.btnbar.addStretch(3)

        # Menubar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(PyQt4.QtGui.QIcon().fromTheme('document-save'),
                           'Save', self._saveCatalog)
        fileMenu.addAction(PyQt4.QtGui.QIcon().fromTheme('document-save'),
                           'Save as QuakeML File', self._saveCatalogDlg)
        fileMenu.addAction(PyQt4.QtGui.QIcon().fromTheme('document-open'),
                           'Load QuakeML File', self._openCatalogDlg)
        fileMenu.addSeparator()
        fileMenu.addAction('Save Plot', self._savePlotDlg)
        fileMenu.addSeparator()
        fileMenu.addAction(PyQt4.QtGui.QIcon().fromTheme('application-exit'),
                           'Exit', self._hardExit)
        # windowMenu = menubar.addMenu('&Windows')
        # windowMenu.addAction('Wadati Diagram', self._opnWadatiPlot)
        aboutMenu = menubar.addMenu('&About')
        aboutMenu.addAction(PyQt4.QtGui.QIcon().fromTheme('info'),
                            'Info', self._infoDlg)

    def _hardExit(self):
        # self.qApp.deleteLater()

        self.deleteLater()
        # self.close()
        # sys.exit()

    def _drawFig(self):
        '''
        Draws all matplotlib figures
        '''
        num_plots = len(self._current_st)
        self.fig.clear()
        self._appFilter(draw=False)
        for _i, tr in enumerate(self._current_st):
            ax = self.fig.add_subplot(num_plots, 1, _i + 1)
            ax.plot(tr.data, 'k')
            ax.axhline(0, color='k', alpha=.05)
            ax.set_xlim([0, tr.data.size])
            ax.text(.925, .9, self._current_st[_i].stats.channel,
                    transform=ax.transAxes, va='top', ma='left')
            ax.channel = tr.stats.channel
            if _i == 0:
                ax.set_xlabel('Seconds')

        # plot picks
        self._drawPicks(draw=False)
        self.fig.suptitle('%s - %s - %s' % (self._current_st[-1].stats.network,
                                            self._current_st[-1].stats.station,
                                            self._current_st[-1].stats.starttime.isoformat()),
                          x=.2)
        self._updateSB()
        self._canvasDraw()

    def _initStations(self):
        '''
        Creates a list holding unique station names
        '''
        self._stations = []
        for _tr in self.st:
            if _tr.stats.station not in self._stations:
                self._stations.append(_tr.stats.station)
        self._stations.sort()

    def _getPhases(self):
        '''
        Creates a list holding unique phase names
        '''
        phases = []
        for _pick in self._picks:
            if _pick.phase_hint not in phases:
                phases.append(_pick.phase_hint)
        return phases

    def _streamStation(self, station):
        '''
        Copies the current stream object from self.st through
        obspy.stream.select(station=)
        '''
        if station not in self._stations:
            return
        self._current_st = self.st.select(station=station).copy()
        self._current_stname = station
        self._current_network = self._current_st[0].stats.network
        # Sort and detrend streams
        self._current_st.sort(['channel'])
        try:
            self._current_st.detrend('linear')
        except:
            pass

    def _setPick(self, xdata, phase, channel, polarity='undecideable'):
        '''
        Write obspy.core.event.Pick into self._picks list
        '''
        picktime = self._current_st[0].stats.starttime + \
                   (xdata * self._current_st[0].stats.delta)

        this_pick = event.Pick()
        overwrite = True
        # Overwrite existing phase's picktime
        for _pick in self._getPicks():
            if _pick.phase_hint == phase and \
                            _pick.waveform_id.channel_code == channel:
                this_pick = _pick
                overwrite = False
                break

        creation_info = event.CreationInfo(
            author='ObsPy.Stream.pick()',
            creation_time=UTCDateTime())
        # Create new event.Pick()
        this_pick.time = picktime
        this_pick.phase_hint = phase
        this_pick.waveform_id = event.WaveformStreamID(
            network_code=self._current_st[0].stats.network,
            station_code=self._current_st[0].stats.station,
            location_code=self._current_st[0].stats.location,
            channel_code=channel)
        this_pick.evaluation_mode = 'manual'
        this_pick.creation_info = creation_info
        this_pick.onset = self.onset_types[self.onsetGrp.checkedId()]
        this_pick.evaluation_status = 'preliminary'
        this_pick.polarity = polarity
        if self._current_filter is not None:
            this_pick.comments.append(event.Comment(
                text=str(self.bpfilter[self.fltrcb.currentIndex()])))
        if overwrite:
            self._picks.append(this_pick)

    def _delPicks(self, network, station, channel):
        '''
        Deletes pick from catalog
        '''
        for _i, _pick in enumerate(self._picks):
            if _pick.waveform_id.network_code == network \
                    and _pick.waveform_id.station_code == station \
                    and _pick.waveform_id.channel_code == channel:
                self._picks.remove(_pick)

    def _getPicks(self):
        '''
        Create a list of picks for the current plot
        '''
        this_st_picks = []
        for _i, pick in enumerate(self._picks):
            if pick.waveform_id.station_code == self._current_stname and \
                                    self._current_st[0].stats.starttime < \
                                    pick.time < self._current_st[0].stats.endtime:
                this_st_picks.append(_i)
        return [self._picks[i] for i in this_st_picks]

    def _getPickXPosition(self, picks):
        '''
        Convert picktimes into relative positions along x-axis
        '''
        xpicks = []
        for _pick in picks:
            xpicks.append((_pick.time - self._current_st[0].stats.starttime)
                          / self._current_st[0].stats.delta)
        return np.array(xpicks)

    def _drawPicks(self, draw=True):
        '''
        Draw picklines onto axes
        '''
        picks = self._getPicks()
        xpicks = self._getPickXPosition(picks)

        for _ax in self.fig.get_axes():
            lines = []
            labels = []
            transOffset = offset_copy(_ax.transData, fig=self.fig,
                                      x=5, y=0, units='points')
            for _i, _xpick in enumerate(xpicks):
                if picks[_i].phase_hint == 'S':
                    color = 'r'
                elif picks[_i].phase_hint == 'P':
                    color = 'g'
                else:
                    color = 'b'
                if _ax.channel != picks[_i].waveform_id.channel_code:
                    alpha = .1
                else:
                    alpha = .8

                lines.append(matplotlib.lines.Line2D([_xpick, _xpick],
                                                     [_ax.get_ylim()[0] * .9, _ax.get_ylim()[1] * .8],
                                                     color=color, alpha=alpha))
                lines[-1].obspy_pick = picks[_i]

                labels.append(matplotlib.text.Text(_xpick,
                                                   _ax.get_ylim()[0] * .8, text=picks[_i].phase_hint,
                                                   color=color, size=10, alpha=alpha,
                                                   transform=transOffset))

            # delete all artists
            del _ax.artists[0:]
            # add updated objects
            for line in lines:
                _ax.add_artist(line)
            for label in labels:
                _ax.add_artist(label)

        if draw:
            self._canvasDraw()

    # Plot Controls
    def _pltOnScroll(self, event):
        '''
        Scrolls/Redraws the plots along x axis
        '''
        if event.inaxes is None:
            return

        if event.key == 'control':
            axes = [event.inaxes]
        else:
            axes = self.fig.get_axes()

        for _ax in axes:
            left = _ax.get_xlim()[0]
            right = _ax.get_xlim()[1]
            extent = right - left
            dzoom = .2 * extent
            aspect_left = (event.xdata - _ax.get_xlim()[0]) / extent
            aspect_right = (_ax.get_xlim()[1] - event.xdata) / extent

            if event.button == 'up':
                left += dzoom * aspect_left
                right -= dzoom * aspect_right
            elif event.button == 'down':
                left -= dzoom * aspect_left
                right += dzoom * aspect_right
            else:
                return
            _ax.set_xlim([left, right])
        self._canvasDraw()

    def _pltOnDrag(self, event):
        '''
        Drags/Redraws the plot upon drag
        '''
        if event.inaxes is None:
            return

        if event.key == 'control':
            axes = [event.inaxes]
        else:
            axes = self.fig.get_axes()

        if event.button == 2:
            if self._plt_drag is None:
                self._plt_drag = event.xdata
                return
            for _ax in axes:
                _ax.set_xlim([_ax.get_xlim()[0] +
                              (self._plt_drag - event.xdata),
                              _ax.get_xlim()[1] + (self._plt_drag - event.xdata)])
        else:
            return
        self._canvasDraw()

    def _pltOnButtonRelease(self, event):
        '''
        On Button Release Reset drag variable
        '''
        self._plt_drag = None

    def _pltOnButtonPress(self, event):
        '''
        This Function is evoked when the user picks
        '''
        if event.key is not None:
            event.key = event.key.lower()
        if event.inaxes is None:
            return
        channel = event.inaxes.channel
        tr_amp = event.inaxes.lines[0].get_ydata()[int(event.xdata) + 3] - \
                 event.inaxes.lines[0].get_ydata()[int(event.xdata)]
        if tr_amp < 0:
            polarity = 'negative'
        elif tr_amp > 0:
            polarity = 'positive'
        else:
            polarity = 'undecideable'

        if event.key == self._shortcuts['pick_p'] and event.button == 1:
            self._setPick(event.xdata, phase='P', channel=channel,
                          polarity=polarity)
        elif event.key == self._shortcuts['pick_p_end'] and event.button == 1:
            self._setPick(event.xdata, phase='Pend', channel=channel,
                          polarity=polarity)

        elif event.key == self._shortcuts['stream_next']:
            self._pltNextStream()
        elif event.key == self._shortcuts['pick_s_end'] and event.button == 1:
            self._setPick(event.xdata, phase='Send', channel=channel,
                          polarity=polarity)

        elif event.key == self._shortcuts['pick_s'] and event.button == 1:
            self._setPick(event.xdata, phase='S', channel=channel,
                          polarity=polarity)
        elif event.key == self._shortcuts['pick_custom'] and event.button == 1:
            text, ok = PyQt4.QtGui.QInputDialog.getItem(self, 'Custom Phase',
                                                        'Enter phase name:', self._getPhases())
            if ok:
                self._setPick(event.xdata, phase=text, channel=channel,
                              polarity=polarity)
        elif event.key == self._shortcuts['pick_remove']:
            self._delPicks(network=self._current_network,
                           station=self._current_stname,
                           channel=channel)

        elif event.key == self._shortcuts['gain_up'] or self._shortcuts['gain_down']:
            self._adjustGain(event)
        else:
            return
        self._updateSB()
        self._drawPicks()

    def _adjustGain(self, event):
        '''
        Allows the gain to be changed on plot
        '''

        if event.key is not None:
            event.key = event.key.lower()
        if event.inaxes is None:
            return

        if event.inaxes is None:
            return

        if event.key == self._shortcuts['gain_up'] or self._shortcuts['gain_down']:
            axes = [event.inaxes]
        else:
            axes = self.fig.get_axes()

        for _ax in axes:
            up = _ax.get_ylim()[1]
            down = _ax.get_ylim()[0]
            extent = up - down
            dzoom = .2 * extent
            aspect_up = (event.ydata - _ax.get_ylim()[0]) / extent
            aspect_down = (_ax.get_ylim()[1] - event.ydata) / extent

            if event.key == self._shortcuts['gain_up']:
                down += dzoom * aspect_up
                up -= dzoom * aspect_down
            elif event.key == self._shortcuts['gain_down']:
                down -= dzoom * aspect_up
                up += dzoom * aspect_down
            else:
                return
            _ax.set_ylim([down, up])
        self._canvasDraw()

    def _pltNextStation(self):
        '''
        Plot next station
        '''
        self._streamStation(self._stationCycle.next())
        self._drawFig()

    def _pltNextStream(self):
        '''
        Plot next Stream, used primarily in Detex loops when
        streamPick is called to know to exit loop or go
        to next stream
        '''

        self.KeepGoing = True
        pickle.dump(self.bpfilter, open('.pick_filters', 'w'))
        # self.closeEvent()
        self.deleteLater()

    def _pltPrevStation(self):
        '''
        Plot previous station
        '''
        for _i in range(len(self._stations) - 1):
            prevStation = self._stationCycle.next()
        self._streamStation(prevStation)
        self._drawFig()

    def _pltStation(self):
        '''
        Plot station from DropDown Menu
        '''
        _i = self.stcb.currentIndex()
        while self._stationCycle.next() != self._stations[_i]:
            pass
        self._streamStation(self._stations[_i])
        self._drawFig()

    # Filter functions
    def _appFilter(self, button=True, draw=True):
        '''
        Apply bandpass filter
        '''
        _i = self.fltrcb.currentIndex()
        self._streamStation(self._current_stname)
        if self.fltrbtn.isChecked() is False:
            self._current_filter = None
        else:
            self._current_st.filter('bandpass',
                                    freqmin=self.bpfilter[_i]['freqmin'],
                                    freqmax=self.bpfilter[_i]['freqmax'],
                                    corners=self.bpfilter[_i]['corners'],
                                    zerophase=True)
            self._current_filter = _i
        for _i, _ax in enumerate(self.fig.get_axes()):
            if len(_ax.lines) == 0:
                continue
            _ax.lines[0].set_ydata(self._current_st[_i].data)
            _ax.relim()
            _ax.autoscale_view()
        if draw is True:
            self._drawPicks(draw=False)
            self._canvasDraw()
        self._updateSB()

    def _newFilter(self):
        '''
        Create new filter
        '''
        newFilter = self.defFilter(self)
        if newFilter.exec_():
            self.bpfilter.append(newFilter.getValues())
            self._updateFilterCB()
            self.fltrcb.setCurrentIndex(len(self.bpfilter) - 1)
            self._appFilter()

    def _editFilter(self):
        '''
        Edit existing filter
        '''
        _i = self.fltrcb.currentIndex()
        this_filter = self.bpfilter[_i]
        editFilter = self.defFilter(self, this_filter)
        if editFilter.exec_():
            self.bpfilter[_i] = editFilter.getValues()
            self._updateFilterCB()
            self.fltrcb.setCurrentIndex(_i)
            self._appFilter()

    def _deleteFilter(self):
        '''
        Delete filter
        '''
        pass
        _i = self.fltrcb.currentIndex()

    #        self.fltrbtn.setChecked(False)
    #        self.bpfilter.pop(_i)
    #        self._updateFilterCB()
    #        self._appFilter()

    def _changeFilter(self, index):
        '''
        Evoke this is filter in drop-down is changed
        '''
        if index == len(self.bpfilter):
            return self._newFilter()
        else:
            return self._appFilter()

    def _updateFilterCB(self):
        '''
        Update the filter QComboBox
        '''
        self.fltrcb.clear()
        self.fltrcb.setCurrentIndex(-1)
        for _i, _f in enumerate(self.bpfilter):
            self.fltrcb.addItem('%s [%.2f - %.2f Hz]' % (_f['name'],
                                                         _f['freqmin'], _f['freqmax']))
        self.fltrcb.addItem('Create new Filter...')

    # Status bar functions
    def _createStatusBar(self):
        '''
        Creates the status bar
        '''
        sb = PyQt4.QtGui.QStatusBar()
        sb.setFixedHeight(18)
        self.setStatusBar(sb)
        self.statusBar().showMessage('Ready')

    def _updateSB(self, statustext=None):
        '''
        Updates the statusbar text
        '''
        if statustext is None:
            self.stcb.setCurrentIndex(
                self._stations.index(self._current_stname))
            msg = 'Station %i/%i - %i Picks' % (
                self._stations.index(self._current_stname) + 1,
                len(self._stations), len(self._getPicks()))
            if self._current_filter is not None:
                msg += ' - Bandpass %s [%.2f - %.2f Hz]' % (
                    self.bpfilter[self._current_filter]['name'],
                    self.bpfilter[self._current_filter]['freqmin'],
                    self.bpfilter[self._current_filter]['freqmax'])
            else:
                msg += ' - Raw Data'
            self.statusBar().showMessage(msg)

    def _openCatalogDlg(self):
        filename = PyQt4.QtGui.QFileDialog.getOpenFileName(self,
                                                           'Load QuakeML Picks',
                                                           os.getcwd(), 'QuakeML Format (*.xml)', '20')
        if filename:
            self._openCatalog(str(filename))
            self.savefile = str(filename)

    def _openCatalog(self, filename):
        '''
        Open existing QuakeML catalog
        '''
        try:
            print('Opening QuakeML Catalog %s' % filename)
            cat = event.readEvents(filename)
            self._picks = cat[0].picks
            self._drawPicks()
        except:
            msg = 'Could not open QuakeML file %s' % (filename)
            raise IOError(msg)

    def _saveCatalogDlg(self):
        '''
        Save catalog through QtDialog
        '''
        self.savefile = PyQt4.QtGui.QFileDialog.getSaveFileName(self,
                                                                'Save QuakeML Picks',
                                                                os.getcwd(), 'QuakeML Format (*.xml)')
        if not self.savefile:
            self.savefile = None
            return
        self.savefile = str(self.savefile)
        if os.path.splitext(self.savefile)[1].lower() != '.xml':
            self.savefile += '.xml'
        self._saveCatalog()

    def _saveCatalog(self, filename=None):
        '''
        Saves the catalog to filename
        '''
        if self.savefile is None and filename is None:
            return self._saveCatalogDlg()
        if filename is not None:
            savefile = filename
        else:
            savefile = self.savefile
        cat = event.Catalog()
        cat.events.append(event.Event(picks=self._picks))
        # cat.write(savefile, format='QUAKEML')
        # print 'Picks saved as %s' % savefile

    def _savePlotDlg(self):
        '''
        Save Plot Image Qt Dialog and Matplotlib wrapper
        '''
        filename = PyQt4.QtGui.QFileDialog.getSaveFileName(self, 'Save Plot',
                                                           os.getcwd(),
                                                           'Image Format (*.png *.pdf *.ps *.svg *.eps)')
        if not filename:
            return
        filename = str(filename)
        format = os.path.splitext(filename)[1][1:].lower()
        if format not in ['png', 'pdf', 'ps', 'svg', 'eps']:
            format = 'png'
            filename += '.' + format
        self.fig.savefig(filename=filename, format=format, dpi=72)

    def getPicks(self):
        return self._picks

    def _opnWadatiPlot(self):
        self._wadatiPlt = PyQt4.QtGui.NewWindow()
        self._wadatiPlt.show()

    def _infoDlg(self):
        msg = """
                <h3><b>obspy.core.stream-Picker</b></h3>
                <br><br>
                <div>
                StreamPick is a lightweight seismological
                wave time picker for <code>obspy.core.Stream()</code>
                objects. It further utilises the <code>obspy.core.event</code>
                class to store picks in the QuakeML format.
                </div>
                <h4>Controls:</h4>
                <blockquote>
                <table>
                    <tr>
                        <td width=20><b>%s</b></td><td>Next station</td>
                    </tr>
                    <tr>
                        <td width=20><b>%s</b></td><td>Previous station</td>
                    </tr>
                    <tr>
                        <td width=20><b>%s</b></td><td>Toggle filter</td>
                    </tr>
                    <tr>
                        <td width=20><b>%s</b></td>
                        <td>Set P-Phase pick at mouse position</td>
                    </tr>
                    <tr>
                        <td width=20><b>%s</b></td>
                        <td>Set S-Phase pick at mouse position</td>
                    </tr>
                    <tr>
                        <td width=20><b>%s</b></td>
                        <td>Set custom phase pick at mouse position</td>
                    </tr>
                    <tr>
                        <td width=20><b>%s</b></td>
                        <td>Remove last pick in trace</td>
                    </tr>
                </table>
                </blockquote>
                <h4>Plot Controls:</h4>
                <blockquote>
                Use mouse wheel to zoom in- and out. Middle mouse button moves
                plot along x-axis.<br>
                Hit <b>Ctrl</b> to manipulate a single plot.
                <br>
                </blockquote>
                <div>
                Programm stores filter parameters in <code>.pick_filter</code>
                and a backup of recent picks in
                <code>.picks-obspy.xml.bak</code>.<br><br>
                See <a href=http://www.github.org/miili/StreamPick>
                http://www.github.org/miili/StreamPick</a> and
                <a href=http://www.obspy.org>http://www.obspy.org</a>
                for further documentation.
                </div>
                """ % (
            self._shortcuts['st_next'],
            self._shortcuts['st_previous'],
            self._shortcuts['filter_apply'],
            self._shortcuts['pick_p'],
            self._shortcuts['pick_s'],
            self._shortcuts['pick_custom'],
            self._shortcuts['pick_remove'],
        )
        PyQt4.QtGui.QMessageBox.about(self, 'About', msg)

    def _canvasDraw(self):
        '''
        Redraws the canvas and re-sets mouse focus
        '''
        for _i, _ax in enumerate(self.fig.get_axes()):
            _ax.set_xticklabels(_ax.get_xticks() *
                                self._current_st[_i].stats.delta)
        self.fig.canvas.draw()
        self.canvas.setFocus()

    def closeEvent(self, evnt):
        '''
        This function is called upon closing the PyQt4.QtGui
        '''
        # Save Picks
        # Save Catalog

    #        if len(self._picks) > 0:
    #            self._saveCatalog('.picks-obspy.xml.bak')
    #        if self.savefile is None and len(self._picks) > 0:
    #            ask = PyQt4.QtGui.QMessageBox.question(self, 'Save Picks?',
    #                'Do you want to save your picks?',
    #                PyQt4.QtGui.QMessageBox.Save |
    #                PyQt4.QtGui.QMessageBox.Discard |
    #                PyQt4.QtGui.QMessageBox.Cancel, PyQt4.QtGui.QMessageBox.Save)
    #            if ask == PyQt4.QtGui.QMessageBox.Save:
    #                self._saveCatalog()
    #            elif ask == PyQt4.QtGui.QMessageBox.Cancel:
    #                evnt.ignore()
    #        print self._picks


    # Filter Dialog
    class defFilter(PyQt4.QtGui.QDialog):
        def __init__(self, parent=None, filtervalues=None):
            '''
            Bandpass filter dialog... Qt layout and stuff
            '''
            PyQt4.QtGui.QDialog.__init__(self, parent)
            self.setWindowTitle('Create new Bandpass-Filter')

            # Frequency QDoubleSpinBoxes
            self.frqmin = PyQt4.QtGui.QDoubleSpinBox(decimals=2, maximum=100,
                                                     minimum=0.01, singleStep=0.1, value=0.1)
            self.frqmax = PyQt4.QtGui.QDoubleSpinBox(decimals=2, maximum=100,
                                                     minimum=0.01, singleStep=0.1, value=10.0)

            # Radio buttons for corners
            _corners = [2, 4, 8]
            _radbtn = []
            for _c in _corners:
                _radbtn.append(PyQt4.QtGui.QRadioButton(str(_c)))
                if _c == 4:
                    _radbtn[-1].setChecked(True)

            self.corner = PyQt4.QtGui.QButtonGroup()
            self.corner.setExclusive(True)

            radiogrp = PyQt4.QtGui.QHBoxLayout()
            for _i, _r in enumerate(_radbtn):
                self.corner.addButton(_r, _corners[_i])
                radiogrp.addWidget(_radbtn[_i])

            # Filter name
            self.fltname = PyQt4.QtGui.QLineEdit('Filter Name')
            self.fltname.selectAll()

            # Make Layout
            grid = PyQt4.QtGui.QGridLayout()
            grid.addWidget(PyQt4.QtGui.QLabel('Filter Name'), 0, 0)
            grid.addWidget(self.fltname, 0, 1)
            grid.addWidget(PyQt4.QtGui.QLabel('Min. Frequency'), 1, 0)
            grid.addWidget(self.frqmin, 1, 1)
            grid.addWidget(PyQt4.QtGui.QLabel('Max. Frequency'), 2, 0)
            grid.addWidget(self.frqmax, 2, 1)
            grid.addWidget(PyQt4.QtGui.QLabel('Corners'), 3, 0)
            grid.addLayout(radiogrp, 3, 1)
            grid.setVerticalSpacing(10)

            btnbox = PyQt4.QtGui.QDialogButtonBox(PyQt4.QtGui.QDialogButtonBox.Ok |
                                                  PyQt4.QtGui.QDialogButtonBox.Cancel)
            btnbox.accepted.connect(self.accept)
            btnbox.rejected.connect(self.reject)

            layout = PyQt4.QtGui.QVBoxLayout()
            layout.addWidget(PyQt4.QtGui.QLabel('Define a minimum and maximum' +
                                                ' frequency\nfor the bandpass filter.\nFunction utilises ' +
                                                'obspy.signal.filter (zerophase=True).\n'))
            layout.addLayout(grid)
            layout.addWidget(btnbox)

            if filtervalues is not None:
                self.fltname.setText(filtervalues['name'])
                self.frqmin.setValue(filtervalues['freqmin'])
                self.frqmax.setValue(filtervalues['freqmax'])
                self.corner.button(filtervalues['corners']).setChecked(True)

            self.setLayout(layout)
            self.setSizeGripEnabled(False)

        def getValues(self):
            '''
            Return filter dialogs values as a dictionary
            '''
            return dict(name=str(self.fltname.text()),
                        freqmin=float(self.frqmin.cleanText()),
                        freqmax=float(self.frqmax.cleanText()),
                        corners=int(int(self.corner.checkedId())))


def deleteWidget(wid):
    wid.setParent(None)


def clearLayout(layout):
    for i in reversed(range(layout.count())):
        layout.itemAt(i).widget().setParent(None)
