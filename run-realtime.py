"""
This is a example script to use the MP model on raw data stream. It connects
to a Mode-S Beast like raw stream.

Usage:
python run-realtime.py --server SERVER_ADDR_OR_IP --port PORT_NUMBER
"""

import os
import matplotlib.pyplot as plt
import time
import threading
import pyModeS as pms
import argparse
from lib import client
import mp_vis
from stream import Stream

parser = argparse.ArgumentParser()
parser.add_argument('--server', help='server address or IP', required=True)
parser.add_argument('--port', help='Raw beast port', required=True)
args = parser.parse_args()
server = args.server
port = int(args.port)

ADSB_TS = []
ADSB_MSGS = []
EHS_TS = []
EHS_MSGS = []

TNOW = 0
TFLAG = 0

STREAM = Stream(lat0=51.99, lon0=4.37)
STREAM.mp.N_AC_PTCS = 500
STREAM.mp.AGING_SIGMA = 180

DATA_LOCK = threading.Lock()

root = os.path.dirname(os.path.realpath(__file__))

class PwmBeastClient(client.BaseClient):
    def __init__(self, host, port):
        super(PwmBeastClient, self).__init__(host, port)

    def handle_messages(self, messages):
        global ADSB_TS
        global ADSB_MSGS
        global EHS_TS
        global EHS_MSGS
        global TNOW
        global TFLAG

        for msg, ts in messages:
            if len(msg) != 28:           # wrong data length
                continue

            df = pms.df(msg)

            if df == 17:
                ADSB_TS.append(ts)
                ADSB_MSGS.append(msg)

            if df == 20 or df == 21:
                EHS_TS.append(ts)
                EHS_MSGS.append(msg)


def gen_plot():
    global ADSB_TS
    global ADSB_MSGS
    global EHS_TS
    global EHS_MSGS
    global TNOW
    global TFLAG

    while True:
        if TFLAG >= 15:
            print("updating plot...")


            DATA_LOCK.acquire()
            data = STREAM.mp.construct()
            DATA_LOCK.release()

            tstr = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(TNOW))

            plt.figure(figsize=(12, 9))
            mp_vis.plot_all_level_wind(STREAM.mp, data=data, return_plot=True, landscape_view=True, barbs=False)
            plt.suptitle('Current wind field (UTC %s)' % tstr)
            plt.savefig(root+'/data/screenshots/nowfield_wind.png')

            plt.figure(figsize=(12, 9))
            mp_vis.plot_all_level_temp(STREAM.mp, data=data, return_plot=True, landscape_view=True)
            plt.suptitle('Current temperature field (UTC %s)' % tstr)
            plt.savefig(root+'/data/screenshots/nowfield_temp.png')

            TFLAG = 0
        time.sleep(0.2)


def update_mp():
    global ADSB_TS
    global ADSB_MSGS
    global EHS_TS
    global EHS_MSGS
    global TNOW
    global TFLAG

    while True:
        if len(EHS_TS) == 0:
            time.sleep(0.1)
            continue

        t = EHS_TS[-1]             # last timestamp

        if t - TNOW > 1:
            TNOW = int(t)
            TFLAG += 1
            STREAM.process_raw(ADSB_TS, ADSB_MSGS, EHS_TS, EHS_MSGS)
            STREAM.compute_current_weather()
            DATA_LOCK.acquire()
            STREAM.update_mp_model()
            DATA_LOCK.release()

            ADSB_TS = []
            ADSB_MSGS = []
            EHS_TS = []
            EHS_MSGS = []

            print("time: %d | n_ptc: %d"  % (TNOW, len(STREAM.mp.PTC_X)))
        else:
            time.sleep(0.1)


thread_gen_plot = threading.Thread(target=gen_plot)

client = PwmBeastClient(host=server, port=port)
thread_client = threading.Thread(target=client.run)

thread_mp = threading.Thread(target=update_mp)

thread_client.start()
thread_mp.start()
thread_gen_plot.start()
