import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import pyModeS as pms
import argparse
from lib import client, aero
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

STREAM = Stream(pwm_ptc=300, pwm_decay=30)
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
            plt.figure(figsize=(12, 9))

            DATA_LOCK.acquire()
            STREAM.pwm.plot_all_level(return_plot=True, landscape_view=True)
            DATA_LOCK.release()

            tstr = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(TNOW))
            plt.suptitle('Current wind field (UTC %s)' % tstr)
            plt.savefig(root+'/data/screenshots/nowfield.png')
            TFLAG = 0
        time.sleep(0.2)


def update_pwm():
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
            STREAM.compute_current_wind()
            DATA_LOCK.acquire()
            STREAM.update_wind_model()
            DATA_LOCK.release()

            ADSB_TS = []
            ADSB_MSGS = []
            EHS_TS = []
            EHS_MSGS = []

            print(TNOW, "| wind model updated | particles:", len(STREAM.pwm.PTC_X))
        else:
            time.sleep(0.1)


thread_gen_plot = threading.Thread(target=gen_plot)

client = PwmBeastClient(host=server, port=port)
thread_client = threading.Thread(target=client.run)

thread_pwm = threading.Thread(target=update_pwm)

thread_client.start()
thread_pwm.start()
thread_gen_plot.start()
