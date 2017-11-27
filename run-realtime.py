import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pyModeS as pms
import argparse
from lib import client, aero
from stream import Stream

class PwmBeastClient(client.BaseClient):
    def __init__(self, host, port):
        super(PwmBeastClient, self).__init__(host, port)
        self.steam = Stream(pwm_ptc=500, pwm_decay=60)
        self.tnow = 0
        self.tflag = 0
        self.reset_buffer()

    def reset_buffer(self):
        self.adsb_ts = []
        self.adsb_msgs = []
        self.ehs_ts = []
        self.ehs_msgs = []

    def handle_messages(self, messages):
        for msg, ts in messages:
            if len(msg) != 28:           # wrong data length
                continue

            df = pms.df(msg)

            if df == 17:
                self.adsb_ts.append(ts)
                self.adsb_msgs.append(msg)

            if df == 20 or df == 21:
                self.ehs_ts.append(ts)
                self.ehs_msgs.append(msg)

        self.steam.process_raw(self.adsb_ts, self.adsb_msgs,
                               self.ehs_ts, self.ehs_msgs)

        self.reset_buffer()

        t = messages[-1][1]             # last timestamp
        if t - self.tnow > 1:
            self.tnow = int(t)
            self.tflag += 1
            self.steam.update_wind_model()
            print self.tnow, "| wind model updated | particles:", len(self.steam.pwm.PTC_X)

        if self.tflag == 30:
            plt.figure(figsize=(12, 9))
            self.steam.compute_current_wind()
            self.steam.pwm.plot_all_level(landscape_view=True)
            plt.show()
            self.tflag = 0

parser = argparse.ArgumentParser()
parser.add_argument('--server', help='server address or IP', required=True)
parser.add_argument('--port', help='Raw beast port', required=True)
args = parser.parse_args()

server = args.server
port = int(args.port)

rtpwm = PwmBeastClient(host=server, port=port)
rtpwm.run()
