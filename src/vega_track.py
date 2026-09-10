#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D
import os
import csv
import glob
from   datetime import datetime
import json
import copy
from scipy.interpolate import interp1d
from matplotlib import gridspec
import json
import math

from sksurgerynditracker.nditracker import NDITracker
from Transformation_fun import *




def main():
    settings = {
        "tracker type": "vega",
        "ip address": "169.254.7.143",   # <-- set your Vega IP here
        "port": 8765,                    # default Vega control port
        "romfiles": [
            "/home/ayoob/Polaris_Vega_VT/marker_definition/Kia_phantom_marker.rom",      # <-- absolute paths
            # "/full/path/to/8700340.rom",
        ],
        "use quaternions": True,       # uncomment to get [qw,qx,qy,qz,x,y,z]
        # "verbose": True,
    }

    tracker = NDITracker(settings)
    try:
        tracker.start_tracking()
        print("Tools:", tracker.get_tool_descriptions())

        for _ in range(20):
            ports, timestamps, frames, tracking, quality = tracker.get_frame()
            # tracking is a list of 4x4 poses (numpy arrays) unless 'use quaternions' is True
            print("ports:", ports, "frame:", frames, "quality:", quality)
            # Example: print first tool pose matrix
            if tracking:
                print("+++++++++++++++++++++++++++++++++++++++++")
                print(tracking[0])
                # pose_refined = tranMat_to_pose(np.array(tracking[0]))
                # print(pose_refined)
            time.sleep(0.1)




    finally:
        tracker.stop_tracking()
        tracker.close()

if __name__ == "__main__":
    main()
