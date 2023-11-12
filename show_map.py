import matplotlib.pyplot as plt
#plt.rcParams['font.family'] = 'IPAexGothic'
from matplotlib import animation
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TKAgg')


import sys
import pandas as pd
from scipy.interpolate import griddata
import argparse
#import terrain_csv
import math
import cv2
import os

#print( plt.rcParams['font.family'] )

from matplotlib.font_manager import FontProperties



font_path = '/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf'
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

import glob
import pickle

#from pympler.tracker import SummaryTracker
#tracker = SummaryTracker()


def read_csv(file_name):
    df = pd.read_csv(file_name)
    return df

class PlotMap:
    def __init__(self):
        self.fig = plt.figure(figsize=(6*2,4*2))
        self.ax0 = self.fig.add_subplot(121, projection='3d')
        self.ax1 = self.fig.add_subplot(122, aspect='equal')

    def plot_map(self, title, img):
        self.fig.suptitle(title)


        rows,cols = img.shape
        x = np.array(range(0, cols))
        y = np.array(range(0, rows))
        z = img.copy()

        X, Y = np.meshgrid(x, y)

        self.ax0.plot_surface(X, Y, z, cmap=cm.gist_earth, linewidth=0, antialiased=False)
        self.ax1.contourf(X, Y, z, cmap=cm.gist_earth)

        #plt.savefig(output_filepath, dpi=120)

    def close(self):
        self.fig.clear()
        plt.close()
        plt.cla()
        plt.clf()
        
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-dataset', default='/data/geo/dataset', type=str, help='dataset dir')
    parser.add_argument('-filename', default='0010659', type=str, help='output dir')
    params = parser.parse_args()

    dataset = params.dataset
    filename = params.filename


    #basefilename = os.path.splitext(os.path.basename(f"{dataset}{filename}"))[0]
    #print('basefilename',basefilename)

    filepath = os.path.join(dataset, filename)
    pkl_file = filepath + '.pkl'
    txt_file = filepath + '.txt'
    title = read_text_file(txt_file)
   
    with open(pkl_file, 'rb') as f:
        img = pickle.load(f) 

    plot = PlotMap()
    plot.plot_map(title, img)

    plt.show()

    plot.close()

