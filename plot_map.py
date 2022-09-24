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

        plt.savefig(output_filepath + '_3d.png', dpi=120)

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-dataset', dest='dataset', type=str, help='dataset dir', required=True)
    parser.add_argument('-output', dest='output', type=str, help='output dir', required=True)
    #parser.add_argument('-basefilename', dest='basefilename', type=str, help='dataset dir', required=True)
    params = parser.parse_args()

    dataset = params.dataset
    #basefilename = params.basefilename
    output = params.output

    #print(f"{dataset}/*.txt")
    #txt_files = glob.glob(f"{dataset}/*.txt")
    #print('txt_files',txt_files)

    for path in glob.glob(f"{dataset}/*.txt"):
        print('path',path)
        # check if current path is a file
        if os.path.isfile(path):
            #res.append(path)

            basefilename = os.path.splitext(os.path.basename(f"{dataset}{path}"))[0]
            #print('basefilename',basefilename)

            output_filepath = os.path.join(output, basefilename)
            filepath = os.path.join(dataset, basefilename)
            png_file = filepath + '.png'
            txt_file = filepath + '.txt'
            title = read_text_file(txt_file)
   
            img = cv2.imread(png_file, cv2.IMREAD_ANYDEPTH)

            p = PlotMap().plot_map(title, img)

            is_show = False
            if is_show:
                plt.show()
