import numpy as np
import sys
import cv2
import os

import glob
import argparse
import tqdm
import pickle


import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TKAgg')


import math

from matplotlib.font_manager import FontProperties



class PlotStats:
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


class Stats:
  def __init__(self, ngrams_file, onehot_file, output_dir):
    self.fig = plt.figure(figsize=(6*2,4*2))
    self.ax0 = self.fig.add_subplot(121, projection='3d')
    self.ax1 = self.fig.add_subplot(122, aspect='equal')

    self.ngrams_file = ngrams_file
    self.onehot_file = onehot_file
    self.output = output_dir
    print(self.ngrams_file)
    print(self.onehot_file)
    print(self.output)

    isExist = os.path.exists(self.output)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(self.output)

    #filepath_kanji_ngrams_list = os.path.join(output_dir,'kanji_ngrams_list.npy')

    self.kanji_ngrams_list = np.load(self.ngrams_file)
    self.one_hot_kanji = np.load(self.onehot_file)


  def show_ngrams_file(self):
      #print('kanji_ngrams_list',self.kanji_ngrams_list.shape, len(self.kanji_ngrams_list), self.kanji_ngrams_list)
      #print('one_hot_kanji',self.one_hot_kanji.shape, len(self.one_hot_kanji), self.one_hot_kanji)
      total_kanji_num = self.one_hot_kanji.sum(axis=0)
      #print('total_kanji_num',total_kanji_num)
      print('total_kanji_num.shape',total_kanji_num.shape)

      statu_dict = dict(zip(self.kanji_ngrams_list,total_kanji_num))
      #print('statu_dict',statu_dict)
      statu_dict_sorted = sorted(statu_dict.items(), key=lambda x:x[1], reverse=True)
      print('statu_dict_sorted',statu_dict_sorted)


  def plot(self, ):
      left = np.array([1, 2, 3, 4, 5])
      height = np.array([100, 200, 300, 400, 500])
      plt.bar(left, height)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    #parser.add_argument('-dataset', dest='dataset', type=str, help='dataset dir', required=True)
    parser.add_argument('-ngrams-file', dest='ngrams_file', default='/ssd/geo_dream/models/kanji_ngrams_list.npy',type=str, help='path for kanji_ngrams_list.npy')
    parser.add_argument('-one-hot-file', dest='onehot_file', default='/ssd/geo_dream/models/one_hot_kanji.npy',type=str, help='path for one_hot_kanji.npy')
    parser.add_argument('-output', dest='output', default='/ssd/geo_dream/stats',type=str, help='stats output dir')
    params = parser.parse_args()

    #dataset = params.dataset
    ngrams_file = params.ngrams_file
    onehot_file = params.onehot_file
    output = params.output

    stats = Stats(ngrams_file, onehot_file, output)
    stats.show_ngrams_file()
    exit()


