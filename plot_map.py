import matplotlib.pyplot as plt
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

def read_csv(file_name):
    df = pd.read_csv(file_name)
    return df

class PlotMap:
    def __init__(self):
        self.fig = plt.figure(figsize=(6*2,4*2))
        self.ax0 = self.fig.add_subplot(121, projection='3d')
        self.ax1 = self.fig.add_subplot(122, aspect='equal')

    def rotate_3d_surface(self, angle):
        self.ax0.view_init(azim=angle)
        self.ax1.view_init(azim=angle)

    def plot_contour(self, ax, x, y, z, cmap=cm.gist_earth):
        ax.contourf(x, y, z, cmap=cmap)

    def plot_3d_surface(self, ax, title, x, y, z, z_max, z_min, cmap=cm.gist_earth):

        surf = ax.plot_surface(x, y, z, cmap=cmap, linewidth=0, antialiased=False)
        ax.set_xlim(np.nanmin(x), np.nanmax(x))
        ax.set_ylim(np.nanmin(y), np.nanmax(y))
        ax.set_zlim(z_min, z_max)
        ax.set_title(title)
        ax.zaxis.set_major_locator(LinearLocator(6))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    def plot_3d_trisurf(self, ax, title, x, y, z, z_max, z_min, cmap=cm.gist_earth):
        surf = ax.plot_trisurf(list(x), list(y), list(z), cmap=cmap, linewidth=0, antialiased=False)
        ax.set_xlim(np.nanmin(x), np.nanmax(x))
        ax.set_ylim(np.nanmin(y), np.nanmax(y))
        ax.set_zlim(z_min, z_max)
        ax.set_title(title)
        ax.zaxis.set_major_locator(LinearLocator(6))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    def plot_map(self, title, img, is_show = False, animated = False):
        rows,cols = img.shape
        x = np.array(range(0, cols))
        y = np.array(range(0, rows))
        z = img.copy()

        X, Y = np.meshgrid(x, y)

        self.ax0.plot_surface(X, Y, z, cmap=cm.gist_earth, linewidth=0, antialiased=False)
        self.ax1.contourf(X, Y, z, cmap=cm.gist_earth)

        if animated:
            rot_animation = animation.FuncAnimation(self.fig, self.rotate_3d_surface, frames=np.arange(0,362,2),interval=100)
            rot_animation.save('soil.gif', dpi=80, writer='imagemagick')
        #print('Save fig. as', title + '.png')
        plt.savefig(title + '.png', dpi=120)

        if is_show:
            plt.show()
        return


    def save_as_file(self, title):
        plt.savefig(title + '.jpg', dpi=120)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-png_file', dest='png_file', type=str, help='png file path', required=True)
    params = parser.parse_args()

    png_file = params.png_file
    img = cv2.imread(png_file, cv2.IMREAD_ANYDEPTH)

    p = PlotMap().plot_map('Test_PlotMap', img, is_show = True, animated = False)

