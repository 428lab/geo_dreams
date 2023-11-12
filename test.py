import numpy as np
import sys
import cv2
import os

import glob
import argparse
import tqdm
import pickle

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Reshape
import keras

from plot_map import PlotMap
import matplotlib.pyplot as plt

class Test:
  def __init__(self, model_file, ngrams_file, output_dir):
    self.model_file = model_file
    self.ngrams_file = ngrams_file
    self.output = output_dir
    print(self.model_file)
    print(self.output)

    isExist = os.path.exists(self.output)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(self.output)



    self.model = load_model(model_file)
    self.kanji_ngrams_list = np.load(self.ngrams_file)


  def predict(self, place_name):
    print(place_name)

    # 町名を分解
    # 「四谷ラボ」なら
    # 四, 谷, ラ, ボ, 四谷, 谷ラ, ラボ, 四谷ラ, 谷ラボ, 四谷ラボ に分解
    #文字数n
    kanjis = []
    for n in range(len(place_name)):
        #開始位置p
        for p in range(len(place_name)):
            if p+n+1 > len(place_name):
                break
            kanji = place_name[p:p+n+1]
            if kanji not in kanjis:
                kanjis.append(kanji)

    kanjis_one_hot = np.zeros(len(self.kanji_ngrams_list))
    indices = [i for i, x in enumerate(list(self.kanji_ngrams_list)) if x in kanjis]
    for idx in indices:
        kanjis_one_hot[idx] = 1
    print(np.array([kanjis_one_hot]))
    input_number = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    print(input_number)
    terrain = self.model.predict(np.array([kanjis_one_hot]))
    print('terrain',terrain.shape, terrain)


    plot = PlotMap()
    plot.plot_map(place_name, terrain.reshape(256,256))
    plt.show()


    #cv2.imshow('terrain',terrain.reshape((256,256)))



    return 

  def get_train_terrains(self):
      return self.terrains

  def read_text_file(self, file_path):
      with open(file_path, 'r') as f:
          return f.read()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    #parser.add_argument('-place-name', dest='place_name', type=str, help='Place name in Japanese', required=True)
    parser.add_argument('-ngrams-file', dest='ngrams_file', default='/ssd/geo_dream/models/kanji_ngrams_list.npy',type=str, help='path for kanji_ngrams_list.npy')
    parser.add_argument('-model', dest='model', default='/ssd/geo_dream/models/geo_model.hdf5',type=str, help='Path to model file')
    parser.add_argument('-output', dest='output', default='/ssd/geo_dream/test',type=str, help='output dir')
    params = parser.parse_args()

    #place_name = params.place_name
    ngrams_file = params.ngrams_file
    model_file = params.model
    output = params.output

    test = Test(model_file, ngrams_file, output)
    test.predict('平')
    test.predict('崎')
    test.predict('竜')
    test.predict('山')
    test.predict('川')
    cv2.waitKey(0)
    exit()


