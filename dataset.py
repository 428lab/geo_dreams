import numpy as np
import sys
import cv2
import os

import glob
import argparse
import tqdm
import pickle

class Dataset:
  def __init__(self, dataset_dir, output_dir):
    self.dataset = dataset_dir
    self.output = output_dir
    print(self.output)

    isExist = os.path.exists(self.output)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(self.output)

    filepath_kanji_ngrams_list = os.path.join(output_dir,'kanji_ngrams_list.npy')
    filepath_one_hot_kanji = os.path.join(output_dir,'one_hot_kanji.npy')
    filepath_terrains = os.path.join(output_dir,'terrains.npy')

    if os.path.isfile(filepath_kanji_ngrams_list) == False:
        print(f'Create {filepath_kanji_ngrams_list} file.')
        self.kanji_ngrams_list = self.create_kanji_ngrams_list(self.dataset)
        np.save(filepath_kanji_ngrams_list, self.kanji_ngrams_list)
    else:
        self.kanji_ngrams_list = np.load(filepath_kanji_ngrams_list)

    if os.path.isfile(filepath_one_hot_kanji) == False or os.path.isfile(filepath_terrains) == False:
        print(f'Create {filepath_one_hot_kanji} and {filepath_terrains} npy file.')
        self.one_hot_kanji, self.terrains = self.create_annotations(self.dataset, self.kanji_ngrams_list)
        np.save(filepath_one_hot_kanji, self.one_hot_kanji)
        np.save(filepath_terrains, self.terrains)
    else:
        self.one_hot_kanji = np.load(filepath_one_hot_kanji)
        self.terrains = np.load(filepath_terrains)
    #print('one_hot_kanji.shape',self.one_hot_kanji.shape)
    #print('terrains.shape',self.terrains.shape)

  def get_train_labels(self):
      return self.one_hot_kanji

  def get_train_terrains(self):
      return self.terrains

  def read_text_file(self, file_path):
      with open(file_path, 'r') as f:
          return f.read()

  def create_annotations(self, dataset, kanji_ngmras_list):
      one_hot_kanji = []
      terrains = []
      #cnt = 0
      for path in tqdm.tqdm(glob.glob(f"{dataset}/*.txt")):
          #if cnt > 10:
          #    break
          #cnt +=1
          # check if current path is a file
          if os.path.isfile(path):

              basefilename = os.path.splitext(os.path.basename(f"{dataset}{path}"))[0]
              #print('basefilename',basefilename)

              filepath = os.path.join(dataset, basefilename)
              pkl_file = filepath + '.pkl'


              with open(pkl_file, 'rb') as f:
                  terrain = pickle.load(f)


              txt_file = filepath + '.txt'
              record = self.read_text_file(txt_file).split(',')
              town_name_kanji = record[2]
              town_name_kana = record[5]

              # 町名を分解
              # 「四谷ラボ」なら
              # 四, 谷, ラ, ボ, 四谷, 谷ラ, ラボ, 四谷ラ, 谷ラボ, 四谷ラボ に分解
              #文字数n
              kanjis = []
              for n in range(len(town_name_kanji)):
                  #開始位置p
                  for p in range(len(town_name_kanji)):
                      if p+n+1 > len(town_name_kanji):
                          break
                      kanji = town_name_kanji[p:p+n+1]
                      if kanji not in kanjis:
                          kanjis.append(kanji)

              kanjis_one_hot = np.zeros(len(kanji_ngmras_list))
              indices = [i for i, x in enumerate(list(kanji_ngmras_list)) if x in kanjis]
              for idx in indices:
                  kanjis_one_hot[idx] = 1
              one_hot_kanji.append(kanjis_one_hot)
              terrains.append(terrain)
      return one_hot_kanji, terrains


  def create_kanji_ngrams_list(self, dataset):
      kanjis = []
      for path in tqdm.tqdm(glob.glob(f"{dataset}/*.txt")):
          # check if current path is a file
          if os.path.isfile(path):

              basefilename = os.path.splitext(os.path.basename(f"{dataset}{path}"))[0]
              #print('basefilename',basefilename)

              filepath = os.path.join(dataset, basefilename)
              pkl_file = filepath + '.pkl'
              txt_file = filepath + '.txt'
              record = self.read_text_file(txt_file).split(',')
              #print(record)
              town_name_kanji = record[2]
              town_name_kana = record[5]
              #print('#',town_name_kanji)

              # 町名を分解
              # 「四谷ラボ」なら
              # 四, 谷, ラ, ボ, 四谷, 谷ラ, ラボ, 四谷ラ, 谷ラボ, 四谷ラボ に分解
              #文字数n
              for n in range(len(town_name_kanji)):
                  #開始位置p
                  for p in range(len(town_name_kanji)):
                      if p+n+1 > len(town_name_kanji):
                          break
                      kanji = town_name_kanji[p:p+n+1]
                      if kanji not in kanjis:
                          #print(kanji)
                          kanjis.append(kanji)

      return kanjis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-dataset', dest='dataset', type=str, help='dataset dir', required=True)
    parser.add_argument('-output', dest='output', default='/ssd/geo_dream/models',type=str, help='output dir')
    params = parser.parse_args()

    dataset = params.dataset
    output = params.output

    dataset = Dataset(dataset, output)
    exit()


