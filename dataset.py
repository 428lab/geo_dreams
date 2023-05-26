import numpy as np
import sys
import cv2
import os

import glob
import argparse
import tqdm
import pickle


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()


def create_annotations(dataset, kanji_ngmras_list):    
    annotations = []
    for path in tqdm.tqdm(glob.glob(f"{dataset}/*.txt")):
        # check if current path is a file
        if os.path.isfile(path):

            basefilename = os.path.splitext(os.path.basename(f"{dataset}{path}"))[0]
            #print('basefilename',basefilename)

            filepath = os.path.join(dataset, basefilename)
            pkl_file = filepath + '.pkl'


            with open(pkl_file, 'rb') as f:
                img = pickle.load(f)


            txt_file = filepath + '.txt'
            record = read_text_file(txt_file).split(',')
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

            #print(kanjis)
            #print('len(kanji_ngmras_list)',len(kanji_ngmras_list))
            kanjis_one_hot = np.zeros(len(kanji_ngmras_list))
            indices = [i for i, x in enumerate(list(kanji_ngmras_list)) if x in kanjis]
            for idx in indices:
                kanjis_one_hot[idx] = 1
            #print([i for i, x in enumerate(list(kanji_ngmras_list)) if x in kanjis])
            #print(list(kanji_ngmras_list).index(kanjis))
            annotations.append([img,kanjis_one_hot])
    return annotations


def create_kanji_ngrams_list(dataset):    
    annotations = []
    kanjis = []
    for path in tqdm.tqdm(glob.glob(f"{dataset}/*.txt")):
        #print('path',path)
        # check if current path is a file
        if os.path.isfile(path):

            basefilename = os.path.splitext(os.path.basename(f"{dataset}{path}"))[0]
            #print('basefilename',basefilename)

            filepath = os.path.join(dataset, basefilename)
            pkl_file = filepath + '.pkl'
            txt_file = filepath + '.txt'
            record = read_text_file(txt_file).split(',')
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
    parser.add_argument('-output', dest='output', type=str, help='output dir')
    params = parser.parse_args()

    dataset = params.dataset
    output = params.output

    if os.path.isfile('kanji_ngrams_list.npy') == False:
        print('Create kanji ngrams npy file.')
        kanji_ngrams_list = create_kanji_ngrams_list(dataset)
        np.save('kanji_ngrams_list.npy', kanji_ngrams_list)
    else:
        kanji_ngrams_list = np.load('kanji_ngrams_list.npy')

    if os.path.isfile('annotations.npy') == False:
        print('Create annotations npy file.')
        annotations = create_annotations(dataset, kanji_ngrams_list)
        np.save('annotations.npy', annotations)
    else:
        annotations = np.load('annotations.npy')
    #print(kanji_ngrams_list)





   

