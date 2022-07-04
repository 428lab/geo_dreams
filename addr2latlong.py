# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import os
import geocoder
from multiprocessing import Pool

import matplotlib.pyplot as plt
from math import log
from math import tan
from math import pi

CSV_FILENAME = 'zenkoku.csv'

parser = argparse.ArgumentParser(description='住所CSVファイルから緯度/経度情報を取得するプログラム')
parser.add_argument('--outfile', default='address_latlong.csv')
parser.add_argument('--dataset_dir', default='dataset')

def main():
    df_data = read_infile(CSV_FILENAME)
    latlongs =  get_latlongs(df_data)

    #with Pool(2) as p:
    #    p.map(save_terrain, latlongs)

    for latlong in latlongs:
        save_terrain(latlong)

    print('Done')


def save_terrain(latlong):

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)


    no = latlong[0]
    prefecture = latlong[1]
    city = latlong[2]
    town = latlong[3]
    address = prefecture + city + town

    prefecture_kana = latlong[4]
    city_kana = latlong[5]
    town_kana = latlong[6]
    address_kana = prefecture_kana + city_kana + town_kana

    ret = geocoder.osm(address, timeout=5.0)
    print(f'{no},{prefecture},{city},{town},{prefecture_kana},{city_kana},{town_kana},{ret.latlng}')
    if ret:
        zoom = 14
        x, y = latlon2tile(ret.latlng[1], ret.latlng[0], zoom)
        nabewari = (zoom, x, y) # タイル座標 (z, x, y)
        nabewari_tile = fetch_tile(*nabewari)
        print('nabewari_tile',nabewari_tile)
        plt.imshow(nabewari_tile)

        file_path = os.path.join(dataset_dir, str(no).zfill(7))
        plt.savefig(file_path + '.png')

        f = open(file_path + '.txt', 'w', encoding='UTF-8')
        f.write(f'{prefecture},{city},{town},{prefecture_kana},{city_kana},{town_kana},{ret.latlng[0]},{ret.latlng[1]}')
        f.close()
   
def fetch_tile(z, x, y):
    url = "https://cyberjapandata.gsi.go.jp/xyz/dem/{z}/{x}/{y}.txt".format(z=z, x=x, y=y)
    df = pd.read_csv(url, header=None).replace("e", 0.0)
    return df.values
 
def latlon2tile(lon, lat, z):
    x = int((lon / 180 + 1) * 2**z / 2) # x座標
    y = int(((-log(tan((45 + lat / 2) * pi / 180)) + pi) * 2**z / (2 * pi))) 
    return x, y

def read_infile(filename):
    usecols = ['都道府県','都道府県カナ','市区町村','市区町村カナ','町域','町域カナ']
    df_data = pd.read_csv(filename, usecols=usecols, encoding="cp932").dropna(how='any')
    df_data_unique = df_data[['都道府県','都道府県カナ','市区町村','市区町村カナ','町域','町域カナ']].drop_duplicates()
    return df_data_unique
    
def get_latlongs(df):
    latlongs = []
    cnt = 0
    map_colmun_and_index = {}
    for index, colmun in enumerate(df.columns, 0):
        map_colmun_and_index[colmun] = index

    for row in df.values:
        no = cnt
        prefecture = row[map_colmun_and_index['都道府県']]
        city = row[map_colmun_and_index['市区町村']]
        town = row[map_colmun_and_index['町域']]
        address = prefecture + city + town

        prefecture_kana = row[map_colmun_and_index['都道府県カナ']]
        city_kana = row[map_colmun_and_index['市区町村カナ']]
        town_kana = row[map_colmun_and_index['町域カナ']]
        address_kana = prefecture_kana + city_kana + town_kana

        latlongs.append([no,prefecture,city,town,prefecture_kana,city_kana,town_kana])
        cnt += 1

        #ret = geocoder.osm(address, timeout=5.0)
        #print(f'{prefecture},{city},{town},{prefecture_kana},{city_kana},{town_kana},{ret.latlng}')
        #if ret:
        #    #print('lat, lng',ret.latlng[0],ret.latlng[1])
        #    df['lat'] = ret.latlng[0]
        #    df['lng'] = ret.latlng[1]

    return latlongs

if __name__ == "__main__":
    args = parser.parse_args()
    outfile = args.outfile
    dataset_dir = args.dataset_dir
    main()
