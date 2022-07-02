# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import os
import geocoder


parser = argparse.ArgumentParser(description='住所CSVファイルから緯度/経度情報を取得するプログラム')
parser.add_argument('infile', help='入力住所CSVファイル')

def main(args):

    infile = args.infile
    df_data = read_infile(infile)
    print('df_data',df_data)
    get_latlong(df_data)
    #print(f'Writing {outfile} ...')
    #df_data_unique.to_csv(outfile)
    print('Done')
    
def read_infile(filename):
    usecols = ['都道府県','都道府県カナ','市区町村','市区町村カナ','町域','町域カナ']
    return pd.read_csv(filename, usecols=usecols)

def get_latlong(df):
    # preparation
    map_colmun_and_index = {}
    for index, colmun in enumerate(df.columns, 0):
        map_colmun_and_index[colmun] = index

    for row in df.values:
        prefecture = row[map_colmun_and_index['都道府県']]
        city = row[map_colmun_and_index['市区町村']]
        town = row[map_colmun_and_index['町域']]

        prefecture_kana = row[map_colmun_and_index['都道府県カナ']]
        city_kana = row[map_colmun_and_index['市区町村カナ']]
        town_kana = row[map_colmun_and_index['町域カナ']]

        ret = geocoder.osm(prefecture + city + town, timeout=5.0)
        print(f'{prefecture},{city},{town},{prefecture_kana},{city_kana},{town_kana},{ret.latlng}')



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
