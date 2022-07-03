# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import os
import geocoder

CSV_FILENAME = 'zenkoku.csv'

parser = argparse.ArgumentParser(description='住所CSVファイルから緯度/経度情報を取得するプログラム')
parser.add_argument('--outfile', default='address_latlong.csv')

def main(args):
    outfile = args.outfile
    df_data = read_infile(CSV_FILENAME)
    df_address_latlong =  get_latlong(df_data)
    #print(df_address_latlong)
    df_address_latlong.to_csv(outfile)
    print('Done')

def read_infile(filename):
    usecols = ['都道府県','都道府県カナ','市区町村','市区町村カナ','町域','町域カナ']
    df_data = pd.read_csv(filename, usecols=usecols, encoding="cp932").dropna(how='any')
    df_data_unique = df_data[['都道府県','都道府県カナ','市区町村','市区町村カナ','町域','町域カナ']].drop_duplicates()
    return df_data_unique
    
def get_latlong(df):
    map_colmun_and_index = {}
    for index, colmun in enumerate(df.columns, 0):
        map_colmun_and_index[colmun] = index

    for row in df.values:
        prefecture = row[map_colmun_and_index['都道府県']]
        city = row[map_colmun_and_index['市区町村']]
        town = row[map_colmun_and_index['町域']]
        address = prefecture + city + town

        prefecture_kana = row[map_colmun_and_index['都道府県カナ']]
        city_kana = row[map_colmun_and_index['市区町村カナ']]
        town_kana = row[map_colmun_and_index['町域カナ']]
        address_kana = prefecture_kana + city_kana + town_kana

        ret = geocoder.osm(address, timeout=5.0)
        print(f'{prefecture},{city},{town},{prefecture_kana},{city_kana},{town_kana},{ret.latlng}')
        if ret:
            #print('lat, lng',ret.latlng[0],ret.latlng[1])
            df['lat'] = ret.latlng[0]
            df['lng'] = ret.latlng[1]

    return df

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
