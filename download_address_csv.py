# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import os
import urllib.error
import urllib.request
import shutil

ZIP_FILENAME = 'csv_zenkoku.zip'
OUT_FILENAME = 'map.csv'
CSV_FILENAME = 'zenkoku.csv'
URL = 'http://jusyo.jp/downloads/new/csv/' + ZIP_FILENAME

parser = argparse.ArgumentParser(description='住所データZIPファイルをダウンロードして、UNZIPするプログラム')

def main(args):
    download_and_unzip_file(URL, '.')
    df_data = read_infile(CSV_FILENAME)
    df_data.to_csv(OUT_FILENAME)

def read_infile(filename):
    usecols = ['都道府県','都道府県カナ','市区町村','市区町村カナ','町域','町域カナ']
    df_data = pd.read_csv(filename, usecols=usecols, encoding="cp932").dropna(how='any')
    df_data_unique = df_data[['都道府県','都道府県カナ','市区町村','市区町村カナ','町域','町域カナ']].drop_duplicates()
    return df_data_unique

def download_and_unzip_file(url, dst_path):
    is_file = os.path.isfile(CSV_FILENAME)
    if is_file:
        print(f'{CSV_FILENAME} exists.')
    else:
        print(f'Downloading {ZIP_FILENAME} ...')
        download_file_to_dir(url, dst_path)
        print(f'Unziping {ZIP_FILENAME} ...')
        shutil.unpack_archive(ZIP_FILENAME, dst_path)

def download_file(url, dst_path):
    try:
        with urllib.request.urlopen(url) as web_file:
            data = web_file.read()
            with open(dst_path, mode='wb') as local_file:
                #local_file.write(data.decode(encoding='shift-jis'))
                local_file.write(data)
    except urllib.error.URLError as e:
        print(e)

def download_file_to_dir(url, dst_dir):
    download_file(url, os.path.join(dst_dir, os.path.basename(url)))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
