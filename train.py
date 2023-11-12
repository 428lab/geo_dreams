import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.callbacks import TensorBoard
import cv2

from dataset import Dataset
import argparse
import os

import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np


def train(train_images, train_labels, ngrams_file):

    # MNISTデータセットの読み込み
    #(train_images, train_labels), (_, _) = mnist.load_data()


    #labels = np.array(labels)
    one_hot_num = train_labels.shape[1]
    print('one_hot_num',one_hot_num)
    print('train_labels.shape1',train_labels.shape)
    train_labels = train_labels.reshape(-1, one_hot_num)
    print('train_labels.shape2',train_labels.shape)

    kanji_ngrams_list = np.load(ngrams_file)




    # train_imagesからランダムに4つの画像を選択
    num_images = 4
    random_indices = np.random.choice(len(train_images), num_images, replace=False)
    sample_images = train_images[random_indices]
    sample_onehot = train_labels[random_indices]
    print('sample_onehot',sample_onehot)
    #sample_labels = kanji_ngrams_list[sample_onehot]

    # サンプル画像の表示
    plt.figure(figsize=(10, 10))

    # データの正規化
    train_images = train_images / 3773.0

    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        print('onehot',sample_onehot[i])

        # ワンホットエンコーディングされた配列の中で1の値を持つインデックスを取得
        onehot_indices = np.where(sample_onehot[i] == 1)[0]
        # それらのインデックスに対応する要素をkanji_ngrams_listから取得
        sample_labels = [kanji_ngrams_list[idx] for idx in onehot_indices]
        print('title',sample_labels)

        plt.title(sample_labels[len(sample_labels)-1])
        plt.imshow(sample_images[i].reshape(256, 256), cmap='gray')
        plt.axis('off')
    plt.show()

    # 入力データの次元を変更
    print('train_images.shape1',train_images.shape)
    train_images = train_images.reshape(-1, 256*256)
    print('train_images.shape2',train_images.shape)

    # 生成モデルの構築
    n_hidden=200
    model = Sequential()
    model.add(Dense(n_hidden, activation='relu', input_shape=(one_hot_num,)))
    model.add(Dense(n_hidden,activation='relu'))
    model.add(Dense(n_hidden,activation='relu'))
    model.add(Dense(n_hidden,activation='relu'))
    model.add(Dense(n_hidden,activation='relu'))
    model.add(Dense(256*256, activation='sigmoid'))

    # モデルのコンパイル
    model.compile(optimizer='adam', loss='binary_crossentropy')


    # TensorBoardのログディレクトリを設定
    log_dir = os.path.join(output_dir, "logs")

    tensorboard_callback = TensorBoard(log_dir=log_dir,
            histogram_freq=1,
            write_grads=True,
            write_images=1,
            embeddings_freq=1
            )

    # モデルの訓練
    model.fit(x=train_labels, y=train_images, epochs=100, batch_size=32, callbacks=[tensorboard_callback])  # コールバックを追加

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-dataset', dest='dataset', type=str, help='dataset dir', required=True)
    parser.add_argument('-output', dest='output', default='/ssd/geo_dream/models',type=str, help='output dir')
    parser.add_argument('-ngrams-file', dest='ngrams_file', default='/ssd/geo_dream/models/kanji_ngrams_list.npy',type=str, help='path for kanji_ngrams_list.npy')
    params = parser.parse_args()

    dataset_dir = params.dataset
    output_dir = params.output
    ngrams_file = params.ngrams_file

    dataset = Dataset(dataset_dir, output_dir)

    train_labels = dataset.get_train_labels()
    train_terrains = dataset.get_train_terrains()

    model = train(train_terrains, train_labels, ngrams_file)

    model_filepath = os.path.join(output_dir, "geo_model.hdf5")
    model.save(model_filepath)

    exit()

