import time
import pickle

import pandas as pd
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

from utils import save_obj, split_data
from utils_w2v import load_model_w2v, word2vec

CATEGORY_MODEL_NAME = 'chuyen_muc'
MODELS = {}

PATH_MODEL_W2V = './model/baomoi.vn.model.bin'
print('Loading Word2Vec model...')
MODEL_W2V = load_model_w2v(PATH_MODEL_W2V, bin=True)

def model(X_train, y_train):
    clf = OneVsRestClassifier(SVC(kernel='linear', C=1, probability=True, verbose=True))
    clf.fit(X_train, y_train)
    return clf


def training(data, save=None):
    print(f"Transforming {CATEGORY_MODEL_NAME} vectorizer")
    transformed = word2vec(data['clean_data'].values.astype('U').tolist(), MODEL_W2V, dims=300)

    print(f"Training {CATEGORY_MODEL_NAME} model")
    MODELS[CATEGORY_MODEL_NAME] = model(transformed, data['category'].values.astype('U'))

    if save is not None:
        save_obj(MODELS, save)
    return MODELS


def main():
    print("Reading data...")
    data = pd.read_csv('./data/video_data_clean.csv')
    print("Shape:", data.shape)

    data_x = pd.DataFrame({'id': [], 'category': [], 'clean_data': []})
    for g in data.groupby('category'):
        data_x = pd.concat([data_x, g[1][:1200]])
    data = data_x

    exp_id = str(int(time.time()))
    train_df, test_df = split_data(data, file=True, exp_id=exp_id)

    training(train_df, save=f'./model/models_video_{exp_id}.pkl')
    print(f'Exp id {exp_id}')

if __name__ == "__main__":
    main()
    pass