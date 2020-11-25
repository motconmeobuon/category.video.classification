import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from utils import vectorizer, load_obj
from utils_w2v import load_model_w2v, word2vec
from preprocessing import preparing_data


id = 1603937330
# PATH_DATA_TEST = f'./data/test_{id}.csv'
PATH_DATA_TEST = './data/video_test_pred_1000_1000.csv'
PATH_MODEL_W2V = './model/baomoi.vn.model.bin'
PATH_MODELS = f'./model/models_video_{id}.pkl'

if __name__ == "__main__":
    print('Loading data test...')
    df_test = pd.read_csv(PATH_DATA_TEST)

    df_test = df_test.drop('category_pred', axis=1)
    df_test['clean_data'] = df_test.apply(lambda x: preparing_data(x['title']), axis=1)

    print('Loading Word2Vec model...')
    model_w2v = load_model_w2v(PATH_MODEL_W2V, bin=True)

    print('Loading models...')
    models = load_obj(PATH_MODELS)

    transformed_test = word2vec(df_test['clean_data'].values.astype('U').tolist(), model_w2v, dims=300)

    x_ = models['chuyen_muc'].predict(transformed_test)
    df_test['category_pred'] = x_

    print(classification_report(df_test['category'].values, df_test['category_pred'].values))

    labels = []
    for g in df_test.groupby('category'):
        labels.append(g[0])
    labels.sort()
    cm = confusion_matrix(df_test['category'].values, df_test['category_pred'].values)
    my_fig, (ax, my_cbar_ax) = plt.subplots(ncols=2, figsize=(28, 24),
                                            gridspec_kw={'width_ratios': [30, 1]})
    sns.heatmap(cm, annot=True, ax = ax, cbar_ax = my_cbar_ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion Matrix Video')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    # plt.savefig(f'./image/test_{id}.png')
    plt.savefig(f'./image/video_test_pred_w2v.png')
    # df_test = df_test.drop('clean_data', axis=1)
    # df_test.to_csv('./data/test/test_800_end_pred.csv', index=False)