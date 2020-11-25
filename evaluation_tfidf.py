# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import joblib
import random
import collections
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, plot_confusion_matrix
import scikitplot as skplt
from utils import vectorizer, load_obj
from preprocessing import preparing_data

id = 1605060691
PATH_DATA_TEST = f'./data/test_{id}.csv'
PATH_DATA_TEST = './data/test/video_test_1106.csv'
PATH_MODELS = f'./model/models_video_{id}.pkl'
PATH_VECTORIZER = f'./model/vectorizer_obj_video_{id}.pkl'

SUB_CATEGORY_MODEL_NAME = ['am_nhac', 'am_thuc', 'bat_dong_san', 'cong_nghe', 'dien_anh', 'doi_song', 'du_lich', 'giai_tri',
                           'giao_duc', 'khoa_hoc', 'kinh_doanh', 'lam_dep', 'phap_luat', 'suc_khoe', 'the_gioi',
                           'the_thao', 'thoi_trang', 'van_hoa', 'xa_hoi', 'xe_co']
CATEGORY_MODEL_NAME = 'chuyen_muc'


def predict(model, vectorizer_obj, data, predict_proba=False):
    y_pred_vectorize = vectorizer(data, vec_obj=vectorizer_obj, fit=False)

    if predict_proba == True:
        y_pred = model.predict_proba(y_pred_vectorize)
        return y_pred
    else:
        y_pred = model.predict(y_pred_vectorize)
        return y_pred


if __name__ == "__main__":
    print('Loading data test...')
    df_test = pd.read_csv(PATH_DATA_TEST)

    # clear_label = ['am_nhac', 'am_thuc', 'dien_anh', 'khoa_hoc', 'lam_dep', 'thoi_trang', 'xe_co']

    # for i in clear_label:
    #     df_test = df_test.drop(df_test[(df_test['category'] == i)].index)

    # df_test = df_test.drop('category_pred', axis=1)
    
    # df_test = df_test[df_test['category'].notna()]
    df_test['clean_data'] = df_test.apply(lambda x: preparing_data(str(x['title']) + str(x['summary'])), axis=1)


    print('Loading models...')
    models = load_obj(PATH_MODELS)
    print('Loading vectorizer...')
    vectorizer_obj = load_obj(PATH_VECTORIZER)

    
    x_ = predict(models[CATEGORY_MODEL_NAME], vectorizer_obj[CATEGORY_MODEL_NAME], df_test['clean_data'].values.astype('U'), predict_proba=False)
    df_test['category_pred'] = x_
    # print(classification_report(df_test['category'].values, df_test['category_pred'].values))


    # labels = []
    # for g in df_test.groupby('category'):
    #     labels.append(g[0])
    # labels.sort()
    # cm = confusion_matrix(df_test['category'].values, df_test['category_pred'].values)
    # my_fig, (ax, my_cbar_ax) = plt.subplots(ncols=2, figsize=(28, 24),
    #                                         gridspec_kw={'width_ratios': [30, 1]})
    # sns.heatmap(cm, annot=True, ax = ax, cbar_ax = my_cbar_ax)
    # ax.set_xlabel('Predicted labels')
    # ax.set_ylabel('True labels')
    # ax.set_title(f'Confusion Matrix Video')
    # ax.xaxis.set_ticklabels(labels)
    # ax.yaxis.set_ticklabels(labels)
    # plt.savefig(f'./image/cm_fig_video_test_tfidf.png')
    # df_test = df_test.drop('clean_data', axis=1)
    df_test.to_csv('./data/video_test_pred.csv', index=False)

    # *** Correlation TF-IDF ***
    # terms = vectorizer_obj[CATEGORY_MODEL_NAME].get_feature_names()
    # for i in df_test.groupby('category'):
    #     print(f'Correlation TF-IDF {i[0]}')
    #     i_vectorize = vectorizer(i[1]['clean_data'].values.astype('U'), vec_obj=vectorizer_obj[CATEGORY_MODEL_NAME], fit=False)
    #     sums = i_vectorize.sum(axis=0)
    #     rank_tf_idf = []
    #     for col, term in enumerate(terms):
    #         rank_tf_idf.append((term, sums[0, col]))
    #     ranking = pd.DataFrame(rank_tf_idf, columns=['term', 'rank'])
    #     ranking = ranking.sort_values(by=['rank'], ascending=False)
    #     print(ranking.head(25))