import pickle
import time

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from utils import save_obj, split_data, vectorizer, penalized_ambiguous_feature, drop_sub_category_little_labels, convert_label

SUB_CATEGORY_MODEL_NAME = ['am_nhac', 'am_thuc', 'bat_dong_san', 'cong_nghe', 'doi_song', 'du_lich', 'giai_tri',
                           'giao_duc',
                           'khoa_hoc', 'kinh_doanh', 'lam_dep', 'phap_luat', 'suc_khoe', 'the_gioi', 'the_thao',
                           'thoi_trang',
                           'van_hoa', 'xa_hoi', 'xe_co', 'dien_anh']

CATEGORY_MODEL_NAME = 'chuyen_muc'
MODELS = {}
VECTORIZER_OBJ = {}


def fit_vectorizer(data, save=None):
    global VECTORIZER_OBJ
    ambiguous_labels = {'khoa_hoc': ['the_gioi'],
                        'xa_hoi': ['kinh_doanh', 'doi_song', 'the_gioi', 'giai_tri', 'suc_khoe', 'bat_dong_san'],
                        'phap_luat': ['doi_song', 'kinh_doanh', 'the_gioi', 'xa_hoi'],
                        'van_hoa': ['doi_song', 'giai_tri', 'du_lich'],
                        'am_nhac': ['dien_anh', 'giai_tri', 'van_hoa'],
                        'giai_tri': ['doi_song', 'dien_anh', 'thoi_trang'],
                        'cong_nghe': ['the_gioi', 'kinh_doanh']}


    for g in data.groupby(['category']):
        print(f"Fitting {g[0]} vectorizer")
        # print(type(g[1]['clean_data'].values.astype('U')))
        VECTORIZER_OBJ[g[0]] = vectorizer(g[1]['clean_data'].values.astype('U'), fit=True, max_features=2000)

    print(f"Fitting {CATEGORY_MODEL_NAME} vectorizer")
    VECTORIZER_OBJ[CATEGORY_MODEL_NAME] = vectorizer(data['clean_data'].values.astype('U'), fit=True, max_features=20000)
    for k, v in ambiguous_labels.items():
        VECTORIZER_OBJ = penalized_ambiguous_feature(k, v, VECTORIZER_OBJ)
    if save is not None:
        save_obj(VECTORIZER_OBJ, save)
    return VECTORIZER_OBJ


def model(x_train, y_train):
    clf = OneVsRestClassifier(SVC(kernel='linear', C=1, probability=True, verbose=True))
    clf.fit(x_train, y_train)
    return clf


def training(data, save=None):
    print(f"Transforming {CATEGORY_MODEL_NAME} vectorizer")
    transformed = vectorizer(data['clean_data'].values.astype('U'),
                             vec_obj=VECTORIZER_OBJ[CATEGORY_MODEL_NAME])
    print(f"Training {CATEGORY_MODEL_NAME} model")
    MODELS[CATEGORY_MODEL_NAME] = model(transformed, data['category'].values.astype('U'))
    if save is not None:
        save_obj(MODELS, save)
    return MODELS


def main():
    print("Reading data...")
    data = pd.read_csv('./data/video_train_clean.csv')
    print("Shape:", data.shape)

    # clear_label = ['am_nhac', 'am_thuc', 'dien_anh', 'khoa_hoc', 'lam_dep', 'thoi_trang', 'xe_co']

    # for i in clear_label:
    #     data = data.drop(data[(data['category'] == i)].index)

    data_x = pd.DataFrame({'id': [], 'category': [], 'clean_data': []})
    for g in data.groupby('category'):
        _g = g[1].sample(frac = 1)
        data_x = pd.concat([data_x, _g[:1200]])
    data = data_x

    exp_id = str(int(time.time()))
    train_df, test_df = split_data(data, file=True, exp_id=exp_id)
    fit_vectorizer(train_df, save=f'./model/vectorizer_obj_video_{exp_id}.pkl')
    training(train_df, save=f'./model/models_video_{exp_id}.pkl')
    print(f'Exp id {exp_id}')
    return VECTORIZER_OBJ, MODELS


if __name__ == '__main__':
    # df = pd.read_csv('../flask_app/data/category_tree_200_clean.csv')
    # split_data(df)
    main()
