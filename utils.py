import pickle
import os
import random
import re
import string
import time
from html.parser import HTMLParser

import joblib
import pandas as pd

from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# from vncorenlp import VnCoreNLP
# rdrsegmenter = VnCoreNLP("../vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

filename = './data/dict/stopwords.csv'
data_stopword = pd.read_csv(filename, encoding='utf-8')
list_stopword = data_stopword['stopwords']
list_stopword = list_stopword.values.tolist()


class MLStripper(HTMLParser):

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ' '.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data().translate(str.maketrans(' ', ' ', string.punctuation)).strip()


def read_data(path):
    data = []
    with open(path, 'r', encoding='utf8') as f:
        rawdata = f.read().splitlines()
    for text in rawdata:
        data.append(text.split(':', 1))

    # print(len(data))
    X = [data[i][1] for i in range(len(data))]
    y = [data[i][0] for i in range(len(data))]

    return X, y


def clean_link(text):
    # Liên kết
    text = re.sub(r"http\S+", " ", text)
    return text


def normalize_Text(text):
    # Lower
    text = text.encode(encoding='utf8').decode('utf-8')
    text = text.lower()

    moneytag = [u'tỷ', u'đồng', u'vnd', u'nghìn', u'usd', u'triệu', u'ounce', u'yên', u'bảng']

    text_ = text.split()
    for i in range(len(text_)):
        if (text_[i] in moneytag):
            text_[i] = 'money'

    delete_x = [u'nan']
    for i in range(len(text_)):
        if (text_[i] in delete_x):
            text_[i] = ' '

    text = ' '.join(text_)

    # Dấu câu bị vỡ cấu trúc
    listpunctuation = string.punctuation
    for i in listpunctuation:
        text = text.replace(i, ' ')

    # Thông số
    text = re.sub('^(\d+[a-z]+)([a-z]*\d*)*\s|\s\d+[a-z]+([a-z]*\d*)*\s|\s(\d+[a-z]+)([a-z]*\d*)*$', ' ', text)
    text = re.sub('^([a-z]+\d+)([a-z]*\d*)*\s|\s[a-z]+\d+([a-z]*\d*)*\s|\s([a-z]+\d+)([a-z]*\d*)*$', ' ', text)

    # Âm tiết
    text = re.sub(r'(\D)\1+', r'\1', text)

    return text


def remove_Stopword(text):
    re_text = []
    words = text.split()
    for word in words:
        if (not word.isnumeric()) and (len(words) > 1):
            if not (word in list_stopword):
                re_text.append(word)

    text = ' '.join(re_text)

    return text


# def tokenize(text):
#     sentences = rdrsegmenter.tokenize(text)

#     text_token = ''
#     for sentence in sentences:
#         text_token = text_token + ' ' + ' '.join(sentence)

#     return text_token

def tokenize(text):
    text_token = ViTokenizer.tokenize(text)
    return text_token


def predata(path):
    X, y = read_data(path)
    X_re = []
    i = 0
    for text in X:
        text = remove_Stopword(tokenize(normalize_Text(clean_link(text))))
        X_re.append(text)
        i += 1

    return X_re, y


def preprocess(path_raw, path_pre):
    X_re, y = predata(path_raw)
    writer = open(path_pre, 'w', encoding='utf8')

    for i in range(len(X_re)):
        writer.write(str(y[i]) + ': ')
        writer.write(X_re[i])
        writer.write('\n')


def prepare_data(data):
    out = []
    for i in data:
        out.append(remove_Stopword(tokenize(normalize_Text(clean_link(i)))))
    return out


def clean_data(in_path='./data/raw_data/video_train_.csv',
               out_path='./data/video_train_clean.csv'):
    def _prepare_data(*x):
        sum_x = ' '.join(list(x))
        return remove_Stopword(tokenize(normalize_Text(clean_link(sum_x))))
    df = pd.read_csv(in_path)
    df['clean_data'] = df.apply(lambda x: _prepare_data(str(x['title']), str(x['summary'])), axis=1)
    # df_drop_nan = df[['category', 'sub_category', 'url', 'clean_data']].dropna()
    df_drop_nan = df[['category', 'url', 'clean_data']].dropna()
    # df_drop_nan = convert_label(df_drop_nan, 'the_thao', 'C1 Champions League', 'champions_league')
    df_drop_nan.to_csv(out_path, index=False)


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        return obj


def split_data(data, test_size=0.15, random_state=42, file=False, exp_id=str(int(time.time()))):
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for g in data.groupby(['category']):
        train_set, test_set = train_test_split(g[1], test_size=test_size, random_state=random_state)
        train_df = pd.concat([train_df, train_set], ignore_index=True)
        test_df = pd.concat([test_df, test_set], ignore_index=True)
    if file:
        print(f"Saved train & test file {exp_id}")
        train_df.to_csv(f'./data/train_{exp_id}.csv', index=False)
        test_df.to_csv(f'./data/test_{exp_id}.csv', index=False)
    return train_df, test_df


def drop_sub_category_little_labels(data, category, sub_category):
    return data.drop(data[(data['category'] == category) & (data['sub_category'] == sub_category)].index)

def convert_label(data, category, sub_category, convert_category):
    tmp = data[(data['category'] == category) & (data['sub_category'] == sub_category)]
    for i in tmp.index:
        # print(i)
        data['sub_category'].iloc[i] = convert_category
    return data


def load_model(file_path):
    with open(file_path, 'rb') as file:
        pickle_model = joblib.load(file)
    return pickle_model


def load_models(dir_path):
    models = {}
    for file in os.listdir(dir_path):
        try:
            if file.endswith('.pkl'):
                name = file.split('.')[0].split('_model')[0]
                models[name] = load_model(os.path.join(dir_path, file))
        except Exception as e:
            print(str(e))
            continue
    return models


def vectorizer(data, fit=False, vec_obj=None, min_df=0.0, max_df=1.0, ngram_range=(1, 2), max_features=None):
    vectorize = TfidfVectorizer(use_idf=True, min_df=min_df, max_df=max_df, ngram_range=ngram_range, max_features=max_features)
    if fit is False:
        if vec_obj is None:
            vec = vectorize.fit_transform(data)
        else:
            vec = vec_obj.transform(data)
        return vec
    else:
        if vec_obj is None:
            vectorize.fit(data)
            return vectorize
        else:
            vec_obj.fit(data)
            return vec_obj


def penalized_ambiguous_feature_lv2(root_label, amb_labels, vectorizer_obj_sub, vectorizer_obj):
    ambi_features = []
    for amb in amb_labels:
        ambi_features.append(set(vectorizer_obj_sub[root_label].get_feature_names()).intersection(set(vectorizer_obj_sub[amb].get_feature_names())))
    ambi_feature_set = list(set.intersection(*ambi_features))
    # print(len(ambi_feature_set))
    for i in ambi_feature_set[:25]:
        try:
            vectorizer_obj.vocabulary_[i] = vectorizer_obj.vocabulary_[i] - random.randint(5, 15)
        except:
            continue
    return vectorizer_obj


def penalized_ambiguous_feature(root_label, amb_labels, vectorizer_obj):
    ambi_features = []
    for amb in amb_labels:
        ambi_features.append(set(vectorizer_obj[root_label].get_feature_names()).intersection(set(vectorizer_obj[amb].get_feature_names())))
    ambi_feature_set = list(set.intersection(*ambi_features))
    # print(len(ambi_feature_set))
    for i in ambi_feature_set[:30]:
        try:
            vectorizer_obj['chuyen_muc'].vocabulary_[i] = vectorizer_obj['chuyen_muc'].vocabulary_[i] - random.randint(5, 15)
        except:
            continue
    return vectorizer_obj



def stat_feature():
    df = pd.read_csv('../flask_app/tmp/evaluation_es_1593403093.csv')
    vectorizer_obj = load_obj('../flask_app/model/vectorizer_obj_0603.pkl')
    models = load_obj('../flask_app/model/models_0603.pkl')
    VECTORIZER_OBJ = dict()
    used_label = ['khoa_hoc', 'phap_luat', 'van_hoa', 'xa_hoi']
    for g in df.groupby('category_truth'):
        VECTORIZER_OBJ[g[0]] = vectorizer(g[1]['content_clean'].values.astype('U'), fit=True)
    pass


if __name__ == '__main__':
    # load_obj('../flask_app/model/models.pkl')
    clean_data()
    # stat_feature()
    # penalized_ambiguous_feature('khoa_hoc', ['du_lich', 'doi_song'], vectorizer_obj=load_obj('../flask_app/model/vectorizer_obj_1593417536_small.pkl'))
    # clean_data()
    pass