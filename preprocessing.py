# -*- coding: utf-8 -*-

import re
import pandas as pd

filename = './data/dict/stopwords.csv'
data_stopword = pd.read_csv(filename)
list_stopword = data_stopword['stopwords']
list_stopword = list_stopword.values.tolist()

# from vncorenlp import VnCoreNLP
# rdrsegmenter = VnCoreNLP("../vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

def preparing_data(data_text):
    import re
    import string

    from pyvi import ViTokenizer, ViPosTagger

    def _clean_html(text):
        cleanr = re.compile(r'<[^>]+>')
        text = re.sub(cleanr, ' ', text)
        # print(text)
        return text

    def _clean_link(text):
        # Liên kết
        text = re.sub(r"http\S+", " ", text)
        return text

    def _normalize_text(text):

        text = _tokenize(text)

        moneytag = [u'tỷ', u'đồng', u'vnd', u'nghìn', u'usd', u'triệu', u'củ', u'ounce', u'yên', u'bảng']

        text_ = text.split()
        for i in range(len(text_)):
            if (text_[i] in moneytag):
                text_[i] = 'money'

        text = ' '.join(text_)

        # Lower
        text = text.encode(encoding='utf8').decode('utf-8')
        text = text.lower()

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

    def _remove_stop_word(text):
        # print(text)
        re_text = []
        words = text.split()
        for word in words:
            if (not word.isnumeric()) and (len(words) > 1):
                if not (word in list_stopword):
                    re_text.append(word)

        text = ' '.join(re_text)

        # print(text)
        return text

    # def _tokenize(text):
    #     sentences = rdrsegmenter.tokenize(text)

    #     text_token = ''
    #     for sentence in sentences:
    #         text_token = text_token + ' ' + ' '.join(sentence)
        
    #     return text_token

    def _tokenize(text):
        text_token = ViTokenizer.tokenize(text)
        # print(text_token)
        return text_token

    if type(data_text) == list:
        clean_text = [_remove_stop_word(_tokenize(_normalize_text(_clean_html(_clean_link(i))))) for i in data_text]
    else:
        clean_text = _remove_stop_word(_tokenize(_normalize_text(_clean_html(_clean_link(data_text)))))

    return clean_text

if __name__ == "__main__":
    pass
