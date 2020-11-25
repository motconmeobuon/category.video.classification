import pandas as pd


def mapping_data(data, map, out_path='./data/raw_data/video_train_.csv'):
    data = data[data['category_id'].notna()]
    data['category'] = data.apply(lambda x: map.loc[(map['category_id_'] == int(x['category_id']))]['category'].iloc[0], axis=1)
    # print(map.loc[(map['category_id_'] == 42)]['category'].iloc[0])
    # print(data)
    data = data.drop('category_id', axis=1)
    data.to_csv(out_path, index=False)


def mapping_data_test(data_test_new, data_test_old, out_path='./data/video_test.csv'):
    data_test_new['category'] = None
    for i in range(len(data_test_new)):
        # print(data_test_new.loc[i, 'url'])
        for j in range(len(data_test_old)):
            if (data_test_new.loc[i, 'url'] == data_test_old.loc[j, 'url']):
                data_test_new.loc[i, 'category'] = data_test_old.loc[j, 'category']
                break
    
    data_test_new.to_csv(out_path, index=False)


def check_data(data):
    for g in data.groupby(['category']):
        print(g[0])
        print(g[1].shape)
        # for i in g[1].groupby(['sub_category']):
        #     print(f'    {i[0]}')
        #     print(f'    {i[1].shape}')
        print()

def test():
    s1 = 'video_test_1106'
    df1 = pd.read_csv(f'./data/test/{s1}.csv').drop_duplicates(subset=['url'])
    df1 = df1.drop('summary', axis=1)
    print(f'Data {s1}: {df1.shape[0]}')

    s2 = 'video_test_1105'
    df2 = pd.read_csv(f'./data/test/{s2}.csv').drop_duplicates(subset=['url'])
    df2 = df2.drop('summary', axis=1)
    print(f'Data {s2}: {df2.shape[0]}')

    data = pd.concat([df1, df2])
    print(f'Data tổng chưa drop_duplicates: {data.shape[0]}')
    data = data.drop_duplicates(subset=['url'])
    print(f'Data tổng đã drop_duplicates: {data.shape[0]}')


if __name__ == "__main__":
    # data_test_old = pd.read_csv('./data/video_test_pred_tfidf.csv')
    # data_test_new = pd.read_csv('./data/test/video_test_1106.csv')
    # mapping_data_test(data_test_new, data_test_old, out_path='./data/video_test.csv')

    data = pd.read_csv('./data/video_train_clean.csv')
    check_data(data)


    # test()

    pass