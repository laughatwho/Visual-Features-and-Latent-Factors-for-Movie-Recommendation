import graphlab as gl

movie = gl.SFrame.read_csv('/Users/sanxi/Desktop/ml-latest/movies.csv')
movie.remove_columns(['title'])
movie['genres'] = movie['genres'].apply(lambda x: x.split('|'))
n = 34208 - 45843
movies_2017 = movie.tail(-n)
movie = movie.filter_by(movies_2017['movieId'], 'movieId', exclude=True)

rating = gl.SFrame.read_csv('/Users/sanxi/Desktop/ml-latest/ratings.csv')
rating.remove_columns(['timestamp'])
rating = rating[rating['userId'] <= 247753]
rating = rating[rating['movieId'] < movies_2017['movieId'][0]]

visual_features_QuantileLog = gl.SFrame.read_csv('/Users/sanxi/Desktop/visual/QuantileLog.csv')
visual_features_QuantileLog.rename({'ML_Id': 'movieId'})

visual_features_log = gl.SFrame.read_csv('/Users/sanxi/Desktop/visual/Log.csv')
visual_features_log.rename({'ML_Id': 'movieId'})

visual_features_Quantile = gl.SFrame.read_csv('/Users/sanxi/Desktop/visual/Quantile.csv')
visual_features_Quantile.rename({'ML_Id':'movieId'})

special_items = rating.groupby('movieId', gl.aggregate.COUNT).sort('Count')
rare_items = special_items[special_items['Count'] <= 5]
popular_items = special_items[special_items['Count'] >= 800]

rating = gl.cross_validation.shuffle(rating)

folds = gl.cross_validation.KFold(rating, 5)


(train1, test1) = folds[0]
(train2, test2) = folds[1]
(train3, test3) = folds[2]
(train4, test4) = folds[3]
(train5, test5) = folds[4]

train1 = train1.filter_by(rare_items['movieId'], 'movieId', exclude=True)
train1 = train1.filter_by(popular_items['movieId'], 'movieId', exclude=True)

train2 = train2.filter_by(rare_items['movieId'], 'movieId', exclude=True)
train2 = train2.filter_by(popular_items['movieId'], 'movieId', exclude=True)

train3 = train3.filter_by(rare_items['movieId'], 'movieId', exclude=True)
train3 = train3.filter_by(popular_items['movieId'], 'movieId', exclude=True)

train4 = train4.filter_by(rare_items['movieId'], 'movieId', exclude=True)
train4 = train4.filter_by(popular_items['movieId'], 'movieId', exclude=True)

train5 = train5.filter_by(rare_items['movieId'], 'movieId', exclude=True)
train5 = train5.filter_by(popular_items['movieId'], 'movieId', exclude=True)

train1 = train1[train1['rating'] > 3]
train2 = train2[train2['rating'] > 3]
train3 = train3[train3['rating'] > 3]
train4 = train4[train4['rating'] > 3]
train5 = train5[train5['rating'] > 3]


train1_data_genre = train1.join(movie, on='movieId', how='left')
train2_data_genre = train2.join(movie, on='movieId', how='left')
train3_data_genre = train3.join(movie, on='movieId', how='left')
train4_data_genre = train4.join(movie, on='movieId', how='left')
train5_data_genre = train5.join(movie, on='movieId', how='left')

train1_data_log = train1.join(visual_features_log, on='movieId', how='left')
train2_data_log = train2.join(visual_features_log, on='movieId', how='left')
train3_data_log = train3.join(visual_features_log, on='movieId', how='left')
train4_data_log = train4.join(visual_features_log, on='movieId', how='left')
train5_data_log = train5.join(visual_features_log, on='movieId', how='left')

train1_data_quantile = train1.join(visual_features_Quantile, on='movieId', how='left')
train2_data_quantile = train2.join(visual_features_Quantile, on='movieId', how='left')
train3_data_quantile = train3.join(visual_features_Quantile, on='movieId', how='left')
train4_data_quantile = train4.join(visual_features_Quantile, on='movieId', how='left')
train5_data_quantile = train5.join(visual_features_Quantile, on='movieId', how='left')

train1_data_quantilelog = train1.join(visual_features_QuantileLog, on='movieId', how='left')
train2_data_quantilelog = train2.join(visual_features_QuantileLog, on='movieId', how='left')
train3_data_quantilelog = train3.join(visual_features_QuantileLog, on='movieId', how='left')
train4_data_quantilelog = train4.join(visual_features_QuantileLog, on='movieId', how='left')
train5_data_quantilelog= train5.join(visual_features_QuantileLog, on='movieId', how='left')

items1_genre = train1_data_genre.select_columns(['movieId', 'genres'])
items2_genre = train2_data_genre.select_columns(['movieId', 'genres'])
items3_genre = train3_data_genre.select_columns(['movieId', 'genres'])
items4_genre = train4_data_genre.select_columns(['movieId', 'genres'])
items5_genre = train5_data_genre.select_columns(['movieId', 'genres'])

items1_log = train1_data_log.select_columns(['movieId', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])
items2_log = train2_data_log.select_columns(['movieId', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])
items3_log = train3_data_log.select_columns(['movieId', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])
items4_log = train4_data_log.select_columns(['movieId', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])
items5_log = train5_data_log.select_columns(['movieId', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])

items1_quantile = train1_data_quantile.select_columns(['movieId', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])
items2_quantile = train2_data_quantile.select_columns(['movieId', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])
items3_quantile = train3_data_quantile.select_columns(['movieId', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])
items4_quantile = train4_data_quantile.select_columns(['movieId', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])
items5_quantile = train5_data_quantile.select_columns(['movieId', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])

items1_quantilelog = train1_data_quantilelog.select_columns(['movieId', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])
items2_quantilelog = train2_data_quantilelog.select_columns(['movieId', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])
items3_quantilelog = train3_data_quantilelog.select_columns(['movieId', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])
items4_quantilelog = train4_data_quantilelog.select_columns(['movieId', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])
items5_quantilelog = train5_data_quantilelog.select_columns(['movieId', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])


items1_log = items1_log.fillna('f1', 0)
items1_log = items1_log.fillna('f2', 0)
items1_log = items1_log.fillna('f3', 0)
items1_log = items1_log.fillna('f4', 0)
items1_log = items1_log.fillna('f5', 0)
items1_log = items1_log.fillna('f6', 0)
items1_log = items1_log.fillna('f7', 0)


items2_log = items2_log.fillna('f1', 0)
items2_log = items2_log.fillna('f2', 0)
items2_log = items2_log.fillna('f3', 0)
items2_log = items2_log.fillna('f4', 0)
items2_log = items2_log.fillna('f5', 0)
items2_log = items2_log.fillna('f6', 0)
items2_log = items2_log.fillna('f7', 0)

items3_log = items3_log.fillna('f1', 0)
items3_log = items3_log.fillna('f2', 0)
items3_log = items3_log.fillna('f3', 0)
items3_log = items3_log.fillna('f4', 0)
items3_log = items3_log.fillna('f5', 0)
items3_log = items3_log.fillna('f6', 0)
items3_log = items3_log.fillna('f7', 0)

items4_log = items4_log.fillna('f1', 0)
items4_log = items4_log.fillna('f2', 0)
items4_log = items4_log.fillna('f3', 0)
items4_log = items4_log.fillna('f4', 0)
items4_log = items4_log.fillna('f5', 0)
items4_log = items4_log.fillna('f6', 0)
items4_log = items4_log.fillna('f7', 0)

items5_log = items5_log.fillna('f1', 0)
items5_log = items5_log.fillna('f2', 0)
items5_log = items5_log.fillna('f3', 0)
items5_log = items5_log.fillna('f4', 0)
items5_log = items5_log.fillna('f5', 0)
items5_log = items5_log.fillna('f6', 0)
items5_log = items5_log.fillna('f7', 0)

items1_quantilelog = items1_quantilelog.fillna('f1', 0)
items1_quantilelog = items1_quantilelog.fillna('f2', 0)
items1_quantilelog = items1_quantilelog.fillna('f3', 0)
items1_quantilelog = items1_quantilelog.fillna('f4', 0)
items1_quantilelog = items1_quantilelog.fillna('f5', 0)
items1_quantilelog = items1_quantilelog.fillna('f6', 0)
items1_quantilelog = items1_quantilelog.fillna('f7', 0)

items2_quantilelog = items2_quantilelog.fillna('f1', 0)
items2_quantilelog = items2_quantilelog.fillna('f2', 0)
items2_quantilelog = items2_quantilelog.fillna('f3', 0)
items2_quantilelog = items2_quantilelog.fillna('f4', 0)
items2_quantilelog = items2_quantilelog.fillna('f5', 0)
items2_quantilelog = items2_quantilelog.fillna('f6', 0)
items2_quantilelog = items2_quantilelog.fillna('f7', 0)

items3_quantilelog = items3_quantilelog.fillna('f1', 0)
items3_quantilelog = items3_quantilelog.fillna('f2', 0)
items3_quantilelog = items3_quantilelog.fillna('f3', 0)
items3_quantilelog = items3_quantilelog.fillna('f4', 0)
items3_quantilelog = items3_quantilelog.fillna('f5', 0)
items3_quantilelog = items3_quantilelog.fillna('f6', 0)
items3_quantilelog = items3_quantilelog.fillna('f7', 0)

items4_quantilelog = items4_quantilelog.fillna('f1', 0)
items4_quantilelog = items4_quantilelog.fillna('f2', 0)
items4_quantilelog = items4_quantilelog.fillna('f3', 0)
items4_quantilelog = items4_quantilelog.fillna('f4', 0)
items4_quantilelog = items4_quantilelog.fillna('f5', 0)
items4_quantilelog = items4_quantilelog.fillna('f6', 0)
items4_quantilelog = items4_quantilelog.fillna('f7', 0)

items5_quantilelog = items5_quantilelog.fillna('f1', 0)
items5_quantilelog = items5_quantilelog.fillna('f2', 0)
items5_quantilelog = items5_quantilelog.fillna('f3', 0)
items5_quantilelog = items5_quantilelog.fillna('f4', 0)
items5_quantilelog = items5_quantilelog.fillna('f5', 0)
items5_quantilelog = items5_quantilelog.fillna('f6', 0)
items5_quantilelog = items5_quantilelog.fillna('f7', 0)

items1_quantile = items1_quantile.fillna('f1', 0)
items1_quantile = items1_quantile.fillna('f2', 0)
items1_quantile = items1_quantile.fillna('f3', 0)
items1_quantile = items1_quantile.fillna('f4', 0)
items1_quantile = items1_quantile.fillna('f5', 0)
items1_quantile = items1_quantile.fillna('f6', 0)
items1_quantile = items1_quantile.fillna('f7', 0)

items2_quantile = items2_quantile.fillna('f1', 0)
items2_quantile = items2_quantile.fillna('f2', 0)
items2_quantile = items2_quantile.fillna('f3', 0)
items2_quantile = items2_quantile.fillna('f4', 0)
items2_quantile = items2_quantile.fillna('f5', 0)
items2_quantile = items2_quantile.fillna('f6', 0)
items2_quantile = items2_quantile.fillna('f7', 0)

items3_quantile = items3_quantile.fillna('f1', 0)
items3_quantile = items3_quantile.fillna('f2', 0)
items3_quantile = items3_quantile.fillna('f3', 0)
items3_quantile = items3_quantile.fillna('f4', 0)
items3_quantile = items3_quantile.fillna('f5', 0)
items3_quantile = items3_quantile.fillna('f6', 0)
items3_quantile = items3_quantile.fillna('f7', 0)

items4_quantile = items4_quantile.fillna('f1', 0)
items4_quantile = items4_quantile.fillna('f2', 0)
items4_quantile = items4_quantile.fillna('f3', 0)
items4_quantile = items4_quantile.fillna('f4', 0)
items4_quantile = items4_quantile.fillna('f5', 0)
items4_quantile = items4_quantile.fillna('f6', 0)
items4_quantile = items4_quantile.fillna('f7', 0)

items5_quantile = items5_quantile.fillna('f1', 0)
items5_quantile = items5_quantile.fillna('f2', 0)
items5_quantile = items5_quantile.fillna('f3', 0)
items5_quantile = items5_quantile.fillna('f4', 0)
items5_quantile = items5_quantile.fillna('f5', 0)
items5_quantile = items5_quantile.fillna('f6', 0)
items5_quantile = items5_quantile.fillna('f7', 0)

gl.SFrame.save(test1, '/Users/sanxi/Desktop/ml-latest-2016/test1.csv')
gl.SFrame.save(test2, '/Users/sanxi/Desktop/ml-latest-2016/test2.csv')
gl.SFrame.save(test3, '/Users/sanxi/Desktop/ml-latest-2016/test3.csv')
gl.SFrame.save(test4, '/Users/sanxi/Desktop/ml-latest-2016/test4.csv')
gl.SFrame.save(test5, '/Users/sanxi/Desktop/ml-latest-2016/test5.csv')

gl.SFrame.save(train1, '/Users/sanxi/Desktop/ml-latest-2016/train1.csv')
gl.SFrame.save(train2, '/Users/sanxi/Desktop/ml-latest-2016/train2.csv')
gl.SFrame.save(train3, '/Users/sanxi/Desktop/ml-latest-2016/train3.csv')
gl.SFrame.save(train4, '/Users/sanxi/Desktop/ml-latest-2016/train4.csv')
gl.SFrame.save(train5, '/Users/sanxi/Desktop/ml-latest-2016/train5.csv')

gl.SFrame.save(items1_log, '/Users/sanxi/Desktop/ml-latest-2016/items1_log.csv')
gl.SFrame.save(items2_log, '/Users/sanxi/Desktop/ml-latest-2016/items2_log.csv')
gl.SFrame.save(items3_log, '/Users/sanxi/Desktop/ml-latest-2016/items3_log.csv')
gl.SFrame.save(items4_log, '/Users/sanxi/Desktop/ml-latest-2016/items4_log.csv')
gl.SFrame.save(items5_log, '/Users/sanxi/Desktop/ml-latest-2016/items5_log.csv')

gl.SFrame.save(items1_quantile, '/Users/sanxi/Desktop/ml-latest-2016/items1_quantile.csv')
gl.SFrame.save(items2_quantile, '/Users/sanxi/Desktop/ml-latest-2016/items2_quantile.csv')
gl.SFrame.save(items3_quantile, '/Users/sanxi/Desktop/ml-latest-2016/items3_quantile.csv')
gl.SFrame.save(items4_quantile, '/Users/sanxi/Desktop/ml-latest-2016/items4_quantile.csv')
gl.SFrame.save(items5_quantile, '/Users/sanxi/Desktop/ml-latest-2016/items5_quantile.csv')

gl.SFrame.save(items1_quantilelog, '/Users/sanxi/Desktop/ml-latest-2016/items1_quantilelog.csv')
gl.SFrame.save(items2_quantilelog, '/Users/sanxi/Desktop/ml-latest-2016/items2_quantilelog.csv')
gl.SFrame.save(items3_quantilelog, '/Users/sanxi/Desktop/ml-latest-2016/items3_quantilelog.csv')
gl.SFrame.save(items4_quantilelog, '/Users/sanxi/Desktop/ml-latest-2016/items4_quantilelog.csv')
gl.SFrame.save(items5_quantilelog, '/Users/sanxi/Desktop/ml-latest-2016/items5_quantilelog.csv')

gl.SFrame.save(items1_genre, '/Users/sanxi/Desktop/ml-latest-2016/items1_genre.csv')
gl.SFrame.save(items2_genre, '/Users/sanxi/Desktop/ml-latest-2016/items2_genre.csv')
gl.SFrame.save(items3_genre, '/Users/sanxi/Desktop/ml-latest-2016/items3_genre.csv')
gl.SFrame.save(items4_genre, '/Users/sanxi/Desktop/ml-latest-2016/items4_genre.csv')
gl.SFrame.save(items5_genre, '/Users/sanxi/Desktop/ml-latest-2016/items5_genre.csv')


