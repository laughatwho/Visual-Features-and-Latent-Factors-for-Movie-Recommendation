import graphlab as gl


test3 = gl.SFrame.read_csv('/Users/sanxi/Desktop/ml-latest-2016/test3.csv')
train3 = gl.SFrame.read_csv('/Users/sanxi/Desktop/ml-latest-2016/train3.csv')
items3 = gl.SFrame.read_csv('/Users/sanxi/Desktop/ml-latest-2016/items3_genre.csv')

model_Genre = gl.factorization_recommender.create(train3, user_id='userId',
                                                  item_id='movieId', target='rating',
                                                  item_data=items3, solver='sgd')

genre_precision_recall = model_Genre.evaluate_precision_recall(test3, cutoffs=[5, 10, 30])


genre_precision = genre_precision_recall['precision_recall_overall']['precision']
genre_recall = genre_precision_recall['precision_recall_overall']['recall']
genre_fscore = 2.0*(genre_precision*genre_recall)/(1.0*(genre_precision+genre_recall))
print 'genre_precision_recall_fscore', genre_precision_recall, genre_fscore




