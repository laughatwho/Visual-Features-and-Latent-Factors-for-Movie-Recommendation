from graphlab import factorization_recommender
import graphlab as gl

test3 = gl.SFrame.read_csv('/Users/sanxi/Desktop/ml-latest-2016/test3.csv')
train3 = gl.SFrame.read_csv('/Users/sanxi/Desktop/ml-latest-2016/train3.csv')
items3 = gl.SFrame.read_csv('/Users/sanxi/Desktop/ml-latest-2016/items3_quantile.csv')

model_Quantile = factorization_recommender.create(train3, user_id='userId',
                                                  item_id='movieId', target='rating',
                                                  item_data=items3, solver='sgd')

feature_precision_recall_Quantile = model_Quantile.evaluate_precision_recall(test3, cutoffs=[5, 10, 30])

feature_Quantile_precision = feature_precision_recall_Quantile['precision_recall_overall']['precision']
feature_Quantile_recall = feature_precision_recall_Quantile['precision_recall_overall']['recall']
feature_Quantile_fscore = 2.0*(feature_Quantile_precision*feature_Quantile_recall)\
                          / (feature_Quantile_precision+feature_Quantile_recall)

print 'feature_Quantile_fscore ',feature_Quantile_fscore



