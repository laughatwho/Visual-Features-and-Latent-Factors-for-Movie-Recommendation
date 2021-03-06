from graphlab import factorization_recommender
import graphlab as gl

test3 = gl.SFrame.read_csv('/Users/sanxi/Desktop/ml-latest-2016/test3.csv')
train3 = gl.SFrame.read_csv('/Users/sanxi/Desktop/ml-latest-2016/train3.csv')
items3 = gl.SFrame.read_csv('/Users/sanxi/Desktop/ml-latest-2016/items3_quantilelog.csv')

model_QuantileLog = factorization_recommender.create(train3, user_id='userId',
                                                     item_id='movieId',target='rating',
                                                     item_data=items3, solver='sgd')

feature_precision_recall_QuantileLog = model_QuantileLog.evaluate_precision_recall(test3, cutoffs=[5, 10, 30])

feature_QuantileLog_precision = feature_precision_recall_QuantileLog['precision_recall_overall']['precision']
feature_QuantileLog_recall = feature_precision_recall_QuantileLog['precision_recall_overall']['recall']
feature_QuantileLog_fscore = 2.0*(feature_QuantileLog_precision*feature_QuantileLog_recall)\
                             /(feature_QuantileLog_precision+feature_QuantileLog_recall)

print 'feature_QuantileLog_fscore',feature_QuantileLog_fscore
print 'feature_QuantileLog_pc',feature_precision_recall_QuantileLog

