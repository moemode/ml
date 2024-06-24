import project1 as p1
import utils
import numpy as np

#-------------------------------------------------------------------------------
# Data loading. There is no need to edit code in this section.
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

dictionary = p1.bag_of_words(train_texts)

train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)

#-------------------------------------------------------------------------------
# Problem 5
#-------------------------------------------------------------------------------

# toy_features, toy_labels = toy_data = utils.load_toy_data('toy_data.tsv')

# T = 1500
# L = 0.2

# thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)
# thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)
# thetas_pegasos = p1.pegasos(toy_features, toy_labels, T, L)

# def plot_toy_results(algo_name, thetas):
#     print('theta for', algo_name, 'is', ', '.join(map(str,list(thetas[0]))))
#     print('theta_0 for', algo_name, 'is', str(thetas[1]))
#     utils.plot_toy_data(algo_name, toy_features, toy_labels, thetas)

# plot_toy_results('Perceptron', thetas_perceptron)
# plot_toy_results('Average Perceptron', thetas_avg_perceptron)
# plot_toy_results('Pegasos', thetas_pegasos)

#-------------------------------------------------------------------------------
# Problem 7
#-------------------------------------------------------------------------------

# T = 10
# L = 0.01

# pct_train_accuracy, pct_val_accuracy = \
#    p1.classifier_accuracy(p1.perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
# print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
# print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))

# avg_pct_train_accuracy, avg_pct_val_accuracy = \
#    p1.classifier_accuracy(p1.average_perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
# print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
# print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))

# avg_peg_train_accuracy, avg_peg_val_accuracy = \
#    p1.classifier_accuracy(p1.pegasos, train_bow_features,val_bow_features,train_labels,val_labels,T=T,L=L)
# print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
# print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))

#-------------------------------------------------------------------------------
# Problem 8
#-------------------------------------------------------------------------------

# data = (train_bow_features, train_labels, val_bow_features, val_labels)

# # values of T and lambda to try
# Ts = [1, 5, 10, 15, 25, 50]
# Ls = [0.001, 0.01, 0.1, 1, 10]

# pct_tune_results = utils.tune_perceptron(Ts, *data)
# print('perceptron valid:', list(zip(Ts, pct_tune_results[1])))
# print('best = {:.4f}, T={:.4f}'.format(np.max(pct_tune_results[1]), Ts[np.argmax(pct_tune_results[1])]))

# avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
# print('avg perceptron valid:', list(zip(Ts, avg_pct_tune_results[1])))
# print('best = {:.4f}, T={:.4f}'.format(np.max(avg_pct_tune_results[1]), Ts[np.argmax(avg_pct_tune_results[1])]))

# # fix values for L and T while tuning Pegasos T and L, respective
# fix_L = 0.01
# peg_tune_results_T = utils.tune_pegasos_T(fix_L, Ts, *data)
# print('Pegasos valid: tune T', list(zip(Ts, peg_tune_results_T[1])))
# print('best = {:.4f}, T={:.4f}'.format(np.max(peg_tune_results_T[1]), Ts[np.argmax(peg_tune_results_T[1])]))

# fix_T = Ts[np.argmax(peg_tune_results_T[1])]
# peg_tune_results_L = utils.tune_pegasos_L(fix_T, Ls, *data)
# print('Pegasos valid: tune L', list(zip(Ls, peg_tune_results_L[1])))
# print('best = {:.4f}, L={:.4f}'.format(np.max(peg_tune_results_L[1]), Ls[np.argmax(peg_tune_results_L[1])]))

# utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
# utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
# utils.plot_tune_results('Pegasos', 'T', Ts, *peg_tune_results_T)
# utils.plot_tune_results('Pegasos', 'L', Ls, *peg_tune_results_L)

#-------------------------------------------------------------------------------
# Use the best method (perceptron, average perceptron or Pegasos) along with
# the optimal hyperparameters according to validation accuracies to test
# against the test dataset. The test data has been provided as
# test_bow_features and test_labels.
#-------------------------------------------------------------------------------

# print(p1.classifier_accuracy(p1.pegasos, train_bow_features, test_bow_features, train_labels, test_labels, L=0.01, T=25))

#-------------------------------------------------------------------------------
# Assign to best_theta, the weights (and not the bias!) learned by your most
# accurate algorithm with the optimal choice of hyperparameters.
#-------------------------------------------------------------------------------
# best_theta, theta_0 = p1.pegasos(train_bow_features, train_labels, L=0.01, T=25)
# wordlist   = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
# # if interested in negative labels, negate best_theta:
# # sorted_word_features = utils.most_explanatory_word(-best_theta, wordlist)
# sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)
# print("Most Explanatory Word Features")
# print(sorted_word_features[:10])


#-------------------------------------------------------------------------------
# Problem 8
#-------------------------------------------------------------------------------
# Implement stop words removal in your feature engineering code. Specifically,
# load the file stopwords.txt, remove the words in the file from your dictionary
# by editing the bag_of_words function, and use features constructed from the new 
# dictionary to train your model and make predictions.
#
# Compare your result in the testing data on Pegasos algorithm using T=25 and L=0.01
# the dictionary before and after removing the stop words.
#
# Accuracy on the test set using the original dictionary: 0.8020
# Accuracy on the test set using the dictionary with stop words removed: 0.808
#-------------------------------------------------------------------------------
# load stopwords from file, one stopword per line
# with open('stopwords.txt', 'r', encoding='utf-8') as f:
#     stopwords = f.read().splitlines()

# dictionary_stop = p1.bag_of_words(train_texts, stopwords)
# train_bow_features_stop = p1.extract_bow_feature_vectors(train_texts, dictionary_stop)
# test_bow_features_stop = p1.extract_bow_feature_vectors(test_texts, dictionary_stop)
# print(p1.classifier_accuracy(p1.pegasos, train_bow_features_stop, test_bow_features_stop, 
#                              train_labels, test_labels, L=0.01, T=25))

#-------------------------------------------------------------------------------
# Again, use the same learning algorithm and the same feature as the last problem.
# However, when you compute the feature vector of a word, use its count in each 
# document rather than a binary indicator.
#
# Try to compare your result to the last problem.
# Accuracy on the test set using the dictionary with stop words removed: 0.808
# Accuracy on the test set using the dictionary with stop words removed and counts
# features: 0.77
# Using the counts leads to a lower accuracy.
#-------------------------------------------------------------------------------
# with open('stopwords.txt', 'r', encoding='utf-8') as f:
#     stopwords = f.read().splitlines()

# dictionary_stop = p1.bag_of_words(train_texts, stopwords)
# train_bow_features_stop = p1.extract_bow_feature_vectors(train_texts, dictionary_stop, False)
# test_bow_features_stop = p1.extract_bow_feature_vectors(test_texts, dictionary_stop, False)
# print(p1.classifier_accuracy(p1.pegasos, train_bow_features_stop, test_bow_features_stop,
#                              train_labels, test_labels, L=0.01, T=25))


#-------------------------------------------------------------------------------
# Some additional features that you might want to explore are:
#
#     Length of the text
#
#     Occurrence of all-cap words (e.g. “AMAZING", “DON'T BUY THIS")
#
#     Word embeddings 
#
# Besides adding new features, you can also change the original unigram feature set.
# For example,
#
#     Threshold the number of times a word should appear in the dataset before adding
#     them to the dictionary. For example, words that occur less than three times 
#     across the train dataset could be considered irrelevant and thus can be removed. 
#     This lets you reduce the number of columns that are prone to overfitting. 
#
# There are also many other things you could change when training your model. Try 
# anything that can help you understand the sentiment of a review. It's worth looking 
# through the dataset and coming up with some features that may help your model. 
# Remember that not all features will actually help so you should experiment with some
# simpler ones before trying anything too complicated.
# Accuracy on the test set using the dictionary with stop words removed: 0.808
# Accuracy on the test set using the dictionary with stop words removed and additional
# features (length of review, number of all caps words): 0.814
# Slight improvement by using additional features
#-------------------------------------------------------------------------------
with open('stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().splitlines()

dictionary_stop = p1.bag_of_words(train_texts, stopwords)
train_bow_features_stop = p1.extract_bow_feature_vectors(train_texts, dictionary_stop)
test_bow_features_stop = p1.extract_bow_feature_vectors(test_texts, dictionary_stop)
print(p1.classifier_accuracy(p1.pegasos, train_bow_features_stop, test_bow_features_stop, 
                             train_labels, test_labels, L=0.01, T=25))
train_bow_features_stop_additional = np.hstack((train_bow_features_stop, 
                                                p1.extract_additional_features(train_texts)))
test_bow_features_stop_additional = np.hstack((test_bow_features_stop, 
                                               p1.extract_additional_features(test_texts)))
print(p1.classifier_accuracy(p1.pegasos, train_bow_features_stop_additional, test_bow_features_stop_additional,
                             train_labels, test_labels, L=0.01, T=25))