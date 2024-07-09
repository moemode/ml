import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *

#######################################################################
# 1. Introduction
#######################################################################

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()
# Plot the first 20 images of the training set.
# plot_images(train_x[0:20, :])

#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################



def run_linear_regression_on_MNIST(lambda_factor=1):
    """
    Trains linear regression, classifies test data, computes test error on test set

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x_bias, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x_bias, test_y, theta)
    return test_error


""" lambdas = np.array([0.01, 0.1, 1])
for lambda_factor in lambdas:
    print('Linear Regression test_error with lambda =', lambda_factor, ':', run_linear_regression_on_MNIST(lambda_factor)) """


#######################################################################
# 3. Support Vector Machine
#######################################################################


def run_svm_one_vs_rest_on_MNIST(C=0.1):
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x, C)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


""" Cs = [0.01, 0.1, 1]
for C in Cs:
    print('SVM one vs. rest test_error with C =', C, ':', run_svm_one_vs_rest_on_MNIST(C)) """
# print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST())


def run_multiclass_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


# print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())

#######################################################################
# 4. Multinomial (Softmax) Regression and Gradient Descent
#######################################################################

def run_softmax_on_MNIST(temp_parameter=1, mod3=False):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(
        train_x,
        train_y,
        temp_parameter,
        alpha=0.3,
        lambda_factor=1.0e-4,
        k=10,
        num_iterations=150,
    )
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    if mod3:
        return compute_test_error_mod3(test_x, test_y, theta, temp_parameter)
    else:
        return test_error


"""
temps = [0.5, 1.0, 2.0]
for temp in temps:
    print('softmax test_error with temp_parameter =', temp, ':', run_softmax_on_MNIST(temp))
"""
#######################################################################
# 6. Changing Labels
#######################################################################
# print(f"Error rate for labels mod 3: {run_softmax_on_MNIST(mod3=True)}")


def run_softmax_on_MNIST_mod3(temp_parameter=1):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y, test_y = update_y(train_y, test_y)
    theta, cost_function_history = softmax_regression(
        train_x,
        train_y,
        temp_parameter,
        alpha=0.3,
        lambda_factor=1.0e-4,
        k=10,
        num_iterations=150,
    )
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    return test_error


# Run run_softmax_on_MNIST_mod3(), report the error rate
# print(f"Error rate for labels mod 3 when trained on modified labels: {run_softmax_on_MNIST_mod3()}")


#######################################################################
# 7. Classification Using Manually Crafted Features
#######################################################################

## Dimensionality reduction via PCA ##
n_components = 18
train_x_centered, feature_means = center_data(train_x)
pcs = principal_components(train_x_centered)
# train_pca (and test_pca) is a representation of our training (and test) data
# after projecting each example onto the first 18 principal components.
train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

# Train your softmax regression model using (train_pca, train_y)
# and evaluate its accuracy on (test_pca, test_y).
def train_test_softmax_regression(
    train_x,
    train_y,
    test_x,
    test_y,
    temp_parameter=1,
    alpha=0.3,
    lambda_factor=1.0e-4,
    k=10,
    num_iterations=150,
):
    theta, cost_function_history = softmax_regression(
        train_x, train_y, temp_parameter, alpha, lambda_factor, k, num_iterations
    )
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    return theta, test_error, cost_function_history


train_test_softmax_regression(train_pca, train_y, test_pca, test_y)
theta, test_error, _ = train_test_softmax_regression(
    train_pca, train_y, test_pca, test_y
)
print(f"Error rate for 18-dimensional PCA features: {test_error}")


# Produce scatterplot of the first 100 MNIST images, as represented in the space
# spanned by the first 2 principal components found above.
plot_PC(train_x[range(000, 100),], pcs, train_y[range(000, 100)], feature_means)


# Use the reconstruct_PC function in features.py to show
#       the first and second MNIST images as reconstructed solely from
#       their 18-dimensional principal component representation.
#       Compare the reconstructed images with the originals.
""" firstimage_reconstructed = reconstruct_PC(
    train_pca[0,], pcs, n_components, train_x, feature_means
)  # feature_means added since release
plot_images(firstimage_reconstructed)
plot_images(train_x[0,])

secondimage_reconstructed = reconstruct_PC(
    train_pca[1,], pcs, n_components, train_x, feature_means
)  # feature_means added since release
plot_images(secondimage_reconstructed)
plot_images(train_x[1,])
 """

## Cubic Kernel ##
# Find the 10-dimensional PCA representation of the training and test set
n_components = 10
pcs_10 = pcs[:, :n_components]
train_pca10 = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca10 = project_onto_PC(test_x, pcs, n_components, feature_means)


# train_cube (and test_cube) is a representation of our training (and test) data
# after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.
train_cube = cubic_features(train_pca10)
test_cube = cubic_features(test_pca10)

# Train your softmax regression model using (train_cube, train_y)
# and evaluate its accuracy on (test_cube, test_y).
theta, test_error, _ = train_test_softmax_regression(
    train_cube, train_y, test_cube, test_y
)
print(f"Error rate for 10-dimensional cubic PCA features: {test_error}")
#### Polynomial SVM using scikit-learn ####

# Apply the cubic polynomial SVM to the 10-dimensional PCA representation of the training data
svm_poly_3 = SVC(kernel='poly', degree=3, random_state=0)
svm_poly_3.fit(train_pca10, train_y)
svm_predictions = svm_poly_3.predict(test_pca10)
svm_test_error = 1 - accuracy_score(test_y, svm_predictions)
print(f"Error rate for 10-dimensional PCA features using cubic polynomial SVM: {svm_test_error}")

# Apply the RBF SVM to the 10-dimensional PCA representation of the training data
svm_rbf = SVC(kernel='rbf', random_state=0)
svm_rbf.fit(train_pca10, train_y)
rbf_svm_predictions = svm_rbf.predict(test_pca10)
# Calculate the error rate
rbf_svm_test_error = 1 - accuracy_score(test_y, rbf_svm_predictions)
print(f"Error rate for 10-dimensional PCA features using RBF SVM: {rbf_svm_test_error}")
