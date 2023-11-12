import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # the KNN model
from sklearn.metrics import f1_score  # typical metric used to measure goodness of a model


def find_random_state(features_df, labels, n=200):
    """Takes a dataframe in and runs the variance code on it.
    It should return the value to use for the random state in the split method.
    """
    model = KNeighborsClassifier(n_neighbors=5)  # instantiate with k=5.
    var = []  # collect test_error/train_error where error based on F1 score

    for i in range(1, n):
        train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,
                                                            random_state=i, stratify=labels)
        model.fit(train_X, train_y)  # train model
        train_pred = model.predict(train_X)  # predict against training set
        test_pred = model.predict(test_X)  # predict against test set
        train_f1 = f1_score(train_y, train_pred)  # F1 on training predictions
        test_f1 = f1_score(test_y, test_pred)  # F1 on test predictions
        f1_ratio = test_f1 / train_f1  # take the ratio
        var.append(f1_ratio)

    rs_value = sum(var) / len(var)  # get average ratio value
    idx = np.array(abs(var - rs_value)).argmin()  # find the index of the smallest value
    return idx


def predict(X, w, b):
    yhat = [(x * w + b) for x in X]
    return yhat


def MSE(Y, Yhat):
    sdiffs = [(yhat - y) ** 2 for y, yhat in zip(Y, Yhat)]
    mse = sum(sdiffs) / len(sdiffs)
    return mse


def sgd(X, Y, w, b, lr=.001):
    # X is list of ages, Y is list of fare_labels, w is starting weight, b is stating bias, lr is learning rate
    for i in range(len(X)):
        # get values for rowi
        xi = X[i]  # e.g., age on rowi
        yi = Y[i]  # e.g., actual fare on rowi
        yhat = w * xi + b  # prediction
        # loss = (yhat - yi)**2 = ((w*xi+b) - yi)**2, i.e., f(g(x,w,b),yi)
        # dloss/dw = dl/dyhat * dyhat/dw by the chain rule
        # dloss/dyhat = 2*(yhat - yi) by the power rule (first part of chain)
        # dyhat/dw = d((w*xi+b) - yi)/dw = xi (second part of chain)
        gradient_w = 2 * (yhat - yi) * xi  # take the partial derivative wrt w to get slope
        # for b same first part of chain but then #dyhat/db = d((w*xi+b) - yi)/db = 1 for second part of chain
        gradient_b = 2 * (yhat - yi) * 1  # take the partial derivative wrt b to get slope
        w = w - lr * gradient_w  # if len(X) is 2000, will change 2000 times
        b = b - lr * gradient_b
        # return the last w and b of the loop
    return w, b


def full_batch(X, Y, w, b, lr=.001):
    gw = []
    gb = []
    for i in range(len(X)):
        xi = X[i]
        yi = Y[i]
        yhat = w * xi + b  # prediction
        gradient_w = 2 * (yhat - yi) * xi
        gradient_b = 2 * (yhat - yi) * 1
        gw.append(gradient_w)  # just collect change, don't make it
        gb.append(gradient_b)
    w = w - lr * sum(gw) / len(gw)  # Now average and make the change.
    b = b - lr * sum(gb) / len(gb)
    return w, b


def mini_batch(X, Y, w, b, batch=1, lr=.001):
    assert batch >= 1
    assert batch <= len(X)

    num = len(X) // batch if len(X) % batch == 0 else len(X) // batch + 1

    for batch_idx in range(num):
        start = batch_idx * batch
        batch_X = X[start:batch + start]
        batch_Y = Y[start:batch + start]
        gw = []
        gb = []
        for i in range(len(batch_X)):
            xi = batch_X[i]
            yi = batch_Y[i]
            yhat = w * xi + b  # prediction
            gradient_w = 2 * (yhat - yi) * xi
            gradient_b = 2 * (yhat - yi) * 1
            gw.append(gradient_w)
            gb.append(gradient_b)
        w = w - lr * sum(gw) / len(gw)
        b = b - lr * sum(gb) / len(gb)
    return w, b