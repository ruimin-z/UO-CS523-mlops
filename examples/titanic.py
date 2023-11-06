from library import *
import matplotlib.pyplot as plt

def linear_regression_model(X_train_transformed, X_test_transformed):
    # Find the model to import then train it using X and y (ages and fares)
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error, r2_score

    # Create linear regression object
    model = linear_model.LinearRegression()

    # Train the model using the training sets
    X = X_train_transformed['Age'].to_numpy().reshape(-1, 1)  # see example above for reshape action
    y = X_train_transformed['Fare'].to_numpy().reshape(-1, 1)
    model.fit(X, y)

    # Make predictions using the testing set
    x_test_transformed = X_test_transformed['Age'].to_numpy().reshape(-1, 1)
    y_test = X_test_transformed['Fare'].to_numpy().reshape(-1, 1)
    y_pred = model.predict(x_test_transformed)

    # The coefficients
    print("Coefficients: \n", model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    # Plot outputs
    plt.scatter(x_test_transformed, y_test, color="black")
    plt.plot(x_test_transformed, y_pred, color="blue", linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def udf_linear_regression(X_train_transformed, X_test_transformed):
    # define x column and y column
    ages = X_train_transformed['Age'].to_list()
    fare_labels = X_train_transformed['Fare'].to_list()  # the yi values

    # draw scatter plot of x and y
    plt.scatter(ages, fare_labels)
    plt.xlabel("Ages")
    plt.ylabel("Fare")
    plt.show()

    # another scatter plot
    ax = plt.axes()
    ax.scatter(ages, fare_labels)
    ax.set_xlabel('Ages')
    ax.set_ylabel('Fare')
    plt.show()

    # Predict fares from age. Record MSE for each epoch.
    w = .5
    b = .05
    mse = []
    epochs = 100
    for i in range(epochs):
        w, b = sgd(ages, fare_labels, w, b)  # note start with w and b from past epoch
        yhats = predict(ages, w, b)
        mse.append(MSE(fare_labels, yhats))

    # minumum epoch and best epoch
    best_epoch = np.argmin(mse)
    print(f'minimum mse = {min(mse)}, best epoch = {best_epoch}')

    # Plot epochs (x axis) against MSE values per epoch.
    plt.plot(mse, scaley=True)
    plt.show()

    # We need the w and b that go with the best epoch. Run the loop again but now only for 52 epochs.
    w = .5
    b = .05
    mse = []
    epochs = best_epoch
    for i in range(epochs):
        w, b = sgd(ages, fare_labels, w, b)  # note start with w and b from past epoch
        yhats = predict(ages, w, b)
        mse.append(MSE(fare_labels, yhats))
    print(f'w={w},b={b}')

    # Use best w,b to get error using X_test_transformed.
    Yhat = predict(X_test_transformed['Age'], w, b)
    mse = MSE(X_test_transformed['Fare'], Yhat)
    print(f'mse on test data = {mse}')


def main():
    '''Read titanic_trimmed.csv File'''
    url = 'https://raw.githubusercontent.com/ruimin-z/mlops/main/datasets/titanic_trimmed.csv'  # trimmed version
    titanic_trimmed = pd.read_csv(url)
    titanic_features = titanic_trimmed.drop(columns='Survived')

    # machine learning algorithm
    labels = titanic_trimmed['Survived'].to_list()
    X_train, X_test, y_train, y_test = train_test_split(titanic_features, labels, test_size=0.2, shuffle=True,
                                                        random_state=titanic_variance_based_split, stratify=labels)
    X_train_transformed = titanic_transformer.fit_transform(X_train, y_train)  # fit with train
    X_test_transformed = titanic_transformer.transform(X_test)  # only transform

    linear_regression_model(X_train_transformed, X_test_transformed)
    udf_linear_regression(X_train_transformed, X_test_transformed)


if __name__ == '__main__':
    main()
