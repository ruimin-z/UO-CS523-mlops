import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # the KNN model
from sklearn.metrics import f1_score  # typical metric used to measure goodness of a model

sklearn.set_config(transform_output="pandas")  # says pass pandas tables through pipeline instead of numpy matrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

import subprocess
import sys
subprocess.call([sys.executable, '-m', 'pip', 'install', 'category_encoders'])  #replaces !pip install
import category_encoders as ce


class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """Given a mapping of values to integers, this class converts values in a given column to corresponding integers."""

    def __init__(self, mapping_column, mapping_dict: dict):
        assert isinstance(mapping_dict,
                          dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
        self.mapping_dict = mapping_dict
        self.mapping_column = mapping_column  # column to focus on

    def fit(self, X, y=None):
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self

    def transform(self, X):
        """Given a dataframe, transform it to ... using a dictionary of mappings"""
        assert isinstance(X,
                          pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  # column legit?

        # Set up for producing warnings. First have to rework nan values to allow set operations to work.
        # In particular, the conversion of a column to a Series, e.g., X[self.mapping_column], transforms nan values in strange ways that screw up set differencing.
        # Strategy is to convert empty values to a string then the string back to np.nan
        placeholder = "NaN"
        column_values = X[self.mapping_column].fillna(
            placeholder).tolist()  # convert all nan values to the string "NaN" in new list
        column_values = [np.nan if v == placeholder else v for v in column_values]  # now convert back to np.nan
        keys_values = self.mapping_dict.keys()

        column_set = set(
            column_values)  # without the conversion above, the set will fail to have np.nan values where they should be.
        keys_set = set(keys_values)  # this will have np.nan values where they should be so no conversion necessary.

        # now check to see if all keys are contained in column.
        keys_not_found = keys_set - column_set
        if keys_not_found:
            print(
                f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these mapping keys do not appear in the column: {keys_not_found}\n")

        # now check to see if some keys are absent
        keys_absent = column_set - keys_set
        if keys_absent:
            print(
                f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these values in the column do not contain corresponding mapping keys: {keys_absent}\n")

        # do actual mapping
        X_ = X.copy()
        X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
        return X_

    def fit_transform(self, X, y=None):
        # self.fit(X,y)
        result = self.transform(X)
        return result


class CustomRenamingTransformer(BaseEstimator, TransformerMixin):
    """This class will rename one or more columns."""

    def __init__(self, renaming_dict: dict):
        assert isinstance(renaming_dict,
                          dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(renaming_dict)} instead.'
        self.renaming_dict = renaming_dict

        # define fit to do nothing but give warning

    def fit(self, X, y=None):
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self

        # write the transform method with asserts. Again, maybe copy and paste from MappingTransformer and fix up.

    def transform(self, X):
        assert isinstance(X,
                          pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

        keys_values = self.renaming_dict.keys()
        column_set = set(X.columns)
        keys_set = set(keys_values)  # this will have np.nan values where they should be so no conversion necessary.

        # now check to see if all keys are contained in column.
        keys_not_found = keys_set - column_set
        assert not keys_not_found, f"{self.__class__.__name__}[{self.renaming_dict}] these renaming keys do not appear in the column: {keys_not_found}\n"

        # do actual renaming
        X_ = X.copy()
        X_.rename(columns=self.renaming_dict, inplace=True)
        return X_

        # write fit_transform that skips fit

    def fit_transform(self, X, y=None):
        # self.fit(X,y)
        result = self.transform(X)
        return result


class CustomOHETransformer(BaseEstimator, TransformerMixin):
    """This class applies One Hot Encoding to a given column"""

    def __init__(self, target_column, dummy_na=False, drop_first=False):
        self.target_column = target_column
        self.dummy_na = dummy_na
        self.drop_first = drop_first

        # fill in the rest below

    def fit(self, X, y=None):
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self

    def transform(self, X):
        assert isinstance(X,
                          pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  # column legit?

        # do actual OHE
        X_ = pd.get_dummies(X,
                            prefix=self.target_column,
                            prefix_sep='_',
                            columns=[self.target_column],
                            dummy_na=self.dummy_na,
                            drop_first=self.drop_first)
        return X_

    def fit_transform(self, X, y=None):
        # self.fit(X,y)
        result = self.transform(X)
        return result


class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
    """
    Clip outliers using 3 Sigma method through standard deviation and mean
    """
    def __init__(self, target_column):
        self.target_column = target_column
        self.minb = None
        self.maxb = None

    def fit(self, X, y=None):
        """Calculate mean and standard deviation"""
        assert isinstance(X,
                          pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  # column legit?
        assert all([isinstance(v, (int, float)) for v in X[self.target_column].to_list()])

        std = X[self.target_column].std()
        mean = X[self.target_column].mean()
        self.minb = mean - 3 * std
        self.maxb = mean + 3 * std
        return self

    def transform(self, X):
        """Move the outliers to the boundary"""
        assert isinstance(X,
                          pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.minb is not None and self.maxb is not None, f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

        X_ = X.copy()
        X_[self.target_column] = X_[self.target_column].clip(lower=self.minb, upper=self.maxb)
        return X_.reset_index(drop=True)

    def fit_transform(self, X, y=None):
        self.fit(X)
        result = self.transform(X)
        return result


class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
    """
    Clip outliers using Tukey method through quartiles and median
    """
    def __init__(self, target_column, fence='outer'):
        assert fence in ['inner', 'outer']
        self.fence = fence
        self.target_column = target_column
        self.minb, self.maxb = None, None

    def fit(self, X, y=None):
        """Calculate boundary values based on fence"""
        assert isinstance(X,
                          pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  # column legit?
        assert all([isinstance(v, (int, float)) for v in X[self.target_column].to_list()])

        q1 = X[self.target_column].quantile(0.25)
        q3 = X[self.target_column].quantile(0.75)
        iqr = q3 - q1
        match self.fence:
            case 'inner':
                self.minb, self.maxb = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            case 'outer':
                self.minb, self.maxb = q1 - 3 * iqr, q3 + 3 * iqr
        return self

    def transform(self, X):
        """Move the outliers to the boundary"""
        assert isinstance(X,
                          pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.minb is not None and self.maxb is not None, f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

        X_ = X.copy()
        X_[self.target_column] = X_[self.target_column].clip(lower=self.minb, upper=self.maxb)
        return X_.reset_index(drop=True)

    def fit_transform(self, X, y=None):
        self.fit(X)
        result = self.transform(X)
        return result


class CustomRobustTransformer(BaseEstimator, TransformerMixin):
    """
    This class scales a given column values in [0,1] using Robust Scaler.
    Formula: value_new = (value – median) / iqr #iqr = q3-q1
    """
    def __init__(self, column):
        self.column = column
        self.iqr, self.med = None, None

    def fit(self, X, y=None):
        """Calculate iqr and median"""
        assert isinstance(X,
                          pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  # column legit?
        assert all([isinstance(v, (int, float)) for v in X[self.column].to_list()])

        q1 = X[self.column].quantile(0.25)
        q3 = X[self.column].quantile(0.75)
        self.iqr = q3 - q1
        self.med = X[self.column].median()
        return self

    def transform(self, X):
        """Scale using Robust Scaler"""
        assert isinstance(X,
                          pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.iqr is not None and self.med is not None, f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

        X_ = X.copy()
        X_[self.column] -= self.med
        X_[self.column] /= self.iqr
        # X_[self.column].fillna(0, inplace=True)
        return X_

    def fit_transform(self, X, y=None):
        self.fit(X)
        result = self.transform(X)
        return result


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



titanic_variance_based_split = 107 # random state that provides best average
customer_variance_based_split = 113 # random state that provides best average
titanic_transformer = Pipeline(steps=[
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('target_joined', ce.TargetEncoder(cols=['Joined'],
                                       handle_missing='return_nan',  # will use imputer later to fill in
                                       handle_unknown='return_nan'  # will use imputer later to fill in
                                       )),
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('tukey_fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('scale_age', CustomRobustTransformer('Age')),  # from chapter 5
    ('scale_fare', CustomRobustTransformer('Fare')),  # from chapter 5
    ('imputer', KNNImputer(n_neighbors=5, weights="uniform", add_indicator=False))  # from chapter 6
], verbose=True)

customer_transformer = Pipeline(steps=[
    ('map_os', CustomMappingTransformer('OS', {'Android': 0, 'iOS': 1})),
    ('target_isp', ce.TargetEncoder(cols=['ISP'],
                                    handle_missing='return_nan',  # will use imputer later to fill in
                                    handle_unknown='return_nan'  # will use imputer later to fill in
                                    )),
    ('map_level', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high': 2})),
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('tukey_age', CustomTukeyTransformer('Age', 'inner')),  # from chapter 4
    ('tukey_time spent', CustomTukeyTransformer('Time Spent', 'inner')),  # from chapter 4
    ('scale_age', CustomRobustTransformer('Age')),  # from 5
    ('scale_time spent', CustomRobustTransformer('Time Spent')),  # from 5
    ('impute', KNNImputer(n_neighbors=5, weights="uniform", add_indicator=False)),
], verbose=True)


'''Fitted Transformer'''
# fitted_pipeline = titanic_transformer.fit(X_train, y_train)  # notice just fit method called
# import joblib
# joblib.dump(fitted_pipeline, 'fitted_pipeline.pkl')


'''Read titanic_trimmed.csv File'''
# url = 'https://raw.githubusercontent.com/ruimin-z/mlops/main/datasets/titanic_trimmed.csv'  # trimmed version
# titanic_trimmed = pd.read_csv('datasets/titanic_trimmed.csv')
# titanic_features = titanic_trimmed.drop(columns='Survived')
# transformed_df = titanic_transformer.fit_transform(titanic_features)

'''Read customer.csv File'''
# url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQPM6PqZXgmAHfRYTcDZseyALRyVwkBtKEo_rtaKq_C7T0jycWxH6QVEzTzJCRA0m8Vz0k68eM9tDm-/pub?output=csv'
# customers_df = pd.read_csv(url)
# customers_trimmed = customers_df.drop(columns='ID')  #this is a useless column which we will drop early
# customers_trimmed = customers_trimmed.drop_duplicates(ignore_index=True)  #get rid of any duplicates
