from Transformers import *
from MachineLearning import *


def dataset_setup(original_table, label_column_name: str, the_transformer, rs, ts=.2):
    labels = original_table[label_column_name].to_list()
    table_features = original_table.drop(columns=label_column_name)
    X_train, X_test, y_train, y_test = train_test_split(table_features, labels, test_size=ts, shuffle=True,
                                                        random_state=rs, stratify=labels)
    X_train_transformed = the_transformer.fit_transform(X_train, y_train)  # fit with train
    X_test_transformed = the_transformer.transform(X_test)  # only transform

    # Transform to numpy array
    X_train_numpy = X_train_transformed.to_numpy()
    X_test_numpy = X_test_transformed.to_numpy()
    y_train_numpy = np.array(y_train)
    y_test_numpy = np.array(y_test)
    return X_train_numpy, X_test_numpy, y_train_numpy, y_test_numpy


def titanic_setup(titanic_table, transformer=titanic_transformer, rs=titanic_variance_based_split, ts=.2):
    return dataset_setup(titanic_table, 'Survived', transformer, rs=rs, ts=ts)


def customer_setup(customer_table, transformer=customer_transformer, rs=customer_variance_based_split, ts=.2):
    return dataset_setup(customer_table, 'Rating', transformer, rs=rs, ts=ts)
