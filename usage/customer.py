from library import *

'''Read customer.csv File'''
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQPM6PqZXgmAHfRYTcDZseyALRyVwkBtKEo_rtaKq_C7T0jycWxH6QVEzTzJCRA0m8Vz0k68eM9tDm-/pub?output=csv'
customers_df = pd.read_csv(url)
customers_trimmed = customers_df.drop(columns='ID')  # this is a useless column which we will drop early
customers_trimmed = customers_trimmed.drop_duplicates(ignore_index=True)  # get rid of any duplicates

x_train_numpy, x_test_numpy, y_train_numpy, y_test_numpy = customer_setup(customers_trimmed)
