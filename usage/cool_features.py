import pandas as pd
from sklearn.pipeline import Pipeline

'''Transform into numpy arrays in pipeline'''
from sklearn.preprocessing import FunctionTransformer

#here is my custom function. Has to have this signature line.
def numpy_converter(X, y=None):
  assert isinstance(X, pd.core.frame.DataFrame)
  return X.to_numpy()

#Now I pass the function in
numpy_transformer = FunctionTransformer(numpy_converter)

# Plug into a pipeline
p = Pipeline(steps=[('numpy', numpy_transformer)])

p.transform(X_test_transformed)