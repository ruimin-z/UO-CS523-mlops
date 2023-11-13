"""
Chapter 9
Author: Ruimin Zhang
Date: Nov 6, 2023
"""

from sklearn.linear_model import LogisticRegressionCV
from library import *


'''Read in Data, Data Wrangling'''
url = 'https://raw.githubusercontent.com/ruimin-z/mlops/main/datasets/titanic_trimmed.csv'  # trimmed version
titanic_trimmed = pd.read_csv('../datasets/titanic_trimmed.csv')
X_train_numpy, X_test_numpy, y_train_numpy, y_test_numpy = titanic_setup(titanic_trimmed)

'''Define Model'''
model = LogisticRegressionCV(cv=5, random_state=1, max_iter=5000)

model.fit(X_train_numpy, y_train_numpy)

scores_df = pd.DataFrame(model.scores_[1], columns=range(1,11))  #accuracies across folds and Cs values
scores_df.describe().T

model.Cs_  #the 10 alternatives that were tried
model.C_  #final Cs chosen - #7
lambda_ = 1 / model.C_

# Module method for prediction
yhat = model.predict(X_test_numpy)
sum([a==b for a,b in zip(yhat, y_test_numpy)])/len(yhat)  #accuracy

# user-defined prediction using probability and sigmoid
yprob = model.predict_proba(X_test_numpy)  #output from sigmoid as pair (perished, survived)
yprob = yprob[:,1]  #grab the 2nd (1) column
threshold = .5  #might want to explore different values - see below
yhat2 = [0 if v<=threshold else 1 for v in yprob]
all(yhat2==yhat)  #yhat is what we got from model.predict
sum([a==b for a,b in zip(yhat2, y_test_numpy)])/len(yhat2)  #accuary on test set

# change sigmoid threshold
threshold = .2  #might want to explore different values - see below
yhat3 = [0 if v<=threshold else 1 for v in yprob]
sum([a==b for a,b in zip(yhat3, y_test_numpy)])/len(yhat3)  #accuracy 0.4828897338403042

'''Lime Explainer'''
import lime # Local Interpretable Model-Agnostic Explanations for machine learning classifiers
from lime import lime_tabular

feature_names = ['Age', 'Gender', 'Class', 'Joined', 'Married', 'Fare']  # X_train_transformed.columns.to_list()

explainer = lime.lime_tabular.LimeTabularExplainer(X_train_numpy,
                    feature_names=feature_names,
                    training_labels=y_train_numpy,
                    class_names=[0,1], #Outcome values
                    verbose=True,
                    mode='classification')

# Store into file
import dill as pickle
with open('../pkl/lime_explainer.pkl', 'wb') as file:
    pickle.dump(explainer, file)

#['Age', 'Gender', 'Class', 'Married', 'Fare', 'Joined']
new_row = np.array([.25, 0, 0, 0, .26, .4])
logreg_explanation = explainer.explain_instance(new_row, model.predict_proba, num_features=len(feature_names))
'''
Intercept 0.8379706320988898
Prediction_local [0.24889557]
Right: 0.07596865072971111
'''

logreg_explanation.predict_proba  #perishing vs surviving - predicting survived at .72
'''array([0.92403135, 0.07596865])'''
logreg_explanation.show_in_notebook()  # graphical view - how far each variable if pushing towards 0 or 1 prediction

logreg_explanation.as_list()  # used as explanation

