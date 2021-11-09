#####################################################
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston

import streamlit as st 
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import warnings

warnings.filterwarnings("ignore")
#########################################################

st.set_page_config(page_title='Boston', page_icon="🏠")
#st.set_page_config("Dashboard",  layout="wide")
st.image('img/bosten.png')
st.title('Boston Housing Prices 🏠')
##########################################################
#load dataset
data = load_boston()
st.write(data['DESCR'])
################################

df = pd.DataFrame(data['data'], columns = data['feature_names']) 
df['MEDV'] = pd.DataFrame(data['target'], columns=['MEDV'])
st.write('#### Top 5 of the dataset')
st.write(df.head())
####################################

X = df.drop(['MEDV'], axis=1)
y = df['MEDV']


def train_DecisionTreeRegressor(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)
    decision_regressor = DecisionTreeRegressor(max_depth=3, random_state=0)
    decision_regressor.fit(X_train, y_train)
    fig = plt.figure(figsize=(25, 14))

    st.title('Visualizations')
    st.write('#### Tree visualization')
    dot_data  = export_graphviz(decision_regressor, feature_names=data['feature_names'], filled=True)
    st.graphviz_chart(dot_data)

    y_pred = decision_regressor.predict(X_test)
    fig = plt.figure(figsize=(8,4))
    st.write('#### Prediction plot')
    plt.scatter(y_pred, y_test)
    m, b = np.polyfit(y_pred, y_test, 1)
    plt.plot(y_pred, m*y_pred + b)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.title('Predicted vs Actual')
    st.write(fig)

    st.title("Evaluating the model")
    MSE = mean_squared_error(y_test, y_pred)
    st.write('#### Mean Squared Error: ',MSE ) 
    d = {"Actual":y_test.ravel(), "Predicted":y_pred.ravel()}
    df1 = pd.DataFrame(d).sample(10)

    st.write()
    st.write('#### Actual vs Predicted dataframe')
    st.write(df1)       

# st.title('Visualization')
# train_DecisionTreeRegressor(X, y)
#########################################################################
def RandomForestRegressor_(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)
    # Bagging: using all features
    max_features = st.sidebar.slider('Select maximum features', 1, 13)
    regr1 = RandomForestRegressor(max_features=max_features, random_state=0)
    regr1.fit(X_train, y_train.values.ravel())

    st.title('Visualizations')
    y_pred = regr1.predict(X_test)
    fig = plt.figure(figsize=(8,4))
    st.write('#### Prediction plot')
    plt.scatter(y_pred, y_test)
    m, b = np.polyfit(y_pred, y_test, 1)
    plt.plot(y_pred, m*y_pred + b)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.title('Predicted vs Actual')
    st.write(fig)

    st.title("Evaluating the model")
    MSE = mean_squared_error(y_test, y_pred)
    st.write('#### Mean Squared Error: ',MSE ) 
    d = {"Actual":y_test.ravel(), "Predicted":y_pred.ravel()}
    df1 = pd.DataFrame(d).sample(10)

    st.write()
    st.write('## Actual vs Predicted dataframe')
    st.write(df1)

    st.write('##  Feature importance')
    importances = regr1.feature_importances_

    def rf_feat_importance(m, df):
        return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_} ).sort_values('imp', ascending=False)
    fi = rf_feat_importance(regr1, X)
    st.write('#### Feature Importance dataframe')
    st.write(fi)

    f = plt.figure(figsize=(12,7))
    plt.barh(fi['cols'], fi['imp'])
    plt.title("Feature importance")
    st.write(f)

# st.title('Visualization')
# RandomForestRegressor_(X, y)
####################################################################

def GridSearchCV_(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)
    hyp = st.sidebar.multiselect('Select hyper-parameters', ('n_estimators', 'max_depth',  'criterion'))
    parametr ={"n_estimators": [100, 200, 300], 
                'max_depth':[8, 10], 
                    'criterion' : ['mse', 'mae']}
    p = {}
    st.write(hyp[0])
    for i in range (len(hyp)): p[hyp[i]] = parametr[hyp[i]]
    #parameters = {'max_depth':[8, 10], 'max_features':[7, 10, 13]}
    rand_forest_reg3 = RandomForestRegressor(random_state=0)
    clf = GridSearchCV(rand_forest_reg3, p)
    clf.fit(X_train, y_train.values.ravel())
   


    y_pred = clf.predict(X_test)
    MSE = mean_squared_error(y_pred, y_test)
    st.write('#### Mean Squared Error: ',MSE )

# GridSearchCV_(X, y )

model = st.sidebar.selectbox('Select model', 
            ('DecisionTreeRegressor','RandomForestRegressor', 'GridSearchCV'))


if model == 'GridSearchCV':  
    st.title('GridSearchCV')
    GridSearchCV_(X, y )
elif model == 'RandomForestRegressor': 
    st.title('Decision TreeRegressor')
    train_DecisionTreeRegressor(X, y)
else: 
    st.title('Random Forests')
    RandomForestRegressor_(X, y)

