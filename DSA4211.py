import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV,ElasticNetCV,LassoLarsCV,RidgeCV
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm
from sklearn.feature_selection import RFECV
from sklearn import linear_model
from warnings import simplefilter

# ignore future warnings
simplefilter(action='ignore', category=FutureWarning)

test_x = pd.read_csv("test-x.csv")
train_xy = pd.read_csv("train-xy.csv")
train_y = train_xy.iloc[:, 0] 
train_x = train_xy.iloc[:, 1:] 

"""
first, we preprocess the data by checking the number of na values per row
"""

def check_na(train_x):
    na_per_column = train_x.isnull().sum()
    for i in range(100):
        na_count = na_per_column[i]
        if na_count>0:
            print(f"Nan in X{i+1} is : {na_count}")

check_na(train_x)

"""
Since X52 has 193 NA values, we remove it.
"""
train_x = train_x.drop(['X52'], axis=1)
test_x = test_x.drop(['X52'], axis=1)

def normalise_x(train_x,test_x):
    train_x_col = train_x.columns
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    train_x = pd.DataFrame(train_x, columns=train_x_col)
    test_x = pd.DataFrame(test_x, columns=train_x_col)
    print(train_x)
    return train_x,test_x

def standardise_x(train_x,test_x):
    train_x_col = train_x.columns
    sc = StandardScaler()
    train_s_x = sc.fit_transform(train_x)
    test_s_x = sc.transform(test_x)
    train_s_x = pd.DataFrame(train_x, columns=train_x_col)
    test_s_x = pd.DataFrame(test_s_x, columns=train_x_col)
    return(train_s_x,test_s_x)

'''
Check for Collinearity
By checking against the VIF for the data, we remove variables that have VIF of more than 10
'''

def get_VIF(train_x,test_x):    
    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = train_x.columns
    
    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(train_x.values, i)
                            for i in range(len(train_x.columns))]
    high_vif =[]
    for i in range(len(vif_data['VIF'])):
        vif = (vif_data['VIF'][i])
        feature = (vif_data['feature'][i])
        if vif>10:
            print(f'The feature {feature} has a vif of {vif}, suggesting it has high collinearity')
            high_vif.append(feature)

    for f in range(len(high_vif)-1):
        feature = high_vif[f]
        train_x = train_x.drop([f'{feature}'], axis=1)
        test_x = test_x.drop([str(feature)], axis=1)
        print(f'The feature {feature} has been dropped')

    return (train_x,test_x)

#remove variables with high VIF 
train_x,test_x = get_VIF(train_x,test_x)

def recursive_feature_elimination(train_x,train_y):
    ols = linear_model.LinearRegression()
    rfecv = RFECV(estimator=ols, scoring='r2')
    features = train_x.columns
    # Fit recursive feature eliminator 
    rfecv.fit(train_x, train_y)
    # Recursive feature elimination
    rfecv.transform(train_x)
    support = rfecv.get_support()
    #print(support)
    features = list(features[support])
    print(f"The {rfecv.n_features_} features selected are {features}.")
    return features

"""
X14                            
X13                            
X12                            
X82                            
X94 
"""
#carry out subset selection
selected_features = recursive_feature_elimination(train_x,train_y)
train_x= train_x[selected_features]
test_x = test_x[selected_features]


def lasso_cv(x,y,cv):
    #perform l1 regularization with lasso on cv folds, 
    #selecting the most appropriate alpha and returning the relevant score and cv score
    lasso_cv = LassoCV(cv=cv)
    lasso_cv = lasso_cv.fit(x,y)
    lasso_score = lasso_cv.score(x,y)
    lasso_alpha = lasso_cv.alpha_
    lasso_cv_score = np.mean(cross_val_score(LassoCV(), x, y, scoring='r2', cv=cv))
    return lasso_alpha, lasso_score , lasso_cv_score

def ridge_cv(x,y,cv):
    #perform l2 regularization with ridge regression on cv folds, 
    #selecting the most appropriate alpha and returning the relevant score and cv score
    ridge_cv = RidgeCV(cv=cv)
    ridge_cv.fit(x,y)
    ridge_score = ridge_cv.score(x,y)
    ridge_alpha = ridge_cv.alpha_
    ridge_cv_score = np.mean(cross_val_score(RidgeCV(), x, y, scoring='r2', cv=cv))
    return ridge_alpha, ridge_score , ridge_cv_score

def elastic_net_cv(x,y,cv):
    #perform l1 and l2 regularization with elastic net on cv folds, 
    #selecting the most appropriate alpha and returning the relevant score and cv score
    elastic_cv = ElasticNetCV(cv=cv)
    elastic_cv.fit(x,y)
    elastic_score = elastic_cv.score(x,y)
    elastic_alpha = elastic_cv.alpha_
    elastic_cv_score = np.mean(cross_val_score(ElasticNetCV(), x, y, scoring='r2', cv=cv))
    return elastic_alpha, elastic_score , elastic_cv_score

def lars_cv(x,y,cv):
    #perform lars regression on cv folds, 
    #selecting the most appropriate alpha and returning the relevant score and cv score
    lars_cv = LassoLarsCV(cv=cv)
    lars_cv = lars_cv.fit(x,y)
    lars_score = lars_cv.score(x,y)
    lars_alpha = lars_cv.alpha_
    lars_cv_score = np.mean(cross_val_score(LassoLarsCV(), x, y, scoring='r2', cv=cv))
    return lars_alpha, lars_score , lars_cv_score 

cv=10
train_n_x,test_n_x = normalise_x(train_x,test_x)
train_s_x,test_s_x = standardise_x(train_x,test_x)

lasso_alpha, lasso_score , lasso_cv_score = lasso_cv(train_x,train_y,cv)
ridge_alpha, ridge_score , ridge_cv_score = ridge_cv(train_x,train_y,cv)
elastic_alpha, elastic_score , elastic_cv_score = elastic_net_cv(train_x,train_y,cv)
lars_alpha, lars_score , lars_cv_score = lars_cv(train_x,train_y,cv)

lasso_n_alpha, lasso_n_score , lasso_n_cv_score = lasso_cv(train_n_x,train_y,cv)
ridge_n_alpha, ridge_n_score , ridge_n_cv_score = ridge_cv(train_n_x,train_y,cv)
elastic_n_alpha, elastic_n_score , elastic_n_cv_score = elastic_net_cv(train_n_x,train_y,cv)
lars_n_alpha, lars_n_score , lars_n_cv_score = lars_cv(train_n_x,train_y,cv)

lasso_s_alpha, lasso_s_score , lasso_s_cv_score = lasso_cv(train_s_x,train_y,cv)
ridge_s_alpha, ridge_s_score , ridge_s_cv_score = ridge_cv(train_s_x,train_y,cv)
elastic_s_alpha, elastic_s_score , elastic_s_cv_score = elastic_net_cv(train_s_x,train_y,cv)
lars_s_alpha, lars_s_score , lars_s_cv_score = lars_cv(train_s_x,train_y,cv)

print(f"In the Lasso model, the alpha is {lasso_alpha}, the score is {lasso_score} and the cv score is {lasso_cv_score}")
print(f"In the Ridge model, the alpha is {ridge_alpha}, the score is {ridge_score} and the cv score is {ridge_cv_score}")
print(f"In the Elastic Net model, the alpha is {elastic_alpha}, the score is {elastic_score} and the cv score is {elastic_cv_score}")
print(f"In the Lars model, the alpha is {lars_alpha}, the score is {lars_score} and the cv score is {lars_cv_score}")

print(f"In the Lasso model with normalised data , the alpha is {lasso_n_alpha}, the score is {lasso_n_score} and the cv score is {lasso_n_cv_score}")
print(f"In the Ridge model with normalised data , the alpha is {ridge_n_alpha}, the score is {ridge_n_score} and the cv score is {ridge_n_cv_score}")
print(f"In the Elastic Net model with normalised data , the alpha is {elastic_n_alpha}, the score is {elastic_n_score} and the cv score is {elastic_n_cv_score}")
print(f"In the Lars model with normalised data , the alpha is {lars_n_alpha}, the score is {lars_n_score} and the cv score is {lars_n_cv_score}")

print(f"In the Lasso model with standardised data, the alpha is {lasso_s_alpha}, the score is {lasso_s_score} and the cv score is {lasso_s_cv_score}")
print(f"In the Ridge model with standardised data, the alpha is {ridge_s_alpha}, the score is {ridge_s_score} and the cv score is {ridge_s_cv_score}")
print(f"In the Elastic Net model with standardised data, the alpha is {elastic_s_alpha}, the score is {elastic_s_score} and the cv score is {elastic_s_cv_score}")
print(f"In the Lars model with standardised data, the alpha is {lars_s_alpha}, the score is {lars_s_score} and the cv score is {lars_s_cv_score}")

'''
#for R1, lars_cv_prediction was used as i used cv=3 instead of cv=10 and that yielded better results for lars instead of ridge
# Since 3 cannot split the dataset equally, I changed my cv to 10 instead. 
def lars_cv_predict(train_x,train_y,test_x,cv):
    lars_cv = LassoLarsCV(cv=cv)
    lars_cv = lars_cv.fit(train_x,train_y)
    lars_score = lars_cv.score(train_x,train_y)
    lars_alpha = lars_cv.alpha_
    lars_cv_score = np.mean(cross_val_score(LassoLarsCV(), train_x, train_y, scoring='r2', cv=cv))
    print(lars_cv_score)
    pred_y = lars_cv.predict(test_x)
    pred_y =  pd.DataFrame(pred_y, columns=["Y"])
    return pred_y
pred_y = lars_cv_predict(train_n_x,train_y,test_n_x,cv)
pred_y.to_csv('A0216288X.csv',index=False)
''' 

#for R2, I used elastic cv as it has the best cv score for k = 10
def elastic_cv_predict(train_x,train_y,test_x,cv):
    elastic_cv = ElasticNetCV(cv=cv)
    elastic_cv = elastic_cv.fit(train_x,train_y)
    elastic_score = elastic_cv.score(train_x,train_y)
    elastic_alpha = elastic_cv.alpha_
    elastic_cv_score = np.mean(cross_val_score(ElasticNetCV(cv=cv), train_x, train_y, scoring='r2', cv=cv))
    #print(elastic_cv_score)
    pred_y = elastic_cv.predict(test_x)
    pred_y =  pd.DataFrame(pred_y, columns=["Y"])
    return pred_y

pred_y = elastic_cv_predict(train_n_x,train_y,test_n_x,cv)
pred_y.to_csv('A0216288X.csv',index=False)
print('The elastic net model on the 5 variables has been created. ')

