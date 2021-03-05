import numpy as np
import pandas as pd
import lightgbm as lgb
import glob
import os

feature_dir = "../input/indoor-navigation-and-location-wifi-features/wifi_features"

# the metric used in this competition
def comp_metric(xhat, yhat, fhat, x, y, f):
    intermediate = np.sqrt(np.power(xhat - x,2) + np.power(yhat-y,2)) + 15 * np.abs(fhat-f)
    return intermediate.sum()/xhat.shape[0]

# get our train and test files
train_files = sorted(glob.glob(os.path.join(feature_dir, 'train/*_train.csv')))
test_files = sorted(glob.glob(os.path.join(feature_dir, 'test/*_test.csv')))
ssubm = pd.read_csv('../input/indoor-location-navigation/sample_submission.csv', index_col=0)

predictions = list()

for e, file in enumerate(train_files):
    data = pd.read_csv(file, index_col=0)
    test_data = pd.read_csv(test_files[e], index_col=0)

    x_train = data.iloc[:,:-4]
    y_trainy = data.iloc[:,-3]
    y_trainx = data.iloc[:,-4]
    y_trainf = data.iloc[:,-2]

    modely = lgb.LGBMRegressor(n_estimators=125, num_leaves=90)
    modely.fit(x_train, y_trainy)

    modelx = lgb.LGBMRegressor(n_estimators=125, num_leaves=90)
    modelx.fit(x_train, y_trainx)

    modelf = lgb.LGBMClassifier(n_estimators=125, num_leaves=90)
    modelf.fit(x_train, y_trainf)
    
    test_predsx = modelx.predict(test_data.iloc[:,:-1])
    test_predsy = modely.predict(test_data.iloc[:,:-1])
    test_predsf = modelf.predict(test_data.iloc[:,:-1])
    
    test_preds = pd.DataFrame(np.stack((test_predsf, test_predsx, test_predsy))).T
    test_preds.columns = ssubm.columns
    test_preds.index = test_data["site_path_timestamp"]
    test_preds["floor"] = test_preds["floor"].astype(int)
    predictions.append(test_preds)

all_preds = pd.concat(predictions)
all_preds = all_preds.reindex(ssubm.index)
all_preds.to_csv('submission.csv')