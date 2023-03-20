corr_features=data.corrwith(data["PCOS (Y/N)"]).abs().sort_values(ascending=False)
#features with correlation more than 0.4
corr_features=corr_features[corr_features>0.4].index
corr_features
y=data2['PCOS (Y/N)']
X=data2.drop(['PCOS (Y/N)'], axis=1)
X_train,X_test,y_train, y_test=train_test_split(X,y, test_size=0.2)

import xgboost as xgb
xgb_cl = xgb.XGBClassifier(learning_rate = 0.001, gamma = 0.03, max_depth = 20,subsample = 0.5 )
xgb_cl.fit(X_train, y_train)

# Predict
preds = xgb_cl.predict(X_test)

# Score
accuracy_score(y_test, preds)
rfc =RandomForestClassifier(n_jobs=-1,n_estimators=150,max_features='sqrt',min_samples_leaf=10) #creates a Random forest model
rfc.fit(X_train, y_train) #trains model on data
pred_rfc = rfc.predict(X_test) #prediction

accuracy = accuracy_score(y_test, pred_rfc)
print(accuracy)

model=lgb.LGBMClassifier ()
model.fit(X_train,y_train)
print(f"Score in Train Data : {model.score(X_train,y_train)}")