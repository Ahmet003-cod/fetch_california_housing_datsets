from sklearn.datasets import fetch_california_housing
import pandas as pd

df=fetch_california_housing(as_frame=True)
type(df)

df.columns

df.isnull().count().sum
df.shape
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X=df.drop("MedHouseVal",axis=1)
y=df["MedHouseVal"]

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


scaler=StandardScaler()



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train_scaler=scaler.fit_transform(X_train)
X_test_scaler=scaler.transform(X_test)



# Modeli oluştur ve eğit
LR = LinearRegression()
LR.fit(X_train_scaler, y_train)

# Test verisiyle tahmin yap
y_pred = LR.predict(X_test_scaler)

# Modelin başarımını ölç
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression MSE: {mse:.4f}")
print(f"Linear Regression R²: {r2:.4f}")

from sklearn.linear_model import Lasso

lasso=Lasso(alpha=0.1)
lasso.fit(X_train_scaler,y_train)
y_pred=lasso.predict(X_test_scaler)

print("Lasso MSE=",mean_squared_error(y_test,y_pred))
print("Lasso r2_score=",r2_score(y_test,y_pred))

from sklearn.linear_model import Ridge
for i in [0.01,0.1,1,10,100]:
    ridge=Ridge(alpha=i)
    ridge.fit(X_train_scaler,y_train)
    y_pred=ridge.predict(X_test_scaler)
    print(f"Ridge alpha={i} MSE=",mean_squared_error(y_test,y_pred))
    print(f"Ridge alpha={i} r2_score=",r2_score(y_test,y_pred))

from  sklearn.tree import DecisionTreeRegressor

DTR=DecisionTreeRegressor(random_state=42)
DTR.fit(X_train_scaler,y_train)
y_pred=DTR.predict(X_test_scaler)

print("DecisionTreeRegressor MSE=",mean_squared_error(y_test,y_pred))
print("DecisionTreeRegressor r2_score=",r2_score(y_test,y_pred))

from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(n_estimators=100,random_state=42)
rfr.fit(X_train_scaler,y_train)
y_pred=rfr.predict(X_test_scaler)

print("RandomForestRegressor MSE=",mean_squared_error(y_test,y_pred))
print("RandomForestRegressor r2_score=",r2_score(y_test,y_pred))


from xgboost import XGBRegressor

xgb_model=XGBRegressor()
xgb_model.fit(X_train_scaler,y_train)
y_pred=xgb_model.predict(X_test_scaler)

print("XGBRegressor MSE=",mean_squared_error(y_test,y_pred))
print("XGBRegressor r2_score=",r2_score(y_test,y_pred))


from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaler, y_train)
y_pred_knn = knn_model.predict(X_test_scaler)

print("KNN Regressor MSE:", mean_squared_error(y_test, y_pred_knn))
print("KNN Regressor R²:", r2_score(y_test, y_pred_knn))


from sklearn.svm import SVR

svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaler, y_train)
y_pred_svr = svr_model.predict(X_test_scaler)

print("SVR MSE:", mean_squared_error(y_test, y_pred_svr))
print("SVR R²:", r2_score(y_test, y_pred_svr))

import matplotlib.pyplot as plt

models = ['XGBoost','Random Forest','SVR', 'KNN','Decision Tree','Ridge','Linear Regression', 'Lasso']
r2_scores = [0.8301, 0.8053, 0.7276, 0.6700, 0.6230, 0.5778, 0.5758, 0.4814]

plt.figure(figsize=(12,6))
bars = plt.bar(models, r2_scores)
color=["green","blue","skyblue","Yellow","orange","brown","brown","red"]
j=0
for i in color:
    bars[j].set_color(i)
    j=j+1

plt.ylabel('R² Score')
plt.title('Model Performans Karşılaştırması (R²)')

# Barların üstüne yüzdelik R² değerini yazdırma
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{score:.2%}", ha='center', fontsize=10)

plt.ylim(0, 1)
plt.show()