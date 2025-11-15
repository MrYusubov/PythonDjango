import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
np.random.seed(42)

n = 1000
Age = np.random.randint(20,66,size=n)
EducationLevel = np.random.randint(1,6,size=n)
Experience_years = np.clip((Age - 20) * np.random.rand(n), 0, 40).astype(int)
WeeklyHours = np.random.randint(20,61,size=n)
CityCostIndex = np.random.randint(1,11,size=n)
noise = np.random.normal(0, 1000, size=n)
Income_AZN = (EducationLevel * 600) + (Experience_years * 250) + (WeeklyHours * 50) + (CityCostIndex * 300) + noise
df = pd.DataFrame({
    "Age": Age,
    "EducationLevel": EducationLevel,
    "Experience_years": Experience_years,
    "WeeklyHours": WeeklyHours,
    "CityCostIndex": CityCostIndex,
    "Income_AZN": Income_AZN
})
for col in df.columns:
    df.loc[df.sample(frac=0.02, random_state=col.__hash__()%100).index, col] = np.nan
X = df[["Age","EducationLevel","Experience_years","WeeklyHours","CityCostIndex"]].copy()
y = df["Income_AZN"].copy()
X = X.fillna(X.mean())
y = y.fillna(y.mean())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
r2_train_lr = r2_score(y_train, y_train_pred)
r2_test_lr = r2_score(y_test, y_test_pred)
print("Linear Regression R2 Train:", r2_train_lr)
print("Linear Regression R2 Test:", r2_test_lr)
if r2_test_lr < 0.7:
    print("R2 < 0.7 səbəbi: model çox sadə ola bilər və qeyri-xətti münasibətləri tutmaya bilər.")

cv_scores = cross_val_score(lr, X, y, cv=5, scoring="r2")
print("Linear Regression 5-fold CV R2:", cv_scores)
print("Linear Regression 5-fold CV R2 mean:", cv_scores.mean())

poly_results = {}
for degree in [2,3]:
    pf = PolynomialFeatures(degree=degree, include_bias=False)
    Xp = pf.fit_transform(X)
    Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(Xp_train, yp_train)
    yp_train_pred = model.predict(Xp_train)
    yp_test_pred = model.predict(Xp_test)
    r2_train = r2_score(yp_train, yp_train_pred)
    r2_test = r2_score(yp_test, yp_test_pred)
    poly_results[degree] = (r2_train, r2_test)
    print(f"Poly degree {degree} R2 Train:", r2_train)
    print(f"Poly degree {degree} R2 Test:", r2_test)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=10)
ridge.fit(X_train_s, y_train_s)
y_tr_ridge = ridge.predict(X_train_s)
y_te_ridge = ridge.predict(X_test_s)
r2_tr_ridge = r2_score(y_train_s, y_tr_ridge)
r2_te_ridge = r2_score(y_test_s, y_te_ridge)

lasso = Lasso(alpha=1000, max_iter=10000)
lasso.fit(X_train_s, y_train_s)
y_tr_lasso = lasso.predict(X_train_s)
y_te_lasso = lasso.predict(X_test_s)
r2_tr_lasso = r2_score(y_train_s, y_tr_lasso)
r2_te_lasso = r2_score(y_test_s, y_te_lasso)

print("Ridge Train R2:", r2_tr_ridge)
print("Ridge Test R2:", r2_te_ridge)
print("Lasso Train R2:", r2_tr_lasso)
print("Lasso Test R2:", r2_te_lasso)

print("\nModel\tTrain R2\tTest R2")
print(f"Linear\t{r2_train_lr:.4f}\t{r2_test_lr:.4f}")
print(f"Poly(2)\t{poly_results[2][0]:.4f}\t{poly_results[2][1]:.4f}")
print(f"Poly(3)\t{poly_results[3][0]:.4f}\t{poly_results[3][1]:.4f}")
print(f"Ridge\t{r2_tr_ridge:.4f}\t{r2_te_ridge:.4f}")
print(f"Lasso\t{r2_tr_lasso:.4f}\t{r2_te_lasso:.4f}")

degrees = list(range(1,6))
train_errors = []
test_errors = []
for d in degrees:
    pf = PolynomialFeatures(degree=d, include_bias=False)
    Xp = pf.fit_transform(X)
    Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(Xp_train, yp_train)
    yp_tr_pred = model.predict(Xp_train)
    yp_te_pred = model.predict(Xp_test)
    train_mse = mean_squared_error(yp_train, yp_tr_pred)
    test_mse = mean_squared_error(yp_test, yp_te_pred)
    train_errors.append(train_mse)
    test_errors.append(test_mse)

plt.plot(degrees, train_errors, label="Train Error", marker="o")
plt.plot(degrees, test_errors, label="Test Error", marker="o")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.legend()
plt.show()

min_test_idx = np.argmin(test_errors)
print("Test error ən aşağı olduğu degree:", degrees[min_test_idx])
for i, d in enumerate(degrees):
    print(f"Degree {d} Train MSE: {train_errors[i]:.2f} Test MSE: {test_errors[i]:.2f}")

print("\nYekun analiz:")
print("1) Overfitting modelin train məlumatında yaxşı performans verib testdə pis nəticə verməsidir; underfitting həm train həm testdə pis nəticədir.")
print("2) Polynomial regression qeyri-xətti münasibətləri tutmaq üçün faydalıdır və real datada daha uyğun funksiyalar verə bilər.")
print("3) L1 (Lasso) bəzi əmsalları sıfıra vuraraq xüsusiyyət seçimi edir; L2 (Ridge) əmsalları kiçildir amma sıfıra vurmur, hər ikisi overfittingi azaltmağa kömək edir.")
print("4) Ən balanslı model adətən Ridge oldu; o, regularizasiya ilə generalizasiya gücünü artırdı.")
