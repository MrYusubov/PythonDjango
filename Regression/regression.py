import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Dataset yarat
np.random.seed(42)

likes = np.random.randint(50, 1000, 100)
comments = np.random.randint(5, 200, 100)
views = likes * 20 + comments * 15 + np.random.randint(1000, 5000, 100)

df = pd.DataFrame({
    "Likes": likes,
    "Comments": comments,
    "Views": views
})

df.to_excel("youtube_data.xlsx", index=False)
print("Dataset:")
print(df.head())

# 2. Data Analizi

print("\nDescriptive Statistics:")
print(df.describe())

print("\nCorrelation Matrix:")
print(df.corr())

print("\nViews-a ən çox təsir edən dəyişən: Likes")

# 3. Linear Regression

X = df[["Likes", "Comments"]]
y = df["Views"]

model = LinearRegression()
model.fit(X, y)

print("\nLinear Regression Nəticələri:")
print("Coef (a, b):", model.coef_)
print("Intercept (c):", model.intercept_)

if model.coef_[0] > model.coef_[1]:
    print("Likes Views-a daha güclü təsir edir")
else:
    print("Comments Views-a daha güclü təsir edir")

# 4. Gradient Descent (1 input: Likes)

m = 0
b = 0
L = 0.0001
epochs = 1000
errors = []

for i in range(epochs):
    y_pred = m * df["Likes"] + b
    error = ((df["Views"] - y_pred) ** 2).mean()
    errors.append(error)

    dm = -2 * (df["Likes"] * (df["Views"] - y_pred)).mean()
    db = -2 * (df["Views"] - y_pred).mean()

    m -= L * dm
    b -= L * db

    if i % 100 == 0:
        print(f"Iteration {i}, Error = {error}")

print("\nGradient Descent nəticələri:")
print("m =", m)
print("b =", b)


# 5. Error qrafiki

plt.plot(errors)
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("Gradient Descent Error Azalması")
plt.show()


# 6. Təxmin et (Linear Regression model ilə)

like_input = int(input("Like sayını daxil et: "))
comment_input = int(input("Comment sayını daxil et: "))

predicted_views = model.predict([[like_input, comment_input]])
print(f"📈 Təxmin edilən Views: {int(predicted_views[0])}")

# 7. Metriklər

y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\nMetriklər:")
print("MAE:", mae)
print("MSE:", mse)
print("R²:", r2)
