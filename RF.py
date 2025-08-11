import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# 加载数据
df = pd.read_csv("trainingdata.csv")
X = df.drop(columns=["y"])
y = df["y"]

# 划分训练集与验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 设定包含默认参数的网格
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt', 0.5],  # 包含默认'sqrt'
    'bootstrap': [True]
}

# 进行GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

# 手动验证每组参数在验证集上的RMSE
results = []
for i, params in enumerate(grid_search.cv_results_['params']):
    model = RandomForestRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds) ** 0.5
    results.append((params, rmse))
    print(f"Model {i+1} - RMSE: {rmse:.5f} - Params: {params}")

# 输出最优组合
best = min(results, key=lambda x: x[1])
print("\n最佳参数组合:")
print(best[0])
print(f"验证集 RMSE: {best[1]:.5f}")

