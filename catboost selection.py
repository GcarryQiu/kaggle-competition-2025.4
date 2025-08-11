import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import itertools

# 读取数据
df = pd.read_csv("trainingdata.csv")
X = df.drop(columns=["y"])
y = df["y"]

# 用RF筛选累计重要性≤95%的特征
rf_temp = RandomForestRegressor(random_state=42)
rf_temp.fit(X, y)
importances = rf_temp.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)
importance_df["Cumulative"] = importance_df["Importance"].cumsum()
selected_features = importance_df[importance_df["Cumulative"] <= 0.95]["Feature"].tolist()
X_selected = X[selected_features]

# 划分训练与验证集(8:2)
X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 自定义参数
depth_list = [4, 5, 6]
learning_rate_list = [0.01, 0.03, 0.05, 0.07]
l2_leaf_reg_list = [1, 3, 5]

param_grid = list(itertools.product(depth_list, learning_rate_list, l2_leaf_reg_list))

results = []
for depth, lr, l2 in param_grid:
    # 动态调整迭代
    if lr == 0.01:
        iters = 3000
    elif lr == 0.03:
        iters = 1500
    elif lr == 0.05:
        iters = 1000
    else:  # lr == 0.07
        iters = 700

    model = CatBoostRegressor(
        iterations=iters,
        depth=depth,
        learning_rate=lr,
        l2_leaf_reg=l2,
        loss_function='RMSE',
        verbose=0,
        random_seed=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds) ** 0.5
    results.append(((depth, lr, l2, iters), rmse))

    print(f"depth={depth}, lr={lr}, l2={l2}, iters={iters} RMSE: {rmse:.5f}")

# 输出最优组合
best = min(results, key=lambda x: x[1])
print("\n最优参数组合:")
print(f"depth={best[0][0]}, learning_rate={best[0][1]}, l2_leaf_reg={best[0][2]}, iterations={best[0][3]}")
print(f"验证集 RMSE: {best[1]:.5f}")

