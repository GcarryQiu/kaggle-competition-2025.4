import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 加载数据
df = pd.read_csv("trainingdata.csv")
X = df.drop(columns=["y"])
y = df["y"]

# 划分训练集/验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用最佳参数训练模型
best_params = {
    'n_estimators': 300,
    'max_depth': None,
    'max_features': 0.5,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'bootstrap': True,
    'random_state': 42
}
model = RandomForestRegressor(**best_params)
model.fit(X_train, y_train)
val_preds = model.predict(X_val)

# 绘图：真实 vs 预测
plt.figure(figsize=(6, 6))
plt.scatter(y_val, val_preds, alpha=0.6, edgecolors='k')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')  # 理想预测线
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest on Validation Set")
plt.grid(True)
plt.tight_layout()
plt.savefig("rf_validation_plot.png", dpi=300)
plt.show()
