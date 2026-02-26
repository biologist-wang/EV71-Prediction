import joblib
import pandas as pd


# 加载模型
package = joblib.load('EV71_Best_Model_Package.pkl')
model = package['model']
scaler = package['scaler']
features = package['features']


# 1. 准备数据：每一行代表一个新发现的病毒毒株
data = {
    'Accession': ['NEW_V1', 'NEW_V2', 'NEW_V3'], # 毒株编号
    'P2124_Z1': [1.45, -0.22, 0.88],           # 必须与 features 列表中的名称完全一致
    'P1246_Z3': [-0.10, 0.55, -0.34],
    'P997_Z2': [0.77, -0.12, 1.20],
    'P1711_Z1': [0.33, 0.44, 0.11],
    'P1743_Z2': [-0.55, 0.21, -0.90],
    'Country': ['China', 'USA', 'Thailand'],    # 其他元数据（模型会自动忽略，但保留可读性）
    'Year': [2025, 2024, 2025]
}

# 2. 实例化 DataFrame
new_df = pd.DataFrame(data)

# 3. 设置索引（可选，建议设为 Accession 方便查看结果）
new_df.set_index('Accession', inplace=True)


# new_df 是包含新病毒理化性质的 DataFrame
# 4. 提取特定位点
X_new = new_df[features]
# 5. 标准化（必须使用训练时的缩放参数）
X_scaled = scaler.transform(X_new.values)
# 6. 预测风险概率
risk_scores = model.predict_proba(X_scaled)[:, 1]
print(f"该毒株的神经毒力风险概率为: {risk_scores[0]:.2%}")
# 7. 整理预测结果表
results = new_df.copy()
results['Risk_Probability'] = risk_scores
results['Prediction'] = ['High Risk' if p > 0.5 else 'Low Risk' for p in risk_scores]

# 8. 格式化输出
print("\n--- 🔍 EV71 神经毒力预测报告 ---")
print(results[['Risk_Probability', 'Prediction']])

# 9. (可选) 导出结果
# results.to_csv('EV71_New_Samples_Predictions.csv', encoding='utf_8_sig')