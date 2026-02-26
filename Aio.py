import pandas as pd
import numpy as np
import os
from scipy.stats import mannwhitneyu, chi2_contingency
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import shap
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import joblib


# 设置中文字体（根据系统环境可能需要调整，如 Windows 用 SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 核心配置：理化性质 Z-scales (Sandberg et al.)
# P1(P2124)->Z1, P2(P997)->Z2, P3(P1246)->Z3, P4(P1743)->Z2, P5(P1711)->Z1
# ==========================================
z_dict = {
    'A': [0.07, -1.73, 0.09], 'R': [2.88, 2.52, -3.44], 'N': [3.22, 1.45, 0.84],
    'D': [3.64, 1.13, 2.36], 'C': [0.71, -0.97, 4.15], 'Q': [2.18, 0.53, -1.14],
    'E': [3.08, 0.39, -0.07], 'G': [2.23, -5.36, 0.30], 'H': [2.41, 1.74, 1.11],
    'I': [-4.44, -1.68, -1.03], 'L': [-4.19, -1.03, -0.98], 'K': [2.84, 1.41, -3.14],
    'M': [-2.49, -0.27, -0.41], 'F': [-4.92, 1.30, 0.45], 'P': [-1.22, 0.88, 2.23],
    'S': [1.96, -1.63, 0.57], 'T': [0.92, -2.09, -1.40], 'W': [-4.75, 3.65, 0.85],
    'Y': [-1.39, 2.32, 0.01], 'V': [-2.69, -2.53, -1.29], '-': [0, 0, 0]
}

# ==========================================
# 数据清洗与年份修正
# ==========================================
def step1_prepare_master(file_path):
    df = pd.read_csv(file_path)

    # A. 临床标签判定
    cns_keys = ['CSF', 'BRAIN', 'CNS', 'ENCEPHALITIS', 'SPINAL', 'NERVE']
    df['Label'] = df['Tissue_Specimen_Source'].fillna('Unknown').str.upper().apply(
        lambda x: 1 if any(k in x for k in cns_keys) else 0
    )

    df['Year'] = pd.to_numeric(df['Collection_Date'], errors='coerce')
    if not df['Year'].isnull().all():
        df['Year'] = df['Year'].fillna(df['Year'].mode()[0]).astype(int)
    else:
        df['Year'] = 2012  # 最后的强制保底

    # C. 地理位置提取
    df['Country'] = df['Geo_Location'].fillna('Unknown').str.split(':').str[0].str.strip()

    # D. 氨基酸 Z-scales 编码
    std_len = df['aa'].apply(len).mode()[0]
    encoded = []
    for seq in df['aa']:
        vec = []
        for i in range(std_len):
            char = seq[i] if i < len(seq) else '-'
            vec.extend(z_dict.get(char, [0, 0, 0]))
        encoded.append(vec)

    feat_names = [f"P{i + 1}_Z{j}" for i in range(std_len) for j in [1, 2, 3]]
    X_matrix = pd.DataFrame(encoded, columns=feat_names)

    master_df = pd.concat([df[['Accession', 'Country', 'Year', 'Label']], X_matrix], axis=1)
    master_df.to_csv('EV71_Master_Dataset.csv', index=False, encoding='utf_8_sig')
    print(f"✅ Master 数据固化完成。CNS 样本: {sum(df['Label'] == 1)}, Non-CNS: {sum(df['Label'] == 0)}")
    return master_df

# ==========================================
# 核心统计分析与 Table 1 生成（基线表）
# ==========================================
def step2_generate_clinical_table(df, top_n=25):
    pos_df = df[df['Label'] == 1]
    neg_df = df[df['Label'] == 0]
    rows = []

    # --- 1. 地理分布 (卡方检验) ---
    ctab = pd.crosstab(df['Country'], df['Label'])
    _, p_country, _, _ = chi2_contingency(ctab)
    rows.append({
        '特征变量': '地理分布 (Country)', '统计描述方法': 'n (%)', '统计检验方法': '卡方检验',
        'CNS组 (n=7)': '-', 'Non-CNS组 (n=260)': '-', 'P值': f"{p_country:.4f}", 'OR (95% CI)': '-'
    })
    for c in sorted(df['Country'].unique()):
        n_p = sum(pos_df['Country'] == c)
        n_n = sum(neg_df['Country'] == c)
        rows.append({
            '特征变量': f"  - {c}", '统计描述方法': 'n (%)', '统计检验方法': '-',
            'CNS组 (n=7)': f"{n_p} ({n_p / 7:.1%})",
            'Non-CNS组 (n=260)': f"{n_n} ({n_n / 260:.1%})",
            'P值': '', 'OR (95% CI)': '-'
        })

    # --- 2. 采集年份 (Median + IQR) ---
    _, p_year = mannwhitneyu(pos_df['Year'], neg_df['Year'])
    rows.append({
        '特征变量': '采集年份', '统计描述方法': 'Median (IQR)', '统计检验方法': 'Mann-Whitney U',
        'CNS组 (n=7)': f"{pos_df['Year'].median():.0f} ({pos_df['Year'].quantile(0.25):.0f}-{pos_df['Year'].quantile(0.75):.0f})",
        'Non-CNS组 (n=260)': f"{neg_df['Year'].median():.0f} ({neg_df['Year'].quantile(0.25):.0f}-{neg_df['Year'].quantile(0.75):.0f})",
        'P值': f"{p_year:.4f}", 'OR (95% CI)': '-'
    })

    # --- 3. 理化位点分析 ---
    phys_cols = [c for c in df.columns if c.startswith('P')]
    p_results = []
    for col in phys_cols:
        _, p = mannwhitneyu(pos_df[col], neg_df[col])
        p_results.append({'feat': col, 'p': p})

    # 筛选最显著的 Top 位点
    top_feats = pd.DataFrame(p_results).sort_values('p').head(top_n)
    z_map = {'Z1': '疏水性', 'Z2': '分子量/体积', 'Z3': '极性/电荷'}

    for _, f_info in top_feats.iterrows():
        col = f_info['feat']
        p_val = f_info['p']

        # 描述统计：统一 Median (IQR)
        q1_p, q3_p = pos_df[col].quantile(0.25), pos_df[col].quantile(0.75)
        q1_n, q3_n = neg_df[col].quantile(0.25), neg_df[col].quantile(0.75)
        desc_pos = f"{pos_df[col].median():.2f} ({q1_p:.2f}-{q3_p:.2f})"
        desc_neg = f"{neg_df[col].median():.2f} ({q1_n:.2f}-{q3_n:.2f})"

        # 计算 OR (95% CI)
        try:
            X = sm.add_constant(df[col])
            logit_mod = sm.Logit(df['Label'], X).fit(disp=0)
            or_val = np.exp(logit_mod.params[1])
            conf = np.exp(logit_mod.conf_int().iloc[1])
            or_str = f"{or_val:.2f} ({conf[0]:.2f}-{conf[1]:.2f})"
        except:
            or_str = "N/A (样本极度不平衡)"

        z_type = col.split('_')[-1]
        rows.append({
            '特征变量': f"{col} ({z_map[z_type]})",
            '统计描述方法': 'Median (IQR)',
            '统计检验方法': 'Mann-Whitney U',
            'CNS组 (n=7)': desc_pos,
            'Non-CNS组 (n=260)': desc_neg,
            'P值': f"{p_val:.4e}",
            'OR (95% CI)': or_str
        })

    # 输出文件
    final_df = pd.DataFrame(rows)
    final_df.to_csv('Clinical_Adaptive_Table1.csv', index=False, encoding='utf_8_sig')
    print(f"✨ 最终统计表已生成: Clinical_Adaptive_Table1.csv")
    return top_feats['feat'].tolist()

# ==========================================
# 位点差异图生成
# ==========================================
def generate_forest_plot(
        input_table='Clinical_Adaptive_Table1.csv',
        top_n_plot=15,
        output_pdf='Neurovirulence_Forest_Plot.pdf'
):
    """仅负责森林图的解析与绘制"""
    print(f"📊 正在生成森林图: {output_pdf}...")
    table1 = pd.read_csv(input_table)

    # 1. 筛选并解析 OR 值
    plot_df = table1[table1['OR (95% CI)'].str.contains(r'\(', na=False)].copy()

    def parse_or(x):
        try:
            parts = x.split(' ')
            val = float(parts[0])
            ci_parts = parts[1].strip('()').split('-')
            return val, float(ci_parts[0]), float(ci_parts[1])
        except:
            return 1.0, 1.0, 1.0

    plot_df[['OR', 'Lower', 'Upper']] = plot_df['OR (95% CI)'].apply(lambda x: pd.Series(parse_or(x)))
    plot_df = plot_df.sort_values('P值').head(top_n_plot)

    # 2. 绘图
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(plot_df))

    # 绘制 CI 线与 OR 点
    plt.errorbar(plot_df['OR'], y_pos,
                 xerr=[plot_df['OR'] - plot_df['Lower'], plot_df['Upper'] - plot_df['OR']],
                 fmt='s', color='firebrick', ecolor='steelblue', capsize=4,
                 markersize=8, label='Odds Ratio (95% CI)')

    plt.axvline(x=1, color='black', linestyle='--', alpha=0.7)
    plt.yticks(y_pos, plot_df['特征变量'])
    plt.xlabel('比值比 (Odds Ratio) 与 95% 置信区间')
    plt.title(f'EV71 神经毒力关联位点森林图 (Top {top_n_plot})')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ 森林图已导出。")

# ==========================================
# 模型训练并导出
# ==========================================
def train_and_freeze_model(
        input_table='Clinical_Adaptive_Table1.csv',
        master_csv='Master_Dataset.csv',
        top_n_model=5,
        model_output_name='Combined_Predictive_Model.pkl',
        roc_pdf='Prediction_ROC_Curve.pdf'
):
    """负责模型训练、验证及持久化固化"""

    table1 = pd.read_csv(input_table)
    master_df = pd.read_csv(master_csv)

    # 1. 动态筛选特征 (从统计表 P 值最小的理化位点中选取)
    # 确保只选取以 'P' 开头的理化特征列
    physico_feats = table1[table1['特征变量'].str.contains('^P', regex=True)]
    top_cols = physico_feats.sort_values('P值')['特征变量'].str.split(' ').str[0].tolist()[:top_n_model]

    X = master_df[top_cols]
    y = master_df['Label']

    # 2. 训练模型
    lr_model = LogisticRegression(class_weight='balanced', solver='liblinear')
    lr_model.fit(X, y)

    # 3. 固化模型及其元数据 (包含特征顺序)
    model_payload = {
        'model': lr_model,
        'features': top_cols,
        'model_type': 'Logistic Regression',
        'n_samples': len(master_df)
    }
    joblib.dump(model_payload, model_output_name)
    print(f"💾 模型及特征元数据已固化至: {model_output_name}")

    # 4. 生成 ROC 评估图
    y_scores = lr_model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate')
    plt.title(f'Top {top_n_model} 特征模型预测性能')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.savefig(roc_pdf, bbox_inches='tight')
    plt.close()

    print(f"✅ ROC 曲线已导出: {roc_pdf} (AUC: {roc_auc:.3f})")
    return top_cols



def extract_model_formula_and_predict():

    # 1. 加载数据
    master_df = pd.read_csv('Master_Dataset.csv')
    table1 = pd.read_csv('Clinical_Adaptive_Table1.csv')

    # 2. 获取 Top 5 核心特征 (基于 P 值排名)
    # 过滤掉非理化特征行，只取 P 开头的特征
    physico_results = table1[table1['特征变量'].str.startswith('P', na=False)]
    top_5_features = physico_results.sort_values('P值')['特征变量'].str.split(' ').str[0].tolist()[:5]

    print(f"✅ 选定的核心建模指标: {top_5_features}")

    # 3. 构建多因素逻辑回归模型 (使用 statsmodels 以获取详细统计参数)
    X = master_df[top_5_features]
    y = master_df['Label']
    X_with_const = sm.add_constant(X)  # 添加常数项 (Intercept)

    model = sm.Logit(y, X_with_const).fit(disp=0)

    # 4. 导出权重系数表
    summary_df = pd.DataFrame({
        '特征位点': X_with_const.columns,
        '权重系数 (Beta)': model.params,
        'P值': model.pvalues,
        'OR值': np.exp(model.params)
    })
    summary_df.to_csv('Model_Coefficients_Weight.csv', index=False, encoding='utf_8_sig')

    # 5. 计算全样本预测概率
    # 使用训练好的模型给每个样本打分
    master_df['Risk_Probability'] = model.predict(X_with_const)

    # 6. 生成预测报告
    # 重点看 CNS 组的预测表现
    report_df = master_df[['Accession', 'Label', 'Year', 'Country', 'Risk_Probability']].copy()
    report_df = report_df.sort_values('Risk_Probability', ascending=False)

    # 保存预测清单
    report_df.to_csv('Sample_Risk_Predictions.csv', index=False, encoding='utf_8_sig')

    print("-" * 30)
    print("📈 模型数学公式预览 (Logit P = Σ Beta*X + Const):")
    for i, row in summary_df.iterrows():
        print(f"   [{row['特征位点']}] 权重: {row['权重系数 (Beta)']:+.4f}")

    print("-" * 30)
    # 检查那 7 个 CNS 样本的平均预测得分
    cns_avg = report_df[report_df['Label'] == 1]['Risk_Probability'].mean()
    non_cns_avg = report_df[report_df['Label'] == 0]['Risk_Probability'].mean()
    print(f"📊 CNS组平均风险得分: {cns_avg:.2%}")
    print(f"📊 Non-CNS组平均风险得分: {non_cns_avg:.2%}")
    print("✨ 预测清单已保存至: Sample_Risk_Predictions.csv")


# ==========================================
# 核心位点理化景观热图
# ==========================================
def plot_physicochemical_heatmap():

    # 1. 加载数据
    df = pd.read_csv('Master_Dataset.csv')
    top_features = ['P2124_Z1', 'P1246_Z3', 'P997_Z2', 'P1711_Z1', 'P1743_Z2']

    # 2. 准备绘图数据：提取 7 个 CNS 样本和 随机 20 个 Non-CNS 样本进行对比
    cns_samples = df[df['Label'] == 1]
    non_cns_samples = df[df['Label'] == 0].sample(20, random_state=42)
    plot_data = pd.concat([cns_samples, non_cns_samples])

    # 3. 数据标准化 (Z-score 归一化，使不同性质的 Z-scales 具有可比性)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(plot_data[top_features])
    plot_df_scaled = pd.DataFrame(scaled_values, columns=top_features)
    plot_df_scaled['Group'] = ['CNS'] * len(cns_samples) + ['Non-CNS'] * len(non_cns_samples)

    # 4. 绘图
    plt.figure(figsize=(12, 8))
    # 设置侧边颜色条，区分组别
    group_colors = plot_df_scaled['Group'].map({'CNS': 'firebrick', 'Non-CNS': 'dodgerblue'})

    g = sns.clustermap(
        plot_df_scaled[top_features],
        cmap='RdYlBu_r',  # 红蓝色调，红色代表理化分值高，蓝色代表低
        row_colors=group_colors,
        yticklabels=False,
        linewidths=.5,
        cbar_pos=(0.02, 0.8, 0.03, 0.15)
    )

    plt.title('EV71 神经毒力核心位点理化景观热图', pad=100)
    g.savefig('Physicochemical_Heatmap.pdf', dpi=300, bbox_inches='tight')
    print("✅ 热图已保存：Physicochemical_Heatmap")

# ==========================================
# 留一法交叉验证验证
# ==========================================
def run_loocv_validation():

    # 1. 加载数据
    # 确保之前已经运行过 step1 生成了 master 数据集
    df = pd.read_csv('Master_Dataset.csv')

    # 2. 定义建模特征 (使用你之前选定的 Top 5)
    features = ['P2124_Z1', 'P1246_Z3', 'P997_Z2', 'P1711_Z1', 'P1743_Z2']
    X = df[features].values
    y = df['Label'].values

    # 3. 初始化验证环境
    loo = LeaveOneOut()
    y_true = []
    y_probs = []

    # 初始化模型（使用 class_weight='balanced' 应对样本不平衡）
    lr = LogisticRegression(class_weight='balanced', solver='liblinear')

    # 4. 执行交叉验证循环
    # 每次留出一个样本作为测试，其余建模
    count = 0
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 训练模型
        lr.fit(X_train, y_train)

        # 预测被留出的那个样本属于 CNS 的概率
        prob = lr.predict_proba(X_test)[:, 1]

        y_true.append(y_test[0])
        y_probs.append(prob[0])

        count += 1
        if count % 50 == 0:
            print(f"已完成 {count}/267 个样本的轮转验证...")

    # 5. 计算验证后的 ROC 和 AUC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    cv_auc = auc(fpr, tpr)

    # 6. 绘图：验证后的 ROC 曲线
    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'LOOCV ROC (AUC = {cv_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (假阳性率)')
    plt.ylabel('True Positive Rate (真阳性率)')
    plt.title('留一交叉验证 (LOOCV) ROC 曲线\n(评估模型的泛化能力)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.savefig('LOOCV_Validation_ROC.pdf', dpi=300, bbox_inches='tight')
    print(f"\n✅ 验证完成！")
    print(f"📊 交叉验证后的 AUC 为: {cv_auc:.3f}")

    # 7. 保存验证后的预测打分，用于分析哪些样本被“算错”了
    val_results = pd.DataFrame({
        'Accession': df['Accession'],
        'Actual_Label': y_true,
        'Predicted_Prob': y_probs
    })
    val_results.to_csv('LOOCV_Prediction_Results.csv', index=False)

# ==========================================
# 模型对比并保存表现最好模型
# ==========================================
def compare_eight_models_and_save_best(
        master_csv='Master_Dataset.csv',
        model_output_path='./model/Best_Model_Package.pkl'
):


    # 1. 数据准备
    df = pd.read_csv(master_csv)
    features = ['P2124_Z1', 'P1246_Z3', 'P997_Z2', 'P1711_Z1', 'P1743_Z2']

    # 核心：必须保存这个 scaler，否则模型无法在其他数据集上使用
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].values)
    y = df['Label'].values

    # 2. 定义 8 种模型
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', solver='liblinear'),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'SVM (RBF)': SVC(probability=True, class_weight='balanced', kernel='rbf'),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'Ridge Classifier': RidgeClassifier(class_weight='balanced'),
        'Gaussian Naive Bayes': GaussianNB()
    }

    plt.figure(figsize=(12, 9))
    loo = LeaveOneOut()
    results = []

    best_auc = 0
    best_model_name = ""

    # 3. 核心循环：LOOCV 验证
    for name, model in models.items():
        print(f"正在测试算法: {name}...")
        y_true, y_probs = [], []

        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)

            if name == 'Ridge Classifier':
                d = model.decision_function(X_test)
                prob = 1 / (1 + np.exp(-d))
            else:
                prob = model.predict_proba(X_test)[:, 1]

            y_true.append(y_test[0])
            y_probs.append(prob[0])

        # 计算并绘图
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        results.append({'Model': name, 'AUC': roc_auc})
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

        # 记录表现最好的模型
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model_name = name

    # 4. 图表修饰
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate')
    plt.title('8 种模型预测性能对比 (LOOCV)')
    plt.legend(loc="lower right", fontsize='small', ncol=2)
    plt.savefig('Eight_Models_Comparison_ROC.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 重新训练冠军模型并固化导出
    print(f"\n🏆 冠军模型确认: {best_model_name} (AUC: {best_auc:.4f})")

    # 使用全部数据重新训练冠军算法
    final_best_model = models[best_model_name]
    final_best_model.fit(X, y)

    # 导出“预测全家桶”
    best_package = {
        'model_name': best_model_name,
        'model': final_best_model,
        'scaler': scaler,  # 预测新序列时必须先用它缩放数据
        'features': features,  # 记录特征顺序
        'auc_score': best_auc
    }
    joblib.dump(best_package, model_output_path)

    # 输出排名报告
    report = pd.DataFrame(results).sort_values('AUC', ascending=False)
    report.to_csv('Eight_Models_Ranking.csv', index=False)

    print(f"💾 冠军模型全家桶已导出至: {model_output_path}")
    return report

# ==========================================
# shap
# ==========================================
def generate_shap_analysis():
    # 1. 加载数据
    df = pd.read_csv('Master_Dataset.csv')
    features = ['P2124_Z1', 'P1246_Z3', 'P997_Z2', 'P1711_Z1', 'P1743_Z2']
    X = df[features]
    y = df['Label']

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

    # 2. 重新训练你的冠军模型 (Logistic Regression)
    model = LogisticRegression(class_weight='balanced', solver='liblinear')
    model.fit(X_scaled, y)

    # 3. 创建 SHAP 解释器
    # 对于逻辑回归，我们使用 LinearExplainer
    explainer = shap.LinearExplainer(model, X_scaled)
    shap_values = explainer.shap_values(X_scaled)

    # 4. 绘制 SHAP Summary Plot (条形图：展示整体特征重要性)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_scaled, plot_type="bar", show=False)
    plt.title('核心位点对神经毒力预测的贡献度排名 (SHAP Importance)')
    plt.tight_layout()
    plt.savefig('SHAP_Feature_Importance.pdf', dpi=300, bbox_inches='tight')

    # 5. 绘制 SHAP Summary Plot (散点图：展示特征取值高低对风险的影响)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_scaled, show=False)
    plt.title('位点理化值分布对风险的影响分析 (SHAP Summary)')
    plt.tight_layout()
    plt.savefig('SHAP_Summary_Distribution.pdf', dpi=300, bbox_inches='tight')

# ==========================================
# 列线图
# ==========================================
def generate_nomogram_data():
    # 1. 定义模型参数
    intercept = -7.5129
    weights = {
        'P2124_Z1': 1.1684,
        'P997_Z2': 0.9244,
        'P1743_Z2': 0.2431,
        'P1711_Z1': 0.0728,
        'P1246_Z3': -0.2441
    }

    # 设置中文支持与全局字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 计算最大波动范围以归一化 Points
    max_impact = max([abs(w) * 6 for w in weights.values()])

    # 增加画布高度，避免纵向拥挤
    fig, ax = plt.subplots(figsize=(14, 10))

    # ---------------------------------------------------------
    # 2. 绘制顶部 Point 轴 (参考基准)
    # ---------------------------------------------------------
    y_base = 10
    ax.hlines(y_base, 0, 100, colors='black', lw=1.5)
    # 微调点：使用 bbox 增加标签可读性，调整横向偏移
    ax.text(-2, y_base, '单项评分 (Points)', fontweight='bold', ha='right', va='center', fontsize=12)

    for x in range(0, 101, 10):
        ax.vlines(x, y_base, y_base + 0.2, colors='black')
        ax.text(x, y_base + 0.4, str(x), ha='center', fontsize=10)

    # ---------------------------------------------------------
    # 3. 绘制各个特征轴
    # ---------------------------------------------------------
    y_pos = 8.5  # 起始位置
    for feat, weight in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
        ax.hlines(y_pos, 0, 100, colors='lightgray', linestyle='--', alpha=0.6)

        ax.text(-2, y_pos, f"{feat}", fontweight='bold', ha='right', va='center')

        points_per_unit = (abs(weight) * 6) / max_impact * 100

        # 标注刻度
        ticks = [-3, -2, -1, 0, 1, 2, 3]
        for val in ticks:
            # 逻辑微调：如果权重为负，刻度值从小到大应对应分值从大到小
            if weight > 0:
                p = (val + 3) / 6 * points_per_unit
            else:
                p = (3 - val) / 6 * points_per_unit

            ax.vlines(p, y_pos, y_pos + 0.15, colors='navy', lw=1)
            # 仅在整数点标注文字，避免拥挤
            ax.text(p, y_pos - 0.4, f"{val}", fontsize=9, ha='center', color='#333333')

        y_pos -= 1.4  # 增加行间距

    # ---------------------------------------------------------
    # 4. 底部总分与概率轴
    # ---------------------------------------------------------
    y_total = 0
    ax.hlines(y_total, 0, 100, colors='black', lw=2)
    ax.text(-2, y_total, '总评分 (Total Points)', fontweight='bold', ha='right', color='black', fontsize=12)
    for x in range(0, 101, 20):
        ax.vlines(x, y_total, y_total + 0.3, colors='black')
        ax.text(x, y_total - 0.5, str(x * 4), ha='center')  # 假设总分为单项分累加映射

    y_prob = -2
    ax.hlines(y_prob, 0, 100, colors='darkred', lw=2)
    ax.text(-2, y_prob, '神经毒力风险概率', fontweight='bold', ha='right', color='darkred', fontsize=12)

    # 概率刻度非线性映射微调 (示意)
    prob_ticks = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    for prob in prob_ticks:
        x_pos = prob * 100
        ax.vlines(x_pos, y_prob, y_prob + 0.3, colors='darkred')
        # 旋转概率标签，防止重叠
        ax.text(x_pos, y_prob - 0.5, f"{prob:.0%}", ha='center', fontsize=9, rotation=0)

    # 5. 修饰与保存
    ax.set_ylim(-4, 12)
    ax.set_xlim(-15, 110)  # 扩大左边距留给标签
    ax.axis('off')

    plt.title('EV71 神经毒力风险预测列线图 (理化特征模型)', fontsize=16, pad=30)

    # 增加底部注释说明
    plt.figtext(0.5, 0.05, "注：-3 至 3 代表位点 Z-scale 值的波动范围；Points 代表该特征对毒力的贡献得分。",
                ha="center", fontsize=10, style='italic', color='gray')

    plt.savefig('Nomogram_Risk_Prediction_Refined.pdf', bbox_inches='tight')
    plt.close()
    print("✨ 列线图已生成。")

# ==========================================
#     执行逻辑
# ==========================================
if __name__ == "__main__":
    # # ==========================================
    # # 1. 初始化与数据准备
    # # ==========================================
    # source_csv = './data/training_data.csv'
    #
    # # 如果需要重新清洗数据，取消下面代码的注释
    # """
    # if os.path.exists('Master_Dataset.csv'):
    #     os.remove('Master_Dataset.csv')
    #
    # if os.path.exists(source_csv):
    #     df_master = step1_prepare_master(source_csv)
    #     selected_vars = step2_generate_clinical_table(df_master)
    # else:
    #     print(f"❌ 找不到原始数据文件: {source_csv}")
    # """
    #
    # # ==========================================
    # # 2. 核心分析流程 (已对齐)
    # # ==========================================
    #
    # # 森林图生成
    # generate_forest_plot(
    #     input_table='Clinical_Adaptive_Table1.csv',
    #     top_n_plot=15,
    #     output_pdf='Neurovirulence_Forest_Plot.pdf'
    # )
    #
    # # 训练并导出模型 (包含 ROC 曲线)
    # train_and_freeze_model(
    #     input_table='Clinical_Adaptive_Table1.csv',
    #     master_csv='EV71_Master_Dataset.csv',
    #     top_n_model=5,
    #     model_output_name='Combined_Predictive_Model.pkl',
    #     roc_pdf='Prediction_ROC_Curve.pdf'
    # )
    #
    # # 八模型对比并保存冠军全家桶
    # compare_eight_models_and_save_best(
    #     master_csv='Master_Dataset.csv',
    #     model_output_path='EV71_Best_Model_Package.pkl'
    # )
    #
    # # 理化景观热图
    # plot_physicochemical_heatmap()
    #
    # # 留一法交叉验证 (LOOCV)
    # run_loocv_validation()
    #
    # # SHAP 解释性分析
    # generate_shap_analysis()

    # 列线图生成 (Nomogram)
    # generate_nomogram_data()

