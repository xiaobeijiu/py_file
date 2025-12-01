import os
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import shap
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib import rcParams

matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False

data_dir = r"D:\1文件\py_file\筛选结果"
out_base = r"D:\1文件\py_file\图"

subdirs = {
    "violin": os.path.join(out_base, "小提琴图"),
    "bar": os.path.join(out_base, "特征重要性图"),
    "main": os.path.join(out_base, "主效应图"),
    "interaction": os.path.join(out_base, "交互效应图"),
    "metrics": os.path.join(out_base, "精度")
}
for p in subdirs.values():
    os.makedirs(p, exist_ok=True)

interaction_sample_size = 3000

def find_zero_crossings(x_fit, y_fit):
    crossings = []
    for i in range(1, len(y_fit)):
        if (y_fit[i-1] < 0 and y_fit[i] > 0) or (y_fit[i-1] > 0 and y_fit[i] < 0):
            try:
                crossing = fsolve(lambda x: np.interp(x, x_fit, y_fit), x0=x_fit[i], xtol=1e-4, maxfev=100)[0]
                crossings.append(crossing)
            except Exception:
                continue
    return crossings

def process_year(year, data_dir, subdirs):
    print(f"[{year}] 加载数据")
    file_path = os.path.join(data_dir, f"{year}.csv")
    data = pd.read_csv(file_path)
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    print(f"[{year}] 划分数据 X={X.shape} y={y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[{year}] 开始训练模型")
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=2000, max_depth=4, learning_rate=0.03, gamma=0, min_child_weight=3.6, subsample=0.56, colsample_bytree=0.8, reg_lambda=60, reg_alpha=0.1, random_state=1)
    model.fit(X_train, y_train)
    print(f"[{year}] 模型训练完成")
    explainer = shap.Explainer(model, X_train)
    shap_values_test = explainer(X_test)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics_path = os.path.join(subdirs["metrics"], f"{year}.txt")
    print(f"[{year}] 写入精度指标 -> {metrics_path}")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"训练集R2={r2_train:.6f}\n")
        f.write(f"测试集R2={r2_test:.6f}\n")
        f.write(f"训练集MAE={mae_train:.6f}\n")
        f.write(f"测试集MAE={mae_test:.6f}\n")
        f.write(f"训练集RMSE={rmse_train:.6f}\n")
        f.write(f"测试集RMSE={rmse_test:.6f}\n")
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    shap.plots.violin(shap_values_test, plot_type="layered_violin", show=False)
    violin_path = os.path.join(subdirs["violin"], f"{year}.png")
    plt.savefig(violin_path, dpi=400, bbox_inches='tight')
    print(f"[{year}] 小提琴图保存 -> {violin_path}")
    plt.close()
    shap.plots.bar(shap_values_test, show=False)
    bar_path = os.path.join(subdirs["bar"], f"{year}.png")
    plt.savefig(bar_path, dpi=400, bbox_inches='tight')
    print(f"[{year}] 特征重要性图保存 -> {bar_path}")
    plt.close()
    plt.style.use('default')
    rcParams.update({'font.family': 'Times New Roman', 'font.size': 10, 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
    tree_explainer = shap.TreeExplainer(model)
    X_interaction = X_train.sample(n=min(len(X_train), interaction_sample_size), random_state=42)
    shap_interaction_values = tree_explainer.shap_interaction_values(X_interaction)
    main_effects = np.array([shap_interaction_values[:, i, i] for i in range(shap_interaction_values.shape[1])]).T
    main_effects_df = pd.DataFrame(main_effects, columns=X_interaction.columns)
    num_rows, num_cols = 3, 5
    features = main_effects_df.columns[:num_rows * num_cols]
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 8), dpi=400)
    print(f"[{year}] 开始生成主效应图")
    for idx, feature in enumerate(features):
        row, col = divmod(idx, num_cols)
        ax = axes[row, col]
        ax.scatter(X_interaction[feature], main_effects_df[feature], s=10, c='#1f77b4', edgecolor='none', alpha=0.6)
        lowess_fit = lowess(main_effects_df[feature], X_interaction[feature], frac=0.4)
        x_fit = lowess_fit[:, 0]
        y_fit = lowess_fit[:, 1]
        ax.plot(x_fit, y_fit, color='#d62728', linewidth=1.5)
        x_intercepts = find_zero_crossings(x_fit, y_fit)
        for x_int in x_intercepts:
            ax.axvline(x=x_int, color='#2ca02c', linestyle='--', linewidth=1)
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel(feature, fontsize=13)
        ax.set_ylabel('SHAP main effect value', fontsize=13)
        ax.tick_params(axis='both', labelsize=10)
    for idx in range(len(features), num_rows * num_cols):
        row, col = divmod(idx, num_cols)
        fig.delaxes(axes[row, col])
    plt.tight_layout(pad=1.0)
    main_path = os.path.join(subdirs["main"], f"main_effects_grid_{year}.png")
    plt.savefig(main_path, dpi=400, bbox_inches='tight')
    print(f"[{year}] 主效应图保存 -> {main_path}")
    plt.close(fig)
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 12, 'axes.unicode_minus': False})
    feature_names = X_interaction.columns
    n_features = len(feature_names)
    print(f"[{year}] 开始生成交互效应图 共 {n_features*(n_features-1)//2} 张")
    interaction_year_dir = os.path.join(subdirs["interaction"], str(year))
    os.makedirs(interaction_year_dir, exist_ok=True)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            fig, ax = plt.subplots(figsize=(5.5, 4), dpi=400)
            shap_vals = shap_interaction_values[:, i, j] * 2
            x_data = X_interaction.iloc[:, i]
            color_data = X_interaction.iloc[:, j]
            vmin = np.percentile(color_data, 5)
            vmax = np.percentile(color_data, 95)
            sc = ax.scatter(x_data, shap_vals, c=color_data, cmap='coolwarm', vmin=vmin, vmax=vmax, s=10, alpha=0.6, edgecolor='none')
            cbar = plt.colorbar(sc, ax=ax, aspect=30, shrink=0.95, pad=0.03)
            cbar.set_label(feature_names[j], fontsize=13)
            cbar.ax.tick_params(labelsize=11)
            cbar.outline.set_visible(False)
            ax.set_xlabel(feature_names[i], fontsize=14)
            ax.set_ylabel(f"SHAP({feature_names[i]} × {feature_names[j]})", fontsize=14)
            ax.axhline(y=0, color='black', linestyle=':', linewidth=1)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.spines['right'].set_visible(False)
            pair_dir = os.path.join(interaction_year_dir, feature_names[i])
            os.makedirs(pair_dir, exist_ok=True)
            fname = os.path.join(pair_dir, f"SHAP_{feature_names[i]}_VS_{feature_names[j]}.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=400, bbox_inches='tight')
            plt.close(fig)
        gc.collect()
        print(f"[{year}] 交互效应图: 与 {feature_names[i]} 的 {n_features-i-1} 个配对完成 -> {os.path.join(interaction_year_dir, feature_names[i])}")
    print(f"[{year}] 交互效应图全部保存完成")
    del shap_interaction_values, main_effects_df, main_effects, X_interaction, shap_values_test
    del X_train, X_test, y_train, y_test, data, model, explainer
    plt.close('all')
    gc.collect()
    return year

if __name__ == "__main__":
    years = list(range(2006, 2021))
    for y in years:
        print(f"开始处理 {y} 年")
        try:
            process_year(y, data_dir, subdirs)
            print(f"完成 {y} 年")
        except Exception as e:
            print(f"{y} 年处理失败: {e}")
        plt.close('all')
        gc.collect()