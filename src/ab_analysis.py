"""
A/B 测试分析模块.

数据清洗 -> 描述统计 -> 假设检验 -> 置信区间 -> 效应量 -> 样本量计算.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')


def load_data(data_path="../data/raw/ab_data.csv"):
    """加载 A/B 测试数据."""
    df = pd.read_csv(data_path)
    print(f"数据形状: {df.shape}")
    print(df.head())
    return df


def clean_data(df):
    """数据清洗: 处理不匹配行和重复用户."""
    n_before = len(df)

    # 1. 检查 group 和 landing_page 是否匹配
    #    control 应该对应 old_page, treatment 应该对应 new_page
    mismatch = df[((df['group'] == 'control') & (df['landing_page'] == 'new_page')) |
                  ((df['group'] == 'treatment') & (df['landing_page'] == 'old_page'))]
    print(f"\n不匹配行数: {len(mismatch)} (control看了新页面 或 treatment看了旧页面)")

    # 删除不匹配行
    df_clean = df[~df.index.isin(mismatch.index)].copy()

    # 2. 检查重复用户（同一 user_id 出现多次）
    dup_count = df_clean['user_id'].duplicated().sum()
    print(f"重复用户数: {dup_count}")

    # 删除重复用户，只保留第一次出现
    df_clean = df_clean.drop_duplicates(subset='user_id', keep='first')

    n_after = len(df_clean)
    print(f"清洗前: {n_before} → 清洗后: {n_after} (删除 {n_before - n_after} 行)")

    return df_clean


def descriptive_stats(df):
    """描述统计: 两组转化率对比."""
    print("\n=== 描述统计 ===")

    # 按组别统计
    summary = df.groupby('group').agg(
        用户数=('user_id', 'count'),
        转化数=('converted', 'sum'),
        转化率=('converted', 'mean'),
    )
    summary['转化率'] = summary['转化率'] * 100
    summary['转化率'] = summary['转化率'].round(4)
    print(summary)

    control_cr = df[df['group'] == 'control']['converted'].mean()
    treatment_cr = df[df['group'] == 'treatment']['converted'].mean()
    diff = (treatment_cr - control_cr) * 100

    print(f"\n对照组转化率: {control_cr*100:.4f}%")
    print(f"实验组转化率: {treatment_cr*100:.4f}%")
    print(f"差异: {diff:.4f}% (实验组 - 对照组)")

    if diff > 0:
        print(f"新页面转化率比旧页面高 {abs(diff):.4f} 个百分点")
    else:
        print(f"新页面转化率比旧页面低 {abs(diff):.4f} 个百分点")

    return control_cr, treatment_cr


def plot_conversion_rates(control_cr, treatment_cr, save_path="../images/"):
    """可视化: 两组转化率对比柱状图."""
    import os
    os.makedirs(save_path, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    groups = ['对照组\n(旧页面)', '实验组\n(新页面)']
    rates = [control_cr * 100, treatment_cr * 100]
    colors = ['#4CAF50', '#2196F3']

    bars = ax.bar(groups, rates, color=colors, width=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rate:.2f}%", ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_ylabel('转化率 (%)', fontsize=12)
    ax.set_title('A/B 测试：两组转化率对比', fontsize=14)
    ax.set_ylim(0, max(rates) * 1.3)
    plt.tight_layout()
    plt.savefig(save_path + "conversion_rates.png", dpi=150, bbox_inches='tight')
    print("转化率对比图已保存")
    plt.close()


def hypothesis_test(control_cr, treatment_cr, n_control, n_treatment):
    """假设检验: 双比例 z 检验."""
    from scipy import stats

    print("\n=== 假设检验：双比例 z 检验 ===")

    # 原假设 H0: 两个转化率相等 (p_control = p_treatment)
    # 备择假设 H1: 两个转化率不等 (p_control != p_treatment)
    print("H0: 新旧页面转化率没有差异")
    print("H1: 新旧页面转化率有差异")

    # 合并转化率（原假设下两个转化率相等的估计）
    # pooled_p = (两组转化数之和) / (两组样本量之和)
    x_control = control_cr * n_control
    x_treatment = treatment_cr * n_treatment
    pooled_p = (x_control + x_treatment) / (n_control + n_treatment)

    # z 统计量 = (两比例之差) / (合并标准误)
    se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n_control + 1/n_treatment))
    z_stat = (treatment_cr - control_cr) / se

    # 双侧 p 值
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    print(f"\n合并转化率: {pooled_p*100:.4f}%")
    print(f"标准误: {se:.6f}")
    print(f"z 统计量: {z_stat:.4f}")
    print(f"p 值: {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"p 值 = {p_value:.4f} < 0.05，拒绝 H0 → 两组转化率有显著差异")
    else:
        print(f"p 值 = {p_value:.4f} >= 0.05，不能拒绝 H0 → 两组转化率无显著差异")

    return z_stat, p_value


def confidence_interval(control_cr, treatment_cr, n_control, n_treatment):
    """置信区间: 转化率差异的 95% 置信区间."""
    from scipy import stats

    print("\n=== 95% 置信区间 ===")

    # 用各自的转化率计算标准误（不用合并的）
    se = np.sqrt(control_cr*(1-control_cr)/n_control + treatment_cr*(1-treatment_cr)/n_treatment)
    diff = treatment_cr - control_cr

    z_critical = stats.norm.ppf(0.975)  # 95% 置信水平对应的 z 值 ≈ 1.96
    lower = diff - z_critical * se
    upper = diff + z_critical * se

    print(f"转化率差异: {diff*100:.4f}%")
    print(f"标准误: {se:.6f}")
    print(f"95% 置信区间: [{lower*100:.4f}%, {upper*100:.4f}%]")

    if lower > 0:
        print("区间全在正区间 → 新页面显著优于旧页面")
    elif upper < 0:
        print("区间全在负区间 → 新页面显著差于旧页面")
    else:
        print("区间包含 0 → 无法确定新页面优于或差于旧页面")

    return lower, upper


def effect_size(control_cr, treatment_cr):
    """效应量: Cohen's h."""
    print("\n=== 效应量：Cohen's h ===")

    # Cohen's h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
    h = 2 * (np.arcsin(np.sqrt(treatment_cr)) - np.arcsin(np.sqrt(control_cr)))

    print(f"Cohen's h = {h:.4f}")

    if abs(h) < 0.2:
        print("效应量: 微小 (|h| < 0.2)")
    elif abs(h) < 0.5:
        print("效应量: 小 (0.2 <= |h| < 0.5)")
    elif abs(h) < 0.8:
        print("效应量: 中等 (0.5 <= |h| < 0.8)")
    else:
        print("效应量: 大 (|h| >= 0.8)")

    return h


def sample_size_calculation(baseline_cr, min_detectable_effect=0.01, alpha=0.05, power=0.8):
    """样本量计算: 如果继续做实验需要多少样本."""
    from scipy import stats

    print("\n=== 样本量计算 ===")
    print(f"基准转化率: {baseline_cr*100:.2f}%")
    print(f"最小可检测效应: {min_detectable_effect*100:.2f}% (绝对提升)")
    print(f"显著性水平: {alpha}")
    print(f"统计功效: {power}")

    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    p2 = baseline_cr + min_detectable_effect
    pooled = (baseline_cr + p2) / 2

    n = ((z_alpha * np.sqrt(2 * pooled * (1 - pooled)))**2 +
         (z_beta * np.sqrt(baseline_cr*(1-baseline_cr) + p2*(1-p2)))**2) / (min_detectable_effect**2)

    n = int(np.ceil(n))
    print(f"\n每组所需样本量: {n}")
    print(f"总样本量: {n * 2}")

    return n


def plot_z_test_visual(z_stat, p_value, save_path="../images/"):
    """可视化: z 检验的正态分布图."""
    import os
    os.makedirs(save_path, exist_ok=True)

    from scipy import stats
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color='#333333', linewidth=2)
    ax.fill_between(x, y, where=(x <= -1.96) | (x >= 1.96), color='#f46d43', alpha=0.5, label='拒绝域 (α=0.05)')
    ax.fill_between(x, y, where=(x > -1.96) & (x < 1.96), color='#abd9e9', alpha=0.3, label='接受域')

    ax.axvline(z_stat, color='#d73027', linewidth=2, linestyle='--', label=f'z = {z_stat:.4f}')
    ax.axvline(-1.96, color='#999999', linewidth=1, linestyle='--')
    ax.axvline(1.96, color='#999999', linewidth=1, linestyle='--')

    ax.set_xlabel('z 值', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title(f'假设检验：z = {z_stat:.4f}, p = {p_value:.4f}', fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path + "z_test.png", dpi=150, bbox_inches='tight')
    print("z检验可视化已保存")
    plt.close()


if __name__ == "__main__":
    # 1. 加载数据
    df = load_data()

    # 2. 数据清洗
    df_clean = clean_data(df)

    # 3. 描述统计
    control_cr, treatment_cr = descriptive_stats(df_clean)

    # 4. 转化率对比图
    plot_conversion_rates(control_cr, treatment_cr)

    # 5. 假设检验
    n_control = len(df_clean[df_clean['group'] == 'control'])
    n_treatment = len(df_clean[df_clean['group'] == 'treatment'])
    z_stat, p_value = hypothesis_test(control_cr, treatment_cr, n_control, n_treatment)

    # 6. z 检验可视化
    plot_z_test_visual(z_stat, p_value)

    # 7. 置信区间
    ci_lower, ci_upper = confidence_interval(control_cr, treatment_cr, n_control, n_treatment)

    # 8. 效应量
    h = effect_size(control_cr, treatment_cr)

    # 9. 样本量计算
    sample_size_calculation(control_cr, min_detectable_effect=0.01)

    print("\n=== 分析完成 ===")
