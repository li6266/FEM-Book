import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FormatStrFormatter
import matplotlib.font_manager as fm
import os
import datetime


# ==========================================
# 字体设置函数
# ==========================================
def set_chinese_font():
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'KaiTi',
                     'FangSong', 'Arial Unicode MS', 'PingFang SC', 'Noto Sans CJK SC']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    if selected_font:
        print(f"✅ 成功找到并设置中文字体: {selected_font}")
        plt.rcParams['font.sans-serif'] = [selected_font]
    else:
        print("⚠️ 未找到系统中文字体，尝试使用默认回退方案...")
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        except:
            pass
    plt.rcParams['axes.unicode_minus'] = False


set_chinese_font()


# ==========================================
# 有限元法计算函数
# ==========================================
def finite_element_pi(n, use_extrapolation=False):
    """有限元法计算π的近似值"""
    if n < 1:
        return 0.0, None
    pi_n = n * np.sin(np.pi / n)
    if use_extrapolation and n >= 4 and n % 2 == 0:
        n_half = n // 2
        pi_half = n_half * np.sin(np.pi / n_half)
        pi_extrapolated = (4 * pi_n - pi_half) / 3
        return pi_n, pi_extrapolated
    return pi_n, None


def save_plot_image(fig, filename=None, formats=None, dpi=300):
    """
    保存图表为图片文件

    参数:
    fig: matplotlib图形对象
    filename: 保存的文件名（不含扩展名）
    formats: 保存的格式列表，默认['png', 'pdf']
    dpi: 图片分辨率
    """
    if filename is None:
        # 生成带时间戳的文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fem_convergence_plot_{timestamp}"

    if formats is None:
        formats = ['png', 'pdf']

    # 确保输出目录存在
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []
    for fmt in formats:
        filepath = os.path.join(output_dir, f"{filename}.{fmt}")
        try:
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
            saved_files.append(filepath)
            print(f"✅ 图片已保存为: {filepath}")
        except Exception as e:
            print(f"❌ 保存 {fmt} 格式时出错: {e}")

    return saved_files


def calculate_and_plot():
    max_n = 256
    n_values = [2 ** i for i in range(1, 9)]  # 2 to 256

    h_values = []
    basic_errors = []
    extrap_errors = []

    print("=" * 100)
    print("有限元法求解圆周率 π - 数值结果表")
    print("=" * 100)
    header = f"{'n':<8} {'π_n (FEM)':<15} {'误差':<15} {'外推值':<15} {'外推误差':<15}"
    print(header)
    print("-" * 100)

    exact_pi = np.pi

    for n in n_values:
        pi_n, pi_ext = finite_element_pi(n, use_extrapolation=(n in [4, 16, 64, 256]))
        err = abs(exact_pi - pi_n)
        ext_err = abs(exact_pi - pi_ext) if pi_ext else float('nan')

        h_values.append(1.0 / n)
        basic_errors.append(err)

        if pi_ext:
            extrap_errors.append(ext_err)
        else:
            extrap_errors.append(float('nan'))

        def fmt(x):
            if abs(x) < 1e-5 or abs(x) > 1e5:
                return f"{x:.5e}"
            else:
                return f"{x:.10f}".rstrip('0').rstrip('.')

        ext_str = fmt(pi_ext) if pi_ext else "-"
        ext_err_str = fmt(ext_err) if pi_ext else "-"
        print(f"{n:<8} {fmt(pi_n):<15} {fmt(err):<15} {ext_str:<15} {ext_err_str:<15}")

    # ==========================================
    # 计算蓝线斜率
    # ==========================================
    # 提取有效数据点（去除NaN）
    valid_mask = ~np.isnan(basic_errors)
    log_h = np.log10(h_values)[valid_mask]
    log_err = np.log10(basic_errors)[valid_mask]

    # 线性拟合求斜率 k
    if len(log_h) > 1:
        k, b = np.polyfit(log_h, log_err, 1)
        slope_text = f"Slope = {k:.4f}"
    else:
        slope_text = "Slope = N/A"

    # ==========================================
    # 绘图
    # ==========================================
    fig = plt.figure(figsize=(10, 7))

    # 绘制基本有限元法误差（蓝色三角形）- 这是你要标斜率的线
    valid_idx_basic = [i for i, v in enumerate(basic_errors) if not np.isnan(v)]
    line, = plt.loglog(
        [h_values[i] for i in valid_idx_basic],
        [basic_errors[i] for i in valid_idx_basic],
        marker='v', linestyle='-', color='tab:blue', label='FEM'
    )

    # 标注斜率
    x_pos = h_values[0] * 1.5  # 稍微靠右一点
    y_pos = basic_errors[len(basic_errors) // 2] * 0.1  # 稍微靠下一点

    # 添加文本框
    plt.text(x_pos, y_pos, slope_text,
             fontsize=12, color='tab:blue',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 图表设置
    plt.xlabel('$h = 1/n$', fontsize=12)
    plt.ylabel('$e_n = |\\pi - \\pi_n|$', fontsize=12)
    plt.title('Convergence of Finite Element Method for Pi', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.xlim(1e-3, 1e0)
    plt.ylim(1e-16, 1e5)

    # ==========================================
    # 新增：保存图片
    # ==========================================
    print("\n" + "=" * 100)
    print("保存图片...")
    print("=" * 100)

    # 保存图片为文件
    saved_files = save_plot_image(fig, "fem_pi_convergence", formats=['png', 'pdf', 'svg'], dpi=300)

    # 显示图片
    plt.show()

    return saved_files, slope_text, h_values, basic_errors


if __name__ == "__main__":
    saved_files, slope_value, h_vals, errors = calculate_and_plot()

    # 打印汇总信息
    print("\n" + "=" * 100)
    print("运行结果汇总")
    print("=" * 100)
    print(f"1. 收敛斜率: {slope_value}")
    print(f"2. 保存的图片文件:")
    for file in saved_files:
        print(f"   - {file}")
    print(f"3. 计算参数: n = 2, 4, 8, 16, 32, 64, 128, 256")
    print(f"4. 最终误差: {errors[-1]:.2e} (n=256时)")