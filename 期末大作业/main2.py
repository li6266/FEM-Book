import numpy as np
import matplotlib.pyplot as plt


def element_matrix(kappa, v, le, alpha):
    """返回两节点线性单元的 2 x 2 对流扩散单元矩阵"""
    # 计算稳定化后的人工扩散系数
    kappa_bar = kappa + alpha * v * le / 2

    # 计算扩散项对应的矩阵
    diffusion_term = (kappa_bar / le) * np.array([[1, -1],
                                                  [-1, 1]])

    # 计算对流项对应的矩阵
    convection_term = (v / 2) * np.array([[-1, 1],
                                          [-1, 1]])

    # 组装单元矩阵
    Ke = diffusion_term + convection_term
    return Ke


def alpha_sug(Pe):
    """返回 coth(Pe) - 1/Pe，并处理 Pe 接近 0 时的溢出问题"""
    if abs(Pe) < 1e-5:
        return 0
    else:
        return 1 / np.tanh(Pe) - 1 / Pe


def solve_advection_diffusion(nel, L, v, Pe, alpha):
    """
    求解一维稳态对流扩散方程
    返回节点坐标 x、数值解 theta_num、精确解 theta_exact
    """
    # 1. 生成节点坐标 (nel个单元对应 nel+1 个节点)
    x = np.linspace(0, L, nel + 1)

    # 2. 根据指定的 Peclet 数自动计算扩散系数 kappa
    le = L / nel
    kappa = v * le / (2 * Pe)

    # 3. 初始化总体刚度矩阵 K 和载荷向量 F
    K_global = np.zeros((nel + 1, nel + 1))
    F_global = np.zeros(nel + 1)

    # 4. 遍历所有单元进行组装
    for i in range(nel):
        # 提取当前单元对应的节点全局索引
        node_indices = [i, i + 1]

        # 获取当前单元矩阵
        Ke = element_matrix(kappa, v, le, alpha)

        # 将单元矩阵组装到总体矩阵中
        for row in range(2):
            for col in range(2):
                K_global[node_indices[row], node_indices[col]] += Ke[row, col]

    # 5. 施加边界条件（使用直接置入法，更可靠）
    # 左边界: theta(0) = 0
    K_global[0, :] = 0
    K_global[0, 0] = 1
    F_global[0] = 0

    # 右边界: theta(L) = 1
    K_global[nel, :] = 0
    K_global[nel, nel] = 1
    F_global[nel] = 1

    # 6. 求解线性方程组 K * theta = F
    try:
        theta_num = np.linalg.solve(K_global, F_global)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，使用最小二乘法
        theta_num = np.linalg.lstsq(K_global, F_global, rcond=None)[0]

    # 7. 计算精确解
    # 避免指数溢出
    if kappa > 1e-15:
        exponent_val = v * x / kappa
        exp_L = np.exp(v * L / kappa)
        # 防止除以零
        if abs(exp_L - 1) > 1e-15:
            theta_exact = (np.exp(exponent_val) - 1) / (exp_L - 1)
        else:
            theta_exact = x / L  # 极限情况：线性分布
    else:
        # 纯对流情况，精确解应为阶跃函数
        theta_exact = np.where(x >= L, 1.0, 0.0)

    return x, theta_num, theta_exact


def calculate_max_error(theta_num, theta_exact):
    """计算最大节点误差"""
    return np.max(np.abs(theta_num - theta_exact))


def print_numerical_results(x, theta_num, theta_exact, method_name, Pe):
    """打印节点坐标、数值解和精确解的详细数据"""
    print(f"\n{'=' * 70}")
    print(f"方法: {method_name} | Pe = {Pe}")
    print(f"{'=' * 70}")
    print(f"{'节点':>6} {'x坐标':>10} {'数值解 θ_num':>15} {'精确解 θ_exact':>15} {'误差':>15}")
    print(f"{'-' * 61}")

    for i in range(len(x)):
        error = abs(theta_num[i] - theta_exact[i])
        print(f"{i:>6} {x[i]:>10.6f} {theta_num[i]:>15.8f} {theta_exact[i]:>15.8f} {error:>15.8e}")

    max_error = calculate_max_error(theta_num, theta_exact)
    print(f"{'-' * 61}")
    print(f"{'最大误差:':>47} {max_error:>15.8e}")


def print_analysis_results(Pe, errors):
    """打印任务 3 要求分析的文本说明"""
    print(f"   其最大误差达到 {errors['standard']:.6f}，解在边界附近出现明显非物理振荡。")


def task4_matrix_analysis():
    """执行任务 4: 矩阵性质分析——输出标准 Galerkin 总体矩阵并检查对称性和正定性"""
    print("\n" + "=" * 70)
    print("任务4: 标准 Galerkin 总体矩阵分析 (Pe = 3.0, nel = 20)")
    print("=" * 70)

    # 固定参数
    L = 1
    nel = 20
    v = 1
    Pe = 3.0

    # 计算 kappa 并组装标准 Galerkin 矩阵 (alpha = 0)
    le = L / nel
    kappa = v * le / (2 * Pe)

    # 组装未施加边界条件的原始矩阵
    K_raw = np.zeros((nel + 1, nel + 1))

    for i in range(nel):
        Ke = element_matrix(kappa, v, le, alpha=0)
        node_indices = [i, i + 1]
        for row in range(2):
            for col in range(2):
                K_raw[node_indices[row], node_indices[col]] += Ke[row, col]

    # 输出原始总体矩阵（未施加边界条件）
    print("\n▶ 标准 Galerkin 总体矩阵 (未施加边界条件，21×21)：")

    # 以紧凑格式输出完整矩阵
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    # 输出完整矩阵到控制台（由于矩阵较大，分块输出）
    print("\n▶ 完整矩阵 (第0~10行):")
    for i in range(min(11, nel + 1)):
        row_str = "    ".join([f"{K_raw[i, j]:8.4f}" for j in range(nel + 1)])
        print(f"  行{i:2d}: {row_str}")

    if nel + 1 > 11:
        print("\n▶ 完整矩阵 (第11~20行):")
        for i in range(11, nel + 1):
            row_str = "    ".join([f"{K_raw[i, j]:8.4f}" for j in range(nel + 1)])
            print(f"  行{i:2d}: {row_str}")

    # 1. 检查对称性
    print("\n" + "-" * 50)
    print("▶ 对称性检查:")

    # 计算对称性误差（Frobenius范数）
    sym_error = np.linalg.norm(K_raw - K_raw.T, 'fro')
    is_symmetric = np.allclose(K_raw, K_raw.T, atol=1e-10)

    print(f"   ||K - K^T||_F = {sym_error:.6e}")
    print(f"   矩阵是否对称：{'是 ✓' if is_symmetric else '否 ✗'}")

    if not is_symmetric:
        print("   原因：对流项矩阵是非对称的 (v/2)*[[-1,1],[-1,1]]，破坏了整体对称性")

    # 2. 检查正定性（使用施加边界条件后的矩阵，因为原始矩阵是奇异的）
    print("\n▶ 正定性检查 (施加边界条件后):")

    # 复制矩阵并施加边界条件
    K_bc = K_raw.copy()
    K_bc[0, :] = 0
    K_bc[0, 0] = 1
    K_bc[nel, :] = 0
    K_bc[nel, nel] = 1

    try:
        eigenvalues = np.linalg.eigvals(K_bc)
        min_eig = np.min(eigenvalues)
        max_eig = np.max(eigenvalues)
        is_positive_definite = np.all(eigenvalues > -1e-10)  # 允许微小的数值误差

        print(f"   最小特征值: {min_eig:.6e}")
        print(f"   最大特征值: {max_eig:.6e}")
        print(f"   特征值范围: [{min_eig:.4f}, {max_eig:.4f}]")
        print(f"   矩阵是否正定：{'是 ✓' if is_positive_definite else '否 ✗'}")

        if not is_positive_definite:
            negative_count = np.sum(eigenvalues < -1e-10)
            print(f"   负特征值个数: {negative_count}")
            print("   原因：对流占优时，矩阵失去正定性，导致数值振荡")

        # 输出特征值分布概况
        print(f"\n   特征值分布概况:")
        print(f"     前5个最小特征值: {np.sort(eigenvalues)[:5]}")
        print(f"     前5个最大特征值: {np.sort(eigenvalues)[-5:]}")

    except Exception as e:
        print("   正定性检查失败:", e)

    # 3. 附加分析：矩阵条件数
    print("\n▶ 附加分析 - 条件数:")
    try:
        cond_number = np.linalg.cond(K_bc)
        print(f"   条件数 (cond): {cond_number:.4e}")
        if cond_number > 1e10:
            print("   矩阵高度病态，条件数极大 (>1e10)")
        elif cond_number > 1e5:
            print("   矩阵病态，条件数较大")
        else:
            print("   矩阵条件数适中")
    except Exception as e:
        print("   条件数计算失败:", e)

    # 4. 输出矩阵的非零结构
    print("\n▶ 矩阵带宽分析:")
    n = nel + 1
    nonzero_count = np.count_nonzero(np.abs(K_raw) > 1e-10)
    total_elements = n * n
    sparsity = (1 - nonzero_count / total_elements) * 100
    print(f"   矩阵维度: {n} × {n}")
    print(f"   非零元素个数: {nonzero_count}")
    print(f"   稀疏度: {sparsity:.2f}%")


def main():
    """主函数：执行任务 1、2、3"""
    # 固定参数
    L = 1
    nel = 20
    v = 1

    # 需要计算的 Peclet 数
    Peclet_numbers = [0.1, 3.0]

    # 存储误差用于后续比较
    results_data = {}

    # 创建画布
    plt.figure(figsize=(14, 6))

    for idx, Pe in enumerate(Peclet_numbers):
        # 计算三种 alpha 值
        alpha_standard = 0
        alpha_upwind = 1
        alpha_supg = alpha_sug(Pe)

        alphas = {
            "Standard G": alpha_standard,
            "Upwind": alpha_upwind,
            "SUPG": alpha_supg
        }

        # 求解并绘图
        plt.subplot(1, 2, idx + 1)

        for label, alpha in alphas.items():
            x, theta_num, theta_exact = solve_advection_diffusion(nel, L, v, Pe, alpha)
            max_err = calculate_max_error(theta_num, theta_exact)
            results_data[(Pe, label)] = max_err

            # 打印详细数据
            print_numerical_results(x, theta_num, theta_exact, label, Pe)

            # 绘制曲线
            if label == "Standard G":
                plt.plot(x, theta_num, 'r--', linewidth=2, label=f'{label} (Err: {max_err:.4f})')
            elif label == "Upwind":
                plt.plot(x, theta_num, 'b:', linewidth=2, label=f'{label} (Err: {max_err:.4f})')
            else:
                plt.plot(x, theta_num, 'g-', linewidth=2, label=f'{label} (Err: {max_err:.4f})')

        # 绘制精确解作为基准
        plt.plot(x, theta_exact, 'k-', linewidth=1.5, label='Exact Solution')
        plt.title(f'Advection-Diffusion Solution (Pe = {Pe})')
        plt.xlabel('x')
        plt.ylabel('θ')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

        # 打印误差分析
        print_analysis_results(Pe, {
            "standard": results_data[(Pe, "Standard G")],
            "upwind": results_data[(Pe, "Upwind")],
            "supg": results_data[(Pe, "SUPG")]
        })

    plt.tight_layout()
    plt.show()

    # 执行任务 4
    task4_matrix_analysis()


if __name__ == "__main__":
    main()