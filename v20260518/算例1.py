import numpy as np
from typing import Tuple, Union, List


def truss3d_element_stiffness(x1: np.ndarray, x2: np.ndarray, E: float, A: float) -> Tuple[
    float, np.ndarray, np.ndarray]:
    """
    计算三维杆单元的基本属性和6x6刚度矩阵

    参数:
        x1, x2: 两个节点坐标，例如 [x, y, z]
        E: 弹性模量
        A: 截面积

    返回:
        L: 单元长度
        direction_cosines: 方向余弦 [cx, cy, cz]
        Ke: 6x6 单元刚度矩阵
    """
    # 检查退化单元
    dx = x2 - x1
    L = np.linalg.norm(dx)
    if L < 1e-10:
        raise ValueError("错误：两个节点重合，这是退化单元！")

    # 计算方向余弦
    cx = dx[0] / L
    cy = dx[1] / L
    cz = dx[2] / L
    direction_cosines = np.array([cx, cy, cz])

    # 局部刚度标量
    k = (E * A) / L

    # 构造完整的6x6刚度矩阵
    Ke = np.zeros((6, 6))

    # 填充刚度矩阵
    for i in range(3):
        for j in range(3):
            # 左上块
            Ke[i, j] = k * direction_cosines[i] * direction_cosines[j]
            # 右下块
            Ke[i + 3, j + 3] = k * direction_cosines[i] * direction_cosines[j]
            # 右上块
            Ke[i, j + 3] = -k * direction_cosines[i] * direction_cosines[j]
            # 左下块
            Ke[i + 3, j] = -k * direction_cosines[i] * direction_cosines[j]

    return L, direction_cosines, Ke


def truss3d_element_stress(x1: np.ndarray, x2: np.ndarray, E: float, A: float, de: np.ndarray) -> Tuple[
    float, float, float]:
    """
    计算三维杆单元的应变、应力和轴力

    参数:
        x1, x2: 两个节点坐标，例如 [x, y, z]
        E: 弹性模量
        A: 截面积
        de: 单元节点位移 [u1, v1, w1, u2, v2, w2]

    返回:
        epsilon: 应变
        sigma: 应力
        N: 轴力
    """
    # 检查退化单元
    dx = x2 - x1
    L = np.linalg.norm(dx)
    if L < 1e-10:
        raise ValueError("错误：两个节点重合，这是退化单元！")

    # 计算方向余弦
    cx = dx[0] / L
    cy = dx[1] / L
    cz = dx[2] / L

    # 计算位移差在杆轴方向上的投影
    du = de[3] - de[0]  # u2 - u1
    dv = de[4] - de[1]  # v2 - v1
    dw = de[5] - de[2]  # w2 - w1

    delta = cx * du + cy * dv + cz * dw

    # 计算应变
    epsilon = delta / L

    # 计算应力
    sigma = E * epsilon

    # 计算轴力
    N = sigma * A

    return epsilon, sigma, N


def get_compact_stiffness_matrix(Ke: np.ndarray, tolerance: float = 1e-10) -> Tuple[np.ndarray, List[int]]:
    """
    从完整的6x6刚度矩阵生成紧凑矩阵
    消除为0的行和列

    参数:
        Ke: 6x6刚度矩阵
        tolerance: 容差

    返回:
        Ke_compact: 紧凑刚度矩阵
        active_dofs: 活动的自由度索引
    """
    # 找出非零的自由度
    active_dofs = []
    for i in range(6):
        # 检查行和列是否有非零元素
        row_nonzero = np.any(np.abs(Ke[i, :]) > tolerance)
        col_nonzero = np.any(np.abs(Ke[:, i]) > tolerance)
        if row_nonzero or col_nonzero:
            active_dofs.append(i)

    # 提取紧凑的刚度矩阵
    n_active = len(active_dofs)
    Ke_compact = np.zeros((n_active, n_active))

    for i, dof_i in enumerate(active_dofs):
        for j, dof_j in enumerate(active_dofs):
            Ke_compact[i, j] = Ke[dof_i, dof_j]

    return Ke_compact, active_dofs


# ========================
# ========================
if __name__ == "__main__":
    print("\n=== 三维杆单元计算程序 ===")
    print("在最终输出时自动压缩刚度矩阵\n")

    # ============================================================
    # 在这里直接修改您的输入参数
    # ============================================================

    # 算例1:一维杆
    print("算例1: 一维杆")
    print("-" * 50)

    # 1. 节点坐标
    node1 = np.array([0.0, 0.0, 0.0])  # 节点1坐标
    node2 = np.array([2.0, 0.0, 0.0])  # 节点2坐标

    # 2. 材料参数
    E = 2.0e11  # 弹性模量 (Pa)
    A = 1.0e-4  # 截面积 (m²)

    # 3. 单元节点位移
    de = np.array([0.0, 0.0, 0.0, 1.0e-3, 0, 0])  # [u1, v1, w1, u2, v2, w2]

    # ============================================================
    # 计算和输出
    # ============================================================

    print(f"节点1坐标: {node1}")
    print(f"节点2坐标: {node2}")
    print(f"弹性模量 E = {E:.2e} Pa")
    print(f"截面积 A = {A:.4f} m²")
    print(f"单元节点位移 de = {de}\n")

    try:
        # 1. 计算完整6x6刚度矩阵
        L, direction_cosines, Ke_full = truss3d_element_stiffness(node1, node2, E, A)

        print("--- 基本属性计算结果 ---")
        print(f"单元长度 L = {L:.6f} m")
        print(
            f"方向余弦 [cx, cy, cz] = [{direction_cosines[0]:.6f}, {direction_cosines[1]:.6f}, {direction_cosines[2]:.6f}]")

        # 2. 生成紧凑刚度矩阵
        Ke_compact, active_dofs = get_compact_stiffness_matrix(Ke_full)

        print(f"\n--- 刚度矩阵 (维度 {Ke_compact.shape[0]}x{Ke_compact.shape[1]}) ---")
        for i in range(Ke_compact.shape[0]):
            print("[", end="")
            for j in range(Ke_compact.shape[1]):
                print(f"{Ke_compact[i, j]:12.2f}", end="")
                if j < Ke_compact.shape[1] - 1:
                    print(", ", end="")
            print("]")

        # 检查对称性
        is_sym_compact = np.allclose(Ke_compact, Ke_compact.T, atol=1e-10)
        print(f"矩阵对称性检查: {'✓ 对称' if is_sym_compact else '✗ 不对称'}")

        # 3. 计算应变、应力、轴力
        epsilon, sigma, N = truss3d_element_stress(node1, node2, E, A, de)

        print(f"\n--- 应力应变计算结果 ---")
        print(f"应变 epsilon = {epsilon:.6e}")
        print(f"应力 sigma = {sigma:.2f} Pa")
        print(f"轴力 N = {N:.2f} N")

        #检查退化单元
    except ValueError as e:
        print(f"\n错误：{e}")
    # 选 j = 3（即第4个自由度，节点2的x方向位移）
    j = 3
    de_unit = np.zeros(6)
    de_unit[j] = 1.0

    # 计算 Fe = Ke * de
    Fe = Ke_full @ de_unit

    print(f"\n选定自由度 j = {j}")
    print(f"单位位移向量 de = {de_unit}")
    print(f"计算得到的节点力 Fe = {Fe}")
    print(f"刚度矩阵 Ke 的第 {j} 列 = {Ke_full[:, j]}")

    # 验证是否相等
    is_equal = np.allclose(Fe, Ke_full[:, j], atol=1e-10)
    print(f"\n验证结果：Fe 是否等于 Ke 的第 {j} 列？ {'✓ 是' if is_equal else '✗ 否'}")

    # 说明 kij 的物理意义
    print(f"\n【kij 的物理意义】：")
    print(f"kij 表示当第 j 个自由度发生单位位移（δj = 1），而其余所有自由度保持为零位移时，")
    print(f"在第 i 个自由度上所产生的节点力（或反力）。")
    print(f"例如，k_{j}{j} 是主对角线元素，代表第 j 个自由度的自身刚度；")
    print(f"k_{j - 1}{j} 是非对角线元素，代表第 j 个自由度位移对第 j-1 个自由度的影响。")


