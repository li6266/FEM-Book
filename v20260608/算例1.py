import numpy as np
import json
import os
import time


# ==========================================
# 任务1：实现稠密矩阵 LDL^T 求解器
# ==========================================

def ldlt_factor(K):
    """
    对对称矩阵 K 进行 LDL^T 分解
    K = L D L^T

    参数:
        K (np.ndarray): 输入的对称矩阵 (n x n)

    返回:
        L (np.ndarray): 单位下三角矩阵 (n x n)
        D (np.ndarray): 对角矩阵的对角线元素 (n,)

    异常:
        如果检测到非正主元，抛出 ValueError
    """
    n = K.shape[0]
    L = np.eye(n, dtype=np.float64)
    D = np.zeros(n, dtype=np.float64)

    for j in range(n):
        # 1. 计算 D[j]
        sum_val = 0.0
        for k in range(j):
            sum_val += L[j, k] ** 2 * D[k]

        D[j] = K[j, j] - sum_val

        # 2. 检查非正主元
        if D[j] <= 1e-12:
            raise ValueError(f"矩阵非正定或存在零主元！在列 {j} 处发现 D[{j}] = {D[j]:.6e}")

        # 3. 计算 L[i, j] (i > j)
        for i in range(j + 1, n):
            sum_val = 0.
            for k in range(j):
                sum_val += L[i, k] * L[j, k] * D[k]
            L[i, j] = (K[i, j] - sum_val) / D[j]

    return L, D


def ldlt_solve(L, D, R):
    """
    求解 L D L^T a = R
    包含前向代入、对角求解和后向代入
    """
    n = L.shape[0]
    a = np.zeros(n, dtype=np.float64)

    # 1. 前向代入：L y = R
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L[i, j] * y[j]
        y[i] = R[i] - sum_val  # 因为 L[i,i] = 1

    # 2. 对角求解：D z = y
    z = np.zeros(n, dtype=np.float64)
    for i in range(n):
        z[i] = y[i] / D[i]

    # 3. 后向代入：L^T a = z
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += L[j, i] * a[j]  # L^T 的第 i 列是 L 的第 i 行
        a[i] = z[i] - sum_val

    return a


def residual_norm(K, a, R):
    """
    计算残差 r = R - K a 及其范数
    """
    r = R - np.dot(K, a)
    norm_r = np.linalg.norm(r)
    return r, norm_r


# ==========================================
# 算例1：三对角对称正定矩阵
# ==========================================

def generate_example1_json(filename="example1.json", n=5):
    """
    生成算例1的JSON文件
    K[i,i] = 2, K[i,i-1] = K[i-1,i] = -1
    a_exact = [1, 1, ..., 1]^T
    R = K * a_exact
    """
    # 构造三对角矩阵 K
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        K[i, i] = 2.0
        if i > 0:
            K[i, i - 1] = -1.0
            K[i - 1, i] = -1.0

    # 构造精确解 a_exact
    a_exact = np.ones(n, dtype=np.float64)

    # 计算右端项 R = K * a_exact
    R = np.dot(K, a_exact)

    # 构造JSON数据结构
    data = {
        "Title": "算例1：三对角对称正定矩阵",
        "n": n,
        "K": K.tolist(),
        "R": R.tolist(),
        "a_exact": a_exact.tolist()
    }

    # 写入JSON文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f" 已生成JSON文件: {filename}")
    return filename


def load_and_solve_from_json(filename):
    """
    从JSON文件加载数据，调用LDL^T求解器，并输出结果
    """
    # 1. 加载JSON文件
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    n = data["n"]
    K = np.array(data["K"], dtype=np.float64)
    R = np.array(data["R"], dtype=np.float64)
    a_exact = np.array(data["a_exact"], dtype=np.float64)

    print(f"\n 算例1：n = {n}")
    print(f"矩阵K:\n{K}")
    print(f"右端项R: {R}")
    print(f"精确解a_exact: {a_exact}")

    # 2. 记录开始时间
    start_time = time.time()

    try:
        # 3. LDL^T 分解
        L, D = ldlt_factor(K)
        print(f"L 矩阵:\n{L}")
        print(f"D 向量: {D}")

        # 4. 求解
        a = ldlt_solve(L, D, R)
        print(f"数值解 a: {a}")

        # 5. 计算残差
        r, norm_r = residual_norm(K, a, R)
        print(f"\n 残差范数 ||R - K·a|| = {norm_r:.6e}")

        # 6. 计算与精确解的误差
        error = np.linalg.norm(a - a_exact)
        max_error = np.max(np.abs(a - a_exact))
        print(f"与精确解的最大误差: {max_error:.6e}")
        print(f"与精确解的 L2 误差: {error:.6e}")

    except ValueError as e:
        print(f" 求解失败: {e}")


# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":
    # 1. 生成算例1的JSON文件（n=5）
    filename = "example1.json"
    generate_example1_json(filename, n=5)

    # 2. 从JSON文件加载并求解
    load_and_solve_from_json(filename)