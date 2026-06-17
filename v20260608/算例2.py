import json
import numpy as np


# LDL^T 分解函数
def ldlt_factor(K):
    n = K.shape[0]
    L = np.eye(n)
    D = np.zeros(n)

    for j in range(n):
        sum_val = 0.0
        for k in range(j):
            sum_val += L[j, k] ** 2 * D[k]
        D[j] = K[j, j] - sum_val

        if D[j] <= 1e-10:
            raise ValueError("矩阵非正定：D[{0}] = {1}".format(j, D[j]))

        for i in range(j + 1, n):
            sum_val = 0.0
            for k in range(j):
                sum_val += L[i, k] * L[j, k] * D[k]
            L[i, j] = (K[i, j] - sum_val) / D[j]

    return L, D


# LDL^T 求解函数
def ldlt_solve(L, D, R):
    n = L.shape[0]

    # 前代求解 Ly = R
    y = np.zeros(n)
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L[i, j] * y[j]
        y[i] = R[i] - sum_val

    # 对角线求解 Dz = y
    z = np.zeros(n)
    for i in range(n):
        z[i] = y[i] / D[i]

    # 回代求解 L^T a = z
    a = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += L[j, i] * a[j]
        a[i] = z[i] - sum_val

    return a


# 残差计算函数
def residual_norm(K, a, R):
    r = R - np.dot(K, a)
    return np.linalg.norm(r), r


# ==================== 新算例定义部分 ====================
def generate_example2_json():
    """生成算例2（非正定矩阵）的 JSON 文件"""
    data = {
        "matrix": [[1, 2], [2, 1]],
        "RHS": [1, 1]
    }
    with open("example2.json", "w") as f:
        json.dump(data, f, indent=4)
    print("已生成 example2.json")


def load_and_solve_from_json():
    """从 JSON 加载算例2数据并求解"""
    try:
        with open("example2.json", "r") as f:
            data = json.load(f)

        K = np.array(data["matrix"])
        R = np.array(data["RHS"])

        print("\n--- 开始求解算例2 ---")
        print("测试矩阵 K:\n", K)
        print("右端向量 R:", R)

        # 尝试进行 LDL^T 分解
        L, D = ldlt_factor(K)

        # 求解方程组
        a = ldlt_solve(L, D, R)

        # 计算残差
        res_norm, res_vec = residual_norm(K, a, R)

        print("\n--- 计算结果 ---")
        print("解向量 a:", a)
        print("残差范数 ||R - Ka||:", res_norm)

    except ValueError as e:
        print(f"\n 计算中断：{e}")
    except FileNotFoundError:
        print(" 未找到 example2.json 文件，正在重新生成...")
        generate_example2_json()
        load_and_solve_from_json()


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 清理旧文件，避免混淆
    import os

    if os.path.exists("example2.json"):
        os.remove("example2.json")

    load_and_solve_from_json()