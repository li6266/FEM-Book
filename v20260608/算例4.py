import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
import time
import pypardiso  # 启用 Intel MKL PARDISO 求解器


# -------------------------------
# 1. 参数设置与网格生成 (Q4)
# -------------------------------
def generate_q4_mesh(nx, ny):
    """
    生成单位正方形 [0,1]x[0,1] 上的 Q4 网格。
    返回：节点坐标 (x, y), 单元连接 (ien), 总节点数 (nn), 总单元数 (ne)
    """
    nn = (nx + 1) * (ny + 1)
    ne = nx * ny

    # 节点坐标
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    X, Y = np.meshgrid(x, y)
    nodes = np.column_stack((X.ravel(), Y.ravel()))

    # 单元连接 (ien): 每个单元4个节点，按逆时针顺序
    ien = np.zeros((ne, 4), dtype=int)
    for i in range(nx):
        for j in range(ny):
            elem_idx = i * ny + j
            # 左下、右下、右上、左上
            ien[elem_idx, 0] = j + i * (ny + 1)
            ien[elem_idx, 1] = j + 1 + i * (ny + 1)
            ien[elem_idx, 2] = j + 1 + (i + 1) * (ny + 1)
            ien[elem_idx, 3] = j + (i + 1) * (ny + 1)

    return nodes, ien, nn, ne


# -------------------------------
# 2. 双线性四边形单元 (Q4) 刚度矩阵计算
# -------------------------------
def q4_stiffness_matrix(nodes, elem):
    """
    计算单个 Q4 单元的刚度矩阵 (4x4)
    nodes: 所有节点的坐标数组
    elem: 当前单元的4个节点编号
    """
    # 提取当前单元4个节点的坐标
    x = nodes[elem, 0]
    y = nodes[elem, 1]

    # 2x2 Gauss 积分点坐标和权重
    gauss_pts = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
    weights = np.array([1.0, 1.0])

    Ke = np.zeros((4, 4))

    for i in range(2):
        for j in range(2):
            xi = gauss_pts[i]
            eta = gauss_pts[j]
            w = weights[i] * weights[j]

            # 形函数对 xi, eta 的导数
            dN_dxi = np.array([
                -0.25 * (1 - eta),
                0.25 * (1 - eta),
                0.25 * (1 + eta),
                -0.25 * (1 + eta)
            ])
            dN_deta = np.array([
                -0.25 * (1 - xi),
                -0.25 * (1 + xi),
                0.25 * (1 + xi),
                0.25 * (1 - xi)
            ])

            # 雅可比矩阵 J = [dx/dxi, dy/dxi; dx/deta, dy/deta]
            J = np.array([
                [np.sum(dN_dxi * x), np.sum(dN_dxi * y)],
                [np.sum(dN_deta * x), np.sum(dN_deta * y)]
            ])

            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            # B 矩阵 = [dN/dx, dN/dy] = invJ * [dN/dxi; dN/deta]
            dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
            dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta

            # B 矩阵 (Poisson 方程) = [dN/dx, dN/dy]^T，形状 (2, 4)
            B = np.vstack((dN_dx, dN_dy))

            # 单元刚度矩阵 Ke += B^T * B * detJ * w
            Ke += B.T @ B * detJ * w

    return Ke


# -------------------------------
# 3. 组装总体刚度矩阵 (稀疏)
# -------------------------------
def assemble_global_stiffness(nodes, ien, nn, ne):
    """
    组装总体刚度矩阵 (稀疏 COO 格式)
    """
    rows = []
    cols = []
    data = []

    for e in range(ne):
        elem_nodes = ien[e]
        Ke = q4_stiffness_matrix(nodes, elem_nodes)

        for i in range(4):
            for j in range(4):
                rows.append(elem_nodes[i])
                cols.append(elem_nodes[j])
                data.append(Ke[i, j])

    # 创建稀疏矩阵 (COO 格式)
    K = sp.coo_matrix((data, (rows, cols)), shape=(nn, nn))

    # 转换为 CSR 格式 (更适合求解)
    return K.tocsr()


# -------------------------------
# 4. 计算右端项 (载荷向量 F)
# -------------------------------
def compute_rhs(nodes, ien, ne, nn, f_func):
    """
    计算右端项 F (载荷向量)
    f_func: 右端项函数 f(x, y)
    """
    F = np.zeros(nn)

    # 2x2 Gauss 积分点
    gauss_pts = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
    weights = np.array([1.0, 1.0])

    for e in range(ne):
        elem_nodes = ien[e]
        x = nodes[elem_nodes, 0]
        y = nodes[elem_nodes, 1]

        Fe = np.zeros(4)

        for i in range(2):
            for j in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                w = weights[i] * weights[j]

                # 形函数值 N1, N2, N3, N4
                N = np.array([
                    0.25 * (1 - xi) * (1 - eta),
                    0.25 * (1 + xi) * (1 - eta),
                    0.25 * (1 + xi) * (1 + eta),
                    0.25 * (1 - xi) * (1 + eta)
                ])

                # 高斯点坐标
                x_g = np.sum(N * x)
                y_g = np.sum(N * y)

                # 右端项值 f(x_g, y_g)
                f_val = f_func(x_g, y_g)

                # 计算雅可比行列式 detJ
                dN_dxi = np.array([
                    -0.25 * (1 - eta),
                    0.25 * (1 - eta),
                    0.25 * (1 + eta),
                    -0.25 * (1 + eta)
                ])
                dN_deta = np.array([
                    -0.25 * (1 - xi),
                    -0.25 * (1 + xi),
                    0.25 * (1 + xi),
                    0.25 * (1 - xi)
                ])

                J = np.array([
                    [np.sum(dN_dxi * x), np.sum(dN_dxi * y)],
                    [np.sum(dN_deta * x), np.sum(dN_deta * y)]
                ])
                detJ = np.linalg.det(J)

                Fe += N * f_val * detJ * w

        # 将单元载荷加到总体载荷向量
        for i in range(4):
            F[elem_nodes[i]] += Fe[i]

    return F


# -------------------------------
# 5. 应用 Dirichlet 边界条件 (u=0)
# -------------------------------
def apply_dirichlet(K, F, nodes):
    """
    应用 Dirichlet 边界条件 u=0
    """
    # 找到边界节点 (x=0, x=1, y=0, y=1)
    tol = 1e-12
    boundary_nodes = []

    for i in range(len(nodes)):
        x, y = nodes[i]
        if abs(x) < tol or abs(x - 1) < tol or abs(y) < tol or abs(y - 1) < tol:
            boundary_nodes.append(i)

    # 转换为 lil_matrix 以便修改
    K_lil = K.tolil()

    # 将边界行和列清零，对角线设为1
    for i in boundary_nodes:
        K_lil[i, :] = 0
        K_lil[:, i] = 0
        K_lil[i, i] = 1.0
        F[i] = 0.0

    return K_lil.tocsr(), F


# -------------------------------
# 6. 计算误差
# -------------------------------
def compute_errors(nodes, u, u_exact_func):
    """
    计算最大误差和 L2 相对误差
    """
    u_exact_vals = np.array([u_exact_func(x, y) for x, y in nodes])

    # 最大误差
    max_error = np.max(np.abs(u - u_exact_vals))

    # L2 相对误差
    l2_error = np.sqrt(np.sum((u - u_exact_vals) ** 2)) / np.sqrt(np.sum(u_exact_vals ** 2))

    return max_error, l2_error


# -------------------------------
# 7. 绘图函数
# -------------------------------
def plot_solution(nodes, u, title="Numerical Solution"):
    """
    绘制数值解的云图和三维曲面图
    """
    x = nodes[:, 0]
    y = nodes[:, 1]

    # 创建三角剖分
    triang = Triangulation(x, y)

    # 创建图形和子图
    fig = plt.figure(figsize=(12, 5))

    # 子图1：云图
    ax1 = fig.add_subplot(121)
    tpc = ax1.tripcolor(triang, u, shading='gouraud', cmap='viridis')
    ax1.set_title(title)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig.colorbar(tpc, ax=ax1, label='u(x,y)')

    # 子图2：三维曲面图
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_trisurf(triang, u, cmap='viridis', edgecolor='none')
    ax2.set_title(title)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u(x,y)')

    plt.tight_layout()
    plt.show()


def plot_error(nodes, u, u_exact_func, title="Error Distribution"):
    """
    绘制误差云图
    """
    x = nodes[:, 0]
    y = nodes[:, 1]
    u_exact_vals = np.array([u_exact_func(x, y) for x, y in nodes])
    error = np.abs(u - u_exact_vals)

    triang = Triangulation(x, y)

    plt.figure(figsize=(8, 6))
    tpc = plt.tripcolor(triang, error, shading='gouraud', cmap='hot')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(tpc, label='|u_h - u_exact|')
    plt.tight_layout()
    plt.show()


def plot_3d_solution(nodes, u, title="Numerical Solution 3D"):
    """
    绘制三维曲面图
    """
    x = nodes[:, 0]
    y = nodes[:, 1]
    triang = Triangulation(x, y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(triang, u, cmap='viridis', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    plt.tight_layout()
    plt.show()


# -------------------------------
# 主程序
# -------------------------------
if __name__ == "__main__":
    # 制造解相关函数
    def u_exact(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)


    def f_func(x, y):
        return 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)


    # 测试不同网格规模
    grid_sizes = [50, 100, 200]

    for nx in grid_sizes:
        ny = nx
        print(f"\n{'=' * 65}")
        print(f"Running Q4 Poisson Solver with nx = {nx}, ny = {ny}")
        print(f"{'=' * 65}")

        start_time = time.time()

        # 1. 生成网格
        nodes, ien, nn, ne = generate_q4_mesh(nx, ny)
        print(f"单元类型: Q4")
        print(f"节点数: {nn}, 单元数: {ne}, 未知自由度数: {nn}")
        print(f"总体矩阵非零元个数: {ne * 16} (粗略估计，实际可能略少)")

        # 2. 组装刚度矩阵
        K = assemble_global_stiffness(nodes, ien, nn, ne)
        assemble_time = time.time()

        # 3. 计算右端项
        F = compute_rhs(nodes, ien, ne, nn, f_func)

        # 4. 应用边界条件
        K, F = apply_dirichlet(K, F, nodes)
        bc_time = time.time()

        # 5. 求解 - 使用 Intel MKL PARDISO
        print("调用求解器: pypardiso.spsolve (Intel MKL PARDISO)")
        u = pypardiso.spsolve(K, F)
        solve_time = time.time()

        # 6. 计算误差
        max_error, l2_error = compute_errors(nodes, u, u_exact)

        total_time = time.time()

        # 输出结果
        print(f"\n结果:")
        print(f"1. 装配时间: {assemble_time - start_time:.4f} s")
        print(f"2. 边界条件处理时间: {bc_time - assemble_time:.4f} s")
        print(f"3. 求解时间: {solve_time - bc_time:.4f} s")
        print(f"4. 总时间: {total_time - start_time:.4f} s")
        print(f"5. 相对残差: {np.linalg.norm(K @ u - F) / np.linalg.norm(F):.2e}")
        print(f"6. 节点最大误差: {max_error:.2e}")
        print(f"7. 离散 L2 相对误差: {l2_error:.2e}")

        # 7. 绘图
        plot_solution(nodes, u, title=f"Numerical Solution (nx={nx}, ny={nx})")
        plot_error(nodes, u, u_exact, title=f"Error Distribution (nx={nx}, ny={nx})")
        plot_3d_solution(nodes, u, title=f"Numerical Solution 3D (nx={nx}, ny={nx})")

        print(f"\n{'=' * 65}")
        print(f"Finished nx = {nx}, ny = {ny}")
        print(f"{'=' * 65}\n")