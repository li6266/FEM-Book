import numpy as np
import json
from typing import Dict, List, Tuple, Any
import numpy.linalg as la


class TrussFEA:
    """一维/二维桁架结构有限元分析程序"""

    def __init__(self, input_file: str = None):
        """初始化有限元程序"""
        self.title = ""
        self.nsd = 0  # 空间维度
        self.ndof = 0  # 每个节点的自由度数
        self.nnp = 0  # 节点总数
        self.nel = 0  # 单元总数
        self.nen = 0  # 每个单元的节点数

        # 模型数据
        self.X = None  # 节点坐标 [nnp, nsd]
        self.IEN = None  # 单元连接数组 [nel, nen]
        self.E = None  # 弹性模量 [nel]
        self.A = None  # 截面积 [nel]
        self.EA_L = None  # EA/L值 [nel]

        # 边界条件
        self.ID = None  # 自由度标识数组
        self.d = None  # 节点位移
        self.f = None  # 节点载荷

        # 全局矩阵
        self.K = None  # 总体刚度矩阵
        self.LM = None  # 对号矩阵

        if input_file:
            self.read_input(input_file)

    # ==================== 模型输入模块 ====================
    def read_input(self, input_file: str):
        """前处理：从JSON文件读取模型数据"""
        print("=" * 60)
        print("模块1: 前处理 - 读取输入文件")
        print("=" * 60)

        with open(input_file, 'r') as f:
            data = json.load(f)

        # 读取基本信息
        self.title = data.get("Title", "Untitled")
        self.nsd = data["nsd"]
        self.ndof = data["ndof"]
        self.nnp = data["nnp"]
        self.nel = data["nel"]
        self.nen = data["nen"]

        print(f"问题标题: {self.title}")
        print(f"空间维度: {self.nsd}")
        print(f"每个节点自由度数: {self.ndof}")
        print(f"节点总数: {self.nnp}")
        print(f"单元总数: {self.nel}")
        print(f"每个单元节点数: {self.nen}")

        # 读取节点坐标
        self.X = np.array(data["X"])
        print(f"\n节点坐标 (形状 {self.X.shape}):")
        for i, coord in enumerate(self.X):
            if self.nsd == 1:
                print(f"  节点{i + 1}: x = {coord[0]:.2f}")
            else:
                print(f"  节点{i + 1}: ({coord[0]:.2f}, {coord[1]:.2f})")

        # 读取单元连接
        self.IEN = np.array(data["IEN"]) - 1  # 转换为0-based索引
        print(f"\n单元连接 (0-based索引):")
        for e, conn in enumerate(self.IEN):
            print(f"  单元{e + 1}: 节点 {conn[0] + 1} - 节点 {conn[1] + 1}")

        # 读取材料属性
        if "EA_L" in data:
            self.EA_L = np.array(data["EA_L"])
            self.E = self.EA_L.copy()
            self.A = np.ones_like(self.EA_L)
            print(f"\n单元轴向刚度 EA/L:")
            for e in range(self.nel):
                print(f"  单元{e + 1}: EA/L = {self.EA_L[e]:.2f}")
        else:
            self.E = np.array(data["E"])
            self.A = np.array(data["A"])
            print(f"\n材料属性 (弹性模量和截面积):")
            for e in range(self.nel):
                print(f"  单元{e + 1}: E={self.E[e]:.2f}, A={self.A[e]:.2f}")

        # 处理边界条件
        self.process_boundary_conditions(data)
        print("\n前处理完成!")

    def process_boundary_conditions(self, data: Dict):
        """处理边界条件和载荷"""
        self.ID = np.zeros((self.nnp, self.ndof), dtype=int)

        if "displacement_bc" in data:
            for bc in data["displacement_bc"]:
                node = bc["node"] - 1
                dofs = bc["dofs"]
                values = bc["values"]

                for i, (dof, value) in enumerate(zip(dofs, values)):
                    if dof <= self.ndof:
                        self.ID[node, dof - 1] = 1

        n = self.nnp * self.ndof
        self.d = np.zeros(n)
        self.f = np.zeros(n)

        if "displacement_bc" in data:
            for bc in data["displacement_bc"]:
                node = bc["node"] - 1
                dofs = bc["dofs"]
                values = bc["values"]

                for i, (dof, value) in enumerate(zip(dofs, values)):
                    if dof <= self.ndof:
                        idx = node * self.ndof + (dof - 1)
                        self.d[idx] = value

        if "nodal_loads" in data:
            for load in data["nodal_loads"]:
                node = load["node"] - 1
                forces = load["forces"]

                for dof, force in enumerate(forces):
                    if dof < self.ndof and abs(force) > 1e-10:
                        idx = node * self.ndof + dof
                        self.f[idx] = force

    # ==================== 单元分析模块 ====================
    def compute_element_stiffness(self, e: int) -> Tuple[np.ndarray, float, float, float]:
        """单元分析：计算单元刚度矩阵"""
        node_i, node_j = self.IEN[e]

        if self.nsd == 1:
            xi = self.X[node_i][0]
            xj = self.X[node_j][0]
            L = abs(xj - xi)
            c = 1.0
            s = 0.0
        elif self.nsd == 2:
            xi, yi = self.X[node_i]
            xj, yj = self.X[node_j]

            dx = xj - xi
            dy = yj - yi
            L = np.sqrt(dx ** 2 + dy ** 2)
            c = dx / L
            s = dy / L
        else:
            raise ValueError(f"不支持的空间维度: {self.nsd}")

        if L < 1e-10:
            raise ValueError(f"单元{e + 1}长度为零!")

        if hasattr(self, 'EA_L') and self.EA_L is not None:
            EA_over_L = self.EA_L[e]
        else:
            E = self.E[e]
            A = self.A[e]
            EA_over_L = E * A / L

        if self.nsd == 1:
            ke = EA_over_L * np.array([[1, -1], [-1, 1]])
        else:
            k_local = EA_over_L * np.array([[1, -1], [-1, 1]])
            T = np.array([[c, s, 0, 0],
                          [0, 0, c, s]])
            ke = T.T @ k_local @ T

        return ke, L, c, s

    # ==================== 对号矩阵模块 ====================
    def generate_LM(self):
        """生成对号矩阵LM"""
        print("\n" + "=" * 60)
        print("模块2: 对号矩阵和直接组装")
        print("=" * 60)

        n = self.nen * self.ndof
        self.LM = np.zeros((self.nel, n), dtype=int)

        print("对号矩阵 (单元自由度到总体自由度的映射):")
        for e in range(self.nel):
            for i in range(self.nen):
                node = self.IEN[e, i]
                for j in range(self.ndof):
                    local_idx = i * self.ndof + j
                    global_idx = node * self.ndof + j
                    self.LM[e, local_idx] = global_idx

            print(f"  单元{e + 1}: {self.LM[e]}")

        return self.LM

    # ==================== 总体组装模块 ====================
    def assemble_global_stiffness(self):
        """组装总体刚度矩阵"""
        n = self.nnp * self.ndof
        self.K = np.zeros((n, n))

        print("\n开始组装总体刚度矩阵...")
        for e in range(self.nel):
            ke, L, c, s = self.compute_element_stiffness(e)

            for a in range(ke.shape[0]):
                i = self.LM[e, a]
                for b in range(ke.shape[1]):
                    j = self.LM[e, b]
                    self.K[i, j] += ke[a, b]

            if e < 3:
                print(f"\n  单元{e + 1}: 长度 = {L:.4f}, 方向余弦(c,s) = ({c:.4f}, {s:.4f})")
                print(f"  单元刚度矩阵 (形状 {ke.shape}):")
                for row in ke:
                    print(f"    {row}")

        print(f"\n总体刚度矩阵 (形状 {self.K.shape}):")
        for i in range(self.K.shape[0]):
            print(f"  ", end="")
            for j in range(self.K.shape[1]):
                print(f"{self.K[i, j]:12.6f}", end="")
            print()

        is_symmetric = np.allclose(self.K, self.K.T, atol=1e-10)
        print(f"\n总体刚度矩阵对称性检查: {'对称' if is_symmetric else '不对称'}")

        det_before = np.linalg.det(self.K)
        rank_before = np.linalg.matrix_rank(self.K)
        print(f"施加边界条件前总体刚度矩阵行列式: {det_before:.6e}")
        print(f"施加边界条件前总体刚度矩阵秩: {rank_before}/{self.K.shape[0]}")

        if abs(det_before) < 1e-10:
            print("✓ 检查通过: 施加边界条件前总体刚度矩阵奇异（行列式接近零）")
        else:
            print("警告: 施加边界条件前总体刚度矩阵应该奇异！")

        return self.K

    # ==================== LDLT 求解器 ====================
    def ldlt_decomposition(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        LDL^T 分解：将对称正定矩阵 A 分解为 L * D * L^T
        其中 L 是单位下三角矩阵，D 是对角矩阵

        参数:
            A: 对称正定矩阵 (n x n)
        返回:
            L: 单位下三角矩阵 (n x n)
            D: 对角矩阵的对角元素 (n,)
        """
        n = A.shape[0]
        L = np.eye(n)  # 初始化为单位矩阵
        D = np.zeros(n)

        print(f"\nLDL^T 分解过程 (矩阵大小: {n}x{n}):")
        print("-" * 30)

        for j in range(n):
            # 计算 D[j]
            sum_d = 0.0
            for k in range(j):
                sum_d += L[j, k] ** 2 * D[k]
            D[j] = A[j, j] - sum_d

            # 计算 L[i, j] (i > j)
            for i in range(j + 1, n):
                sum_l = 0.0
                for k in range(j):
                    sum_l += L[i, k] * L[j, k] * D[k]
                if abs(D[j]) < 1e-12:
                    raise ValueError(f"D[{j}] 接近零，矩阵可能奇异或非正定")
                L[i, j] = (A[i, j] - sum_l) / D[j]

            print(f"  Step {j + 1}: D[{j}] = {D[j]:.6f}")

        print("\nL 矩阵:")
        for i in range(n):
            print(f"  {L[i]}")
        print(f"D 对角元: {D}")

        return L, D

    def ldlt_solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        使用 LDL^T 分解求解线性方程组 Ax = b

        步骤:
        1. A = L * D * L^T  (LDL^T 分解)
        2. Ly = b           (前代)
        3. Dz = y           (对角缩放)
        4. L^T x = z        (回代)

        参数:
            A: 系数矩阵 (n x n)
            b: 右端项向量 (n,)
        返回:
            x: 解向量 (n,)
        """
        print("\n" + "=" * 60)
        print("使用 LDL^T 分解求解线性方程组")
        print("=" * 60)

        n = A.shape[0]

        # 1. LDL^T 分解
        L, D = self.ldlt_decomposition(A)

        # 2. 前代: Ly = b
        print("\n前代过程 (Ly = b):")
        y = np.zeros(n)
        for i in range(n):
            sum_y = 0.0
            for j in range(i):
                sum_y += L[i, j] * y[j]
            y[i] = b[i] - sum_y
            print(f"  y[{i}] = {y[i]:.6f}")

        # 3. 对角缩放: Dz = y
        print("\n对角缩放 (Dz = y):")
        z = np.zeros(n)
        for i in range(n):
            if abs(D[i]) < 1e-12:
                raise ValueError(f"D[{i}] 接近零，矩阵奇异")
            z[i] = y[i] / D[i]
            print(f"  z[{i}] = {z[i]:.6f}")

        # 4. 回代: L^T x = z
        print("\n回代过程 (L^T x = z):")
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            sum_x = 0.0
            for j in range(i + 1, n):
                sum_x += L[j, i] * x[j]
            x[i] = z[i] - sum_x
            print(f"  x[{i}] = {x[i]:.6f}")

        return x

    # ==================== 自由度分块模块 ====================
    def solve_by_reduction(self, id_vec: np.ndarray):
        """缩减法求解，使用 LDL^T 分解"""
        free_dofs = np.where(id_vec == 0)[0]
        fixed_dofs = np.where(id_vec == 1)[0]

        n_free = len(free_dofs)
        n_fixed = len(fixed_dofs)

        print(f"自由度数: {n_free}, 约束自由度数: {n_fixed}")
        print(f"自由度编号: {free_dofs}")
        print(f"约束自由度编号: {fixed_dofs}")

        # 分割刚度矩阵和载荷向量
        K_FF = self.K[np.ix_(free_dofs, free_dofs)]
        K_FE = self.K[np.ix_(free_dofs, fixed_dofs)]

        d_E = self.d[fixed_dofs]
        f_F = self.f[free_dofs]

        # 检查缩减矩阵的奇异性
        det_reduced = np.linalg.det(K_FF)
        print(f"\n缩减刚度矩阵行列式: {det_reduced:.6e}")

        if abs(det_reduced) < 1e-10:
            print("警告: 缩减刚度矩阵奇异！")
        else:
            print("缩减刚度矩阵非奇异")

        # 打印缩减矩阵
        print(f"\n缩减刚度矩阵 K_FF (形状 {K_FF.shape}):")
        for i in range(K_FF.shape[0]):
            print(f"  ", end="")
            for j in range(K_FF.shape[1]):
                print(f"{K_FF[i, j]:12.6f}", end="")
            print()

        print(f"\n载荷向量 f_F: {f_F}")
        print(f"已知位移 d_E: {d_E}")

        # 计算右端项
        rhs = f_F - K_FE @ d_E
        print(f"\n方程右边项: K_FF * d_F = {rhs}")

        # 使用 LDL^T 分解求解
        d_F = self.ldlt_solve(K_FF, rhs)

        # 重构完整位移向量
        self.d[free_dofs] = d_F

        # 计算约束反力
        R = np.zeros_like(self.f)
        R[fixed_dofs] = self.K[fixed_dofs, :] @ self.d - self.f[fixed_dofs]

        print(f"\n求解完成:")
        print(f"  未知位移: d_F = {d_F}")
        print(f"  完整位移向量: {self.d}")
        print(f"  约束反力: R = {R[fixed_dofs]}")

        return self.d, R, K_FF

    # ==================== 方程求解模块 ====================
    def apply_boundary_conditions(self, method: str = "reduction"):
        """方程求解：施加边界条件并求解"""
        print("\n" + "=" * 60)
        print(f"模块3: 边界条件处理 ({method}法)")
        print("=" * 60)

        id_vec = self.ID.flatten()

        if method == "reduction":
            d, R, K_reduced = self.solve_by_reduction(id_vec)
        elif method == "penalty":
            d, R, K_reduced = self.solve_by_penalty(id_vec)
        elif method == "modification":
            d, R, K_reduced = self.solve_by_modification(id_vec)
        else:
            raise ValueError(f"未知的方法: {method}")

        return d, R, K_reduced, R

    def solve_by_penalty(self, id_vec: np.ndarray, penalty: float = 1e12):
        """罚函数法求解"""
        print(f"使用罚函数法，罚系数: {penalty:.1e}")

        K_modified = self.K.copy()
        f_modified = self.f.copy()

        fixed_dofs = np.where(id_vec == 1)[0]

        for dof in fixed_dofs:
            K_modified[dof, dof] += penalty
            f_modified[dof] = penalty * self.d[dof]

        self.d = la.solve(K_modified, f_modified)
        R = self.K @ self.d - self.f

        return self.d, R, K_modified

    def solve_by_modification(self, id_vec: np.ndarray):
        """修改法求解"""
        K_modified = self.K.copy()
        f_modified = self.f.copy()

        fixed_dofs = np.where(id_vec == 1)[0]

        for dof in fixed_dofs:
            K_modified[dof, :] = 0
            K_modified[:, dof] = 0
            K_modified[dof, dof] = 1
            f_modified[dof] = self.d[dof]

        self.d = la.solve(K_modified, f_modified)
        R = self.K @ self.d - self.f

        return self.d, R, K_modified

    # ==================== 后处理模块 ====================
    def compute_stresses(self):
        """后处理：计算单元应力"""
        print("\n" + "=" * 60)
        print("模块4: 单元应力计算")
        print("=" * 60)

        if self.nsd == 1:
            print(f"{'单元':<6} {'长度':<10} {'应力':<15} {'轴力':<15}")
            print("-" * 50)
        else:
            print(f"{'单元':<6} {'长度':<10} {'方向余弦(c,s)':<20} {'应力':<15} {'轴力':<15}")
            print("-" * 70)

        stresses = []
        forces = []

        for e in range(self.nel):
            ke, L, c, s = self.compute_element_stiffness(e)

            if self.nsd == 1:
                node_i, node_j = self.IEN[e]
                d1 = self.d[node_i * self.ndof]
                d2 = self.d[node_j * self.ndof]

                strain = (d2 - d1) / L
                stress = self.E[e] * strain
                force = stress * self.A[e]

                stresses.append(stress)
                forces.append(force)

                print(f"{e + 1:<6} {L:<10.4f} {'':<20} {stress:<15.6f} {force:<15.6f}")

            else:
                de = np.zeros(4)
                for i in range(4):
                    de[i] = self.d[self.LM[e, i]]

                B = np.array([-c, -s, c, s]) / L
                stress = self.E[e] * np.dot(B, de)
                force = stress * self.A[e]

                stresses.append(stress)
                forces.append(force)

                print(f"{e + 1:<6} {L:<10.4f} ({c:.4f}, {s:.4f}){'':<8} {stress:<15.6f} {force:<15.6f}")

        return np.array(stresses), np.array(forces)

    def print_displacements(self):
        """打印节点位移"""
        print("\n" + "=" * 60)
        print("节点位移结果")
        print("=" * 60)

        if self.nsd == 1:
            print(f"{'节点':<6} {'UX':<15}")
            print("-" * 25)
            for i in range(self.nnp):
                ux = self.d[i * self.ndof]
                print(f"{i + 1:<6} {ux:<15.6f}")
        else:
            print(f"{'节点':<6} {'UX':<20} {'UY':<20}")
            print("-" * 50)
            for i in range(self.nnp):
                ux = self.d[i * self.ndof]
                uy = self.d[i * self.ndof + 1]
                print(f"{i + 1:<6} {ux:<20.6f} {uy:<20.6f}")

    def print_reactions(self, R: np.ndarray):
        """打印约束反力"""
        print("\n" + "=" * 60)
        print("约束反力结果")
        print("=" * 60)

        if self.nsd == 1:
            print(f"{'节点':<6} {'RX':<15}")
            print("-" * 25)
            for i in range(self.nnp):
                rx = R[i * self.ndof]
                if abs(rx) > 1e-10:
                    print(f"{i + 1:<6} {rx:<15.6f}")
        else:
            print(f"{'节点':<6} {'RX':<20} {'RY':<20}")
            print("-" * 50)
            for i in range(self.nnp):
                rx = R[i * self.ndof]
                ry = R[i * self.ndof + 1]
                if abs(rx) > 1e-10 or abs(ry) > 1e-10:
                    print(f"{i + 1:<6} {rx:<20.6f} {ry:<20.6f}")

    # ==================== 主程序流程 ====================
    def run_analysis(self, input_file: str, method: str = "reduction"):
        """执行完整的有限元分析流程"""
        print("=" * 60)
        print("开始有限元分析")
        print("=" * 60)

        # 1. 前处理
        self.read_input(input_file)

        # 2. 生成对号矩阵
        self.generate_LM()

        # 3. 组装总体刚度矩阵
        self.assemble_global_stiffness()

        # 4. 求解方程（使用 LDL^T 分解）
        d, R, K_reduced, _ = self.apply_boundary_conditions(method)

        # 5. 后处理
        self.print_displacements()
        self.print_reactions(R)
        stresses, forces = self.compute_stresses()

        print("\n" + "=" * 60)
        print("有限元分析完成!")
        print("=" * 60)

        return d, R, stresses, forces, K_reduced


# ==================== 创建输入文件 ====================
def create_example1_input_file():
    """创建示例输入文件1：一维两单元杆结构"""
    example_data = {
        "Title": "算例1：一维两单元杆结构",
        "nsd": 1,
        "ndof": 1,
        "nnp": 3,
        "nel": 2,
        "nen": 2,
        "X": [
            [0.0],
            [1.0],
            [2.0]
        ],
        "IEN": [
            [1, 2],
            [2, 3]
        ],
        "E": [1.0, 1.0],
        "A": [1.0, 1.0],
        "EA_L": [100.0, 200.0],
        "displacement_bc": [
            {
                "node": 1,
                "dofs": [1],
                "values": [0.0]
            }
        ],
        "nodal_loads": [
            {
                "node": 2,
                "forces": [0.0]
            },
            {
                "node": 3,
                "forces": [10.0]
            }
        ]
    }

    with open("example1.json", "w") as f:
        json.dump(example_data, f, indent=2)

    print("已创建输入文件1: example1.json")
    return "example1.json"


def create_example2_input_file():
    """创建示例输入文件2：二维两杆桁架结构"""
    example_data = {
        "Title": "算例2：二维两杆桁架结构",
        "nsd": 2,
        "ndof": 2,
        "nnp": 3,
        "nel": 2,
        "nen": 2,
        "X": [
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0]
        ],
        "IEN": [
            [1, 3],
            [2, 3]
        ],
        "E": [1.0, 1.0],
        "A": [1.0, 1.0],
        "displacement_bc": [
            {
                "node": 1,
                "dofs": [1, 2],
                "values": [0.0, 0.0]
            },
            {
                "node": 2,
                "dofs": [1, 2],
                "values": [0.0, 0.0]
            }
        ],
        "nodal_loads": [
            {
                "node": 3,
                "forces": [10.0, 0.0]
            }
        ],
        "expected_results": {
            "node3_displacement": [38.284271, -10.000000],
            "stress_unit1": -10.000000,
            "stress_unit2": 14.142136
        }
    }

    with open("example2.json", "w") as f:
        json.dump(example_data, f, indent=2)

    print("已创建输入文件2: example2.json")
    return "example2.json"


def main():
    """主函数"""
    print("=" * 60)
    print("有限元分析程序 - 使用 LDL^T 分解求解")
    print("=" * 60)

    try:
        # 可以选择运行哪个算例
        choice = input("请选择算例 (1 或 2): ")

        if choice == "1":
            print("\n运行算例1：一维两单元杆结构")
            input_file = create_example1_input_file()
        elif choice == "2":
            print("\n运行算例2：二维两杆桁架结构")
            input_file = create_example2_input_file()
        else:
            print("无效选择，默认运行算例1")
            input_file = create_example1_input_file()

        fea = TrussFEA()
        d, R, stresses, forces, _ = fea.run_analysis(input_file, method="reduction")

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()