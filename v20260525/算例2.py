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

    # ==================== 前处理模块 ====================
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
        # 初始化自由度标识数组 (0=自由, 1=约束)
        self.ID = np.zeros((self.nnp, self.ndof), dtype=int)

        # 处理位移边界条件
        if "displacement_bc" in data:
            for bc in data["displacement_bc"]:
                node = bc["node"] - 1
                dofs = bc["dofs"]
                values = bc["values"]

                for i, (dof, value) in enumerate(zip(dofs, values)):
                    if dof <= self.ndof:
                        self.ID[node, dof - 1] = 1

        # 初始化位移和载荷向量
        n = self.nnp * self.ndof
        self.d = np.zeros(n)
        self.f = np.zeros(n)

        # 设置已知位移
        if "displacement_bc" in data:
            for bc in data["displacement_bc"]:
                node = bc["node"] - 1
                dofs = bc["dofs"]
                values = bc["values"]

                for i, (dof, value) in enumerate(zip(dofs, values)):
                    if dof <= self.ndof:
                        idx = node * self.ndof + (dof - 1)
                        self.d[idx] = value

        # 处理节点载荷
        if "nodal_loads" in data:
            for load in data["nodal_loads"]:
                node = load["node"] - 1
                forces = load["forces"]

                for dof, force in enumerate(forces):
                    if dof < self.ndof and abs(force) > 1e-10:
                        idx = node * self.ndof + dof
                        self.f[idx] = force

    # ==================== 单元分析模块 ====================
    def compute_element_stiffness(self, e: int) -> np.ndarray:
        """单元分析：计算单元刚度矩阵"""
        # 获取单元节点
        node_i, node_j = self.IEN[e]

        # 根据空间维度获取节点坐标
        if self.nsd == 1:
            # 一维情况
            xi = self.X[node_i][0]
            xj = self.X[node_j][0]
            L = abs(xj - xi)
            c = 1.0
            s = 0.0
        elif self.nsd == 2:
            # 二维情况
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

        # 计算单元刚度矩阵
        E = self.E[e]
        A = self.A[e]

        if self.nsd == 1:
            # 一维杆单元
            ke = (E * A / L) * np.array([[1, -1], [-1, 1]])
        else:
            # 二维桁架单元
            # 局部刚度矩阵
            k_local = (E * A / L) * np.array([[1, -1], [-1, 1]])

            # 全局刚度矩阵（使用变换矩阵）
            T = np.array([[c, s, 0, 0],
                          [0, 0, c, s]])
            ke = T.T @ k_local @ T

        return ke, L, c, s

    # ==================== 对号矩阵和组装模块 ====================
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

    def assemble_global_stiffness(self):
        """组装总体刚度矩阵"""
        n = self.nnp * self.ndof
        self.K = np.zeros((n, n))

        print("\n开始组装总体刚度矩阵...")
        for e in range(self.nel):
            ke, L, c, s = self.compute_element_stiffness(e)

            # 直接组装
            for a in range(ke.shape[0]):
                i = self.LM[e, a]
                for b in range(ke.shape[1]):
                    j = self.LM[e, b]
                    self.K[i, j] += ke[a, b]

            if e < 3:  # 只显示前3个单元的组装信息
                print(f"\n  单元{e + 1}: 长度 = {L:.4f}, 方向余弦(c,s) = ({c:.4f}, {s:.4f})")
                print(f"  单元刚度矩阵 (形状 {ke.shape}):")
                for row in ke:
                    print(f"    {row}")

        # 打印总体刚度矩阵
        print(f"\n总体刚度矩阵 (形状 {self.K.shape}):")
        for i in range(self.K.shape[0]):
            print(f"  ", end="")
            for j in range(self.K.shape[1]):
                print(f"{self.K[i, j]:12.6f}", end="")
            print()

        # 检查对称性
        is_symmetric = np.allclose(self.K, self.K.T, atol=1e-10)
        print(f"\n总体刚度矩阵对称性检查: {'对称' if is_symmetric else '不对称'}")

        # 检查奇异性和行列式
        det_before = np.linalg.det(self.K)
        rank_before = np.linalg.matrix_rank(self.K)
        print(f"施加边界条件前总体刚度矩阵行列式: {det_before:.6e}")
        print(f"施加边界条件前总体刚度矩阵秩: {rank_before}/{self.K.shape[0]}")

        if abs(det_before) < 1e-10:
            print("✓ 检查通过: 施加边界条件前总体刚度矩阵奇异（行列式接近零）")
        else:
            print("警告: 施加边界条件前总体刚度矩阵应该奇异！")

        return self.K

    # ==================== 方程求解模块 ====================
    def apply_boundary_conditions(self, method: str = "reduction"):
        """方程求解：施加边界条件并求解"""
        print("\n" + "=" * 60)
        print(f"模块3: 边界条件处理 ({method}法)")
        print("=" * 60)

        n = self.nnp * self.ndof

        # 创建自由度标识向量
        id_vec = self.ID.flatten()

        if method == "reduction":
            d, R, K_reduced = self.solve_by_reduction(id_vec)
        elif method == "penalty":
            d, R, K_reduced = self.solve_by_penalty(id_vec)
        elif method == "modification":
            d, R, K_reduced = self.solve_by_modification(id_vec)
        else:
            raise ValueError(f"未知的方法: {method}")

        return d, R, K_reduced, R  # 返回4个值：位移、反力、缩减矩阵、反力

    def solve_by_reduction(self, id_vec: np.ndarray):
        """缩减法求解"""
        # 分离自由度和约束自由度
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

        d_E = self.d[fixed_dofs]  # 已知位移
        f_F = self.f[free_dofs]  # 自由度的载荷

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

        # 打印载荷向量
        print(f"\n载荷向量 f_F: {f_F}")
        print(f"已知位移 d_E: {d_E}")

        # 求解方程: K_FF * d_F = f_F - K_FE * d_E
        rhs = f_F - K_FE @ d_E
        print(f"\n方程右边项: K_FF * d_F = {rhs}")

        # 求解未知位移
        d_F = la.solve(K_FF, rhs)

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

    def solve_by_penalty(self, id_vec: np.ndarray, penalty: float = 1e12):
        """罚函数法求解"""
        print(f"使用罚函数法，罚系数: {penalty:.1e}")

        K_modified = self.K.copy()
        f_modified = self.f.copy()

        # 对约束自由度施加大数惩罚
        fixed_dofs = np.where(id_vec == 1)[0]

        for dof in fixed_dofs:
            K_modified[dof, dof] += penalty
            f_modified[dof] = penalty * self.d[dof]

        # 求解
        self.d = la.solve(K_modified, f_modified)

        # 计算反力
        R = self.K @ self.d - self.f

        return self.d, R, K_modified

    def solve_by_modification(self, id_vec: np.ndarray):
        """修改法求解"""
        K_modified = self.K.copy()
        f_modified = self.f.copy()

        fixed_dofs = np.where(id_vec == 1)[0]

        for dof in fixed_dofs:
            # 修改刚度矩阵
            K_modified[dof, :] = 0
            K_modified[:, dof] = 0
            K_modified[dof, dof] = 1

            # 修改载荷向量
            f_modified[dof] = self.d[dof]

        # 求解
        self.d = la.solve(K_modified, f_modified)

        # 计算反力
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
            # 计算单元属性
            ke, L, c, s = self.compute_element_stiffness(e)

            if self.nsd == 1:
                # 一维杆单元
                # 获取单元位移
                node_i, node_j = self.IEN[e]
                d1 = self.d[node_i * self.ndof]
                d2 = self.d[node_j * self.ndof]

                # 计算应力和轴力
                strain = (d2 - d1) / L
                stress = self.E[e] * strain
                force = stress * self.A[e]

                stresses.append(stress)
                forces.append(force)

                # 输出结果
                print(f"{e + 1:<6} {L:<10.4f} {'':<20} {stress:<15.6f} {force:<15.6f}")

            else:
                # 二维桁架单元
                # 获取单元位移
                de = np.zeros(4)
                for i in range(4):
                    de[i] = self.d[self.LM[e, i]]

                # 计算应力
                B = np.array([-c, -s, c, s]) / L
                stress = self.E[e] * np.dot(B, de)
                force = stress * self.A[e]

                stresses.append(stress)
                forces.append(force)

                # 输出结果
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

        # 4. 求解方程
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
            [1.0, 0.0],  # 节点1: (1, 0)
            [0.0, 0.0],  # 节点2: (0, 0)
            [1.0, 1.0]  # 节点3: (1, 1)
        ],
        "IEN": [
            [1, 3],  # 单元1：1-3
            [2, 3]  # 单元2：2-3
        ],
        "E": [1.0, 1.0],
        "A": [1.0, 1.0],
        "displacement_bc": [
            {
                "node": 1,
                "dofs": [1, 2],  # UX, UY
                "values": [0.0, 0.0]
            },
            {
                "node": 2,
                "dofs": [1, 2],  # UX, UY
                "values": [0.0, 0.0]
            }
        ],
        "nodal_loads": [
            {
                "node": 3,
                "forces": [10.0, 0.0]  # FX=10, FY=0
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


def analyze_example2():
    """分析算例2：二维两杆桁架结构"""
    print("=" * 60)
    print("算例2：二维两杆桁架结构分析")
    print("=" * 60)

    # 创建输入文件
    input_file = create_example2_input_file()

    # 创建并运行有限元分析
    fea = TrussFEA()

    try:
        d, R, stresses, forces, K_reduced = fea.run_analysis(input_file, method="reduction")

        # 验证结果
        print("\n" + "=" * 60)
        print("验证结果")
        print("=" * 60)

        # 检查节点3位移
        expected_u3 = 38.284271
        expected_v3 = -10.000000
        actual_u3 = d[4]  # 节点3的UX位移
        actual_v3 = d[5]  # 节点3的UY位移

        print(f"\n节点3位移验证:")
        print(f"计算值: u3 = {actual_u3:.6f}, v3 = {actual_v3:.6f}")
        print(f"期望值: u3 = {expected_u3:.6f}, v3 = {expected_v3:.6f}")

        error_u3 = abs(actual_u3 - expected_u3)
        error_v3 = abs(actual_v3 - expected_v3)

        if error_u3 < 0.001:
            print(f"✓ u3 正确 (误差: {error_u3:.6f})")
        else:
            print(f"✗ u3 错误 (误差: {error_u3:.6f})")

        if error_v3 < 0.001:
            print(f"✓ v3 正确 (误差: {error_v3:.6f})")
        else:
            print(f"✗ v3 错误 (误差: {error_v3:.6f})")

        # 检查单元应力
        if len(stresses) >= 2:
            stress1 = stresses[0]
            stress2 = stresses[1]
            expected_stress1 = -10.000000
            expected_stress2 = 14.142136

            print(f"\n单元应力验证:")
            print(f"单元1应力: 计算值 = {stress1:.6f}, 期望值 = {expected_stress1:.6f}")
            print(f"单元2应力: 计算值 = {stress2:.6f}, 期望值 = {expected_stress2:.6f}")

            error1 = abs(stress1 - expected_stress1)
            error2 = abs(stress2 - expected_stress2)

            if error1 < 0.001:
                print(f"✓ 单元1应力正确 (误差: {error1:.6f})")
            else:
                print(f"✗ 单元1应力错误 (误差: {error1:.6f})")

            if error2 < 0.001:
                print(f"✓ 单元2应力正确 (误差: {error2:.6f})")
            else:
                print(f"✗ 单元2应力错误 (误差: {error2:.6f})")

        # 检查总体刚度矩阵对称性
        K = fea.K
        is_symmetric = np.allclose(K, K.T, atol=1e-10)
        print(f"\n总体刚度矩阵对称性: {'✓ 对称' if is_symmetric else '✗ 不对称'}")

        # 检查施加边界条件前是否奇异
        det_before = np.linalg.det(K)
        if abs(det_before) < 1e-10:
            print(f"✓ 施加边界条件前总体刚度矩阵奇异 (行列式: {det_before:.6e})")
        else:
            print(f"✗ 施加边界条件前总体刚度矩阵应该奇异但非奇异 (行列式: {det_before:.6e})")

        return d, R, stresses, forces

    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def main():
    """主函数"""
    print("=" * 60)
    print("有限元分析程序 - 算例2：二维两杆桁架结构")
    print("=" * 60)

    try:
        # 分析算例2
        d, R, stresses, forces = analyze_example2()

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行错误: {e}")


if __name__ == "__main__":
    main()