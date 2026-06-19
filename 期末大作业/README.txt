README — 一维稳态对流扩散方程有限元求解
项目简介
本项目使用有限元法（FEM）求解一维稳态对流扩散方程，对比了三种数值格式在不同 Peclet 数下的表现：
标准 Galerkin 方法（Standard G）
迎风格式（Upwind）
SUPG 方法（Streamline Upwind Petrov-Galerkin）
此外，还分析了标准 Galerkin 总体矩阵的对称性和正定性，解释了对流占优时数值振荡的成因。

环境依赖
运行本项目需要安装以下 Python 库：
bash
pip install numpy matplotlib
操作系统：Windows / macOS / Linux 均可

文件说明
文件名
main2.py
主程序文件，包含全部求解、绘图和分析代码

运行方法
方式一：命令行运行
打开终端（命令提示符），进入文件所在目录，执行：
bash
python main2.py
方式二：IDE 运行
在 PyCharm、VS Code、Spyder 等 IDE 中打开 main2.py，点击运行按钮即可。
程序输出
运行后将依次输出以下内容：
控制台输出
任务 1 & 2 — 数值结果表格（共 6 张表）
Pe = 0.1 时三种方法的节点坐标、数值解、精确解和误差
Pe = 3.0 时三种方法的节点坐标、数值解、精确解和误差
任务 3 — 误差分析
各方法在不同 Pe 数下的最大误差
任务 4 — 矩阵性质分析
标准 Galerkin 总体矩阵（21×21，分块显示）
对称性检查结果（Frobenius 范数误差）
正定性检查结果（特征值分布、负特征值个数）
条件数和矩阵稀疏度
图形输出
程序会弹出两个子图的对比图：
左图：Pe = 0.1（扩散主导），三种方法均平滑
右图：Pe = 3.0（对流占优），标准 Galerkin 出现振荡，SUPG 最优
参数说明
如需修改计算参数，可在 main()函数中找到以下变量进行调整：
python
L = 1           # 计算域长度
nel = 20        # 单元数量（增大可提高精度）
v = 1           # 对流速度