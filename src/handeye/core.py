"""
手眼标定核心算法 — 求解 AX = XB 问题

三种经典方法:
    1. Tsai-Lenz 方法: 轴角分离法, 先解旋转再解平移
    2. Navy 方法:  对偶四元数法, 用 SVD 同时求解旋转和平移
    3. Park 方法:   李代数法, 将旋转和平移解耦为两个线性系统

数学背景:
    对于 N 个机器人姿态, 可构造 N-1 组相对运动对 (A_i, B_i)。
    手眼标定的核心方程为:
        A_i @ X = X @ B_i  (i = 1, ..., N-1)
    其中 X 为待求的 4×4 手眼变换矩阵。
"""

import numpy as np
from typing import List
from handeye.transform import (
    rotation_matrix_to_axis_angle,
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    _skew,
)


# =============================================================================
#  方法一: Tsai-Lenz (轴角分离 — 最经典)
# =============================================================================


def solve_ax_xb_tsai(A_list: List[np.ndarray], B_list: List[np.ndarray]) -> np.ndarray:
    """
    Tsai-Lenz 手眼标定方法 (1989)

    核心思想: 将 AX=XB 分解为旋转部分和平移部分分别求解。

    步骤:
    -----
    1. 旋转求解:
      从 R_A * R_X = R_X * R_B 出发, 引入修正 Rodrigues 参数:
          P_A = 2·sin(θ_A/2) · axis_A
          P_B = 2·sin(θ_B/2) · axis_B
      构建线性方程组:
          skew(P_A + P_B) · P'_X = P_A - P_B   (核心方程)
      其中 P'_X = tan(θ_X/2) · axis_X 是 Gibbs 向量 (不是修正 Rodrigues!)
      通过最小二乘求解 P'_X, 再反算 R_X。

    2. 平移求解:
      已知 R_X 后, 从平移方程:
          (R_A - I) · t_X = R_X · t_B - t_A
      堆叠成超定方程组, 最小二乘求解 t_X。

    参考
    ----
    R. Y. Tsai and R. K. Lenz, "A new technique for fully autonomous
    and efficient 3D robotics hand/eye calibration", 1989.
    """
    n = len(A_list)
    assert n >= 2, "至少需要 2 组相对运动 (对应 3 个机器人姿态)"

    # ---------- 第一步: 求解旋转 ----------
    # 构建线性方程组: C @ P_X_prime = d
    # P_X_prime = tan(θ_X/2) * axis_X  (Gibbs/Rodrigues 向量)
    C_rows = []
    d_rows = []

    for i in range(n):
        R_A = A_list[i][:3, :3]
        R_B = B_list[i][:3, :3]

        axis_A, theta_A = rotation_matrix_to_axis_angle(R_A)
        axis_B, theta_B = rotation_matrix_to_axis_angle(R_B)

        # 修正 Rodrigues 参数 P = 2·sin(θ/2)·axis (用于 A 和 B)
        P_A = 2.0 * np.sin(theta_A / 2.0) * axis_A
        P_B = 2.0 * np.sin(theta_B / 2.0) * axis_B

        # 构建: skew(P_A + P_B) · P'_X = P_B - P_A
        # (Tsai 原论文: (P_C + P_G) × P'_CG = P_C - P_G, 其中 G=A, C=B)
        C_rows.append(_skew(P_A + P_B))
        d_rows.append(P_B - P_A)

    C = np.vstack(C_rows)  # (3n, 3)
    d = np.concatenate(d_rows)  # (3n,)

    # 最小二乘求解 P'_X (Gibbs 向量)
    P_X_prime, _, _, _ = np.linalg.lstsq(C, d, rcond=None)

    # 从 Gibbs 向量恢复旋转矩阵 R_X
    # P'_X = tan(θ/2) * axis
    p_norm = np.linalg.norm(P_X_prime)
    if p_norm < 1e-12:
        R_X = np.eye(3)
    else:
        theta_X = 2.0 * np.arctan(p_norm)  # θ = 2*arctan(|P'|)
        axis_X = P_X_prime / p_norm
        R_X = axis_angle_to_rotation_matrix(axis_X, theta_X)

    # ---------- 第二步: 求解平移 ----------
    # (R_A - I) @ t_X = R_X @ t_B - t_A
    C_t_rows = []
    d_t_rows = []

    for i in range(n):
        R_A = A_list[i][:3, :3]
        t_A = A_list[i][:3, 3]
        t_B = B_list[i][:3, 3]

        C_t_rows.append(R_A - np.eye(3))
        d_t_rows.append(R_X @ t_B - t_A)

    C_t = np.vstack(C_t_rows)
    d_t = np.concatenate(d_t_rows)

    t_X, _, _, _ = np.linalg.lstsq(C_t, d_t, rcond=None)

    # ---------- 组装结果 ----------
    X = np.eye(4)
    X[:3, :3] = R_X
    X[:3, 3] = t_X
    return X


# =============================================================================
#  方法二: Navy (对偶四元数 + SVD)
# =============================================================================


def solve_ax_xb_navy(A_list: List[np.ndarray], B_list: List[np.ndarray]) -> np.ndarray:
    """
    Navy 手眼标定方法 (Chou-Kamel / 四元数 SVD)

    核心思想: 利用 q_A ⊗ q_X = q_X ⊗ q_B 构造线性约束 (L_A - R_B)@q_X = 0。

    关键: 四元数提取有符号歧义 (q 和 -q 表示同一旋转)。解决方法:
    先将所有 q_A 的符号统一 (首分量 w >= 0), 所有 q_B 也统一。
    由于 q_A_i = q_X ⊗ q_B_i ⊗ q_X* 要求一致的符号整体性,
    统一化后关系得以保持。再堆叠 SVD 求解 q_X。

    参考
    ----
    K. Daniilidis, "Hand-eye calibration using dual quaternions", 1999.
    """
    n = len(A_list)
    assert n >= 2, "至少需要 2 组相对运动"

    # ---------- 第一步: 提取四元数并统一符号 ----------
    q_A_list = []
    q_B_list = []
    for i in range(n):
        R_A = A_list[i][:3, :3]
        R_B = B_list[i][:3, :3]
        qa = rotation_matrix_to_quaternion(R_A)
        qb = rotation_matrix_to_quaternion(R_B)
        q_A_list.append(qa)
        q_B_list.append(qb)

    # 统一符号: 使所有 q_A 的 w 分量 >= 0
    for i in range(n):
        if q_A_list[i][0] < 0:
            q_A_list[i] = -q_A_list[i]
    # 同样统一所有 q_B
    for i in range(n):
        if q_B_list[i][0] < 0:
            q_B_list[i] = -q_B_list[i]

    # ---------- 第二步: 构建约束矩阵 ----------
    T_rows = []
    for i in range(n):
        wA, xA, yA, zA = q_A_list[i]
        wB, xB, yB, zB = q_B_list[i]

        L_A = np.array(
            [
                [wA, -xA, -yA, -zA],
                [xA, wA, -zA, yA],
                [yA, zA, wA, -xA],
                [zA, -yA, xA, wA],
            ]
        )
        R_B = np.array(
            [
                [wB, -xB, -yB, -zB],
                [xB, wB, zB, -yB],
                [yB, -zB, wB, xB],
                [zB, yB, -xB, wB],
            ]
        )
        T_rows.append(L_A - R_B)

    T = np.vstack(T_rows)  # (4n, 4)

    # ---------- 第三步: SVD 求解 ----------
    U, S, Vt = np.linalg.svd(T, full_matrices=False)
    q_X = Vt[-1, :]
    q_X = q_X / np.linalg.norm(q_X)
    R_X = quaternion_to_rotation_matrix(q_X)

    # ---------- 第四步: 求解平移 ----------
    C_t_rows = []
    d_t_rows = []
    for i in range(n):
        R_A = A_list[i][:3, :3]
        t_A = A_list[i][:3, 3]
        t_B = B_list[i][:3, 3]
        C_t_rows.append(R_A - np.eye(3))
        d_t_rows.append(R_X @ t_B - t_A)

    C_t = np.vstack(C_t_rows)
    d_t = np.concatenate(d_t_rows)
    t_X, _, _, _ = np.linalg.lstsq(C_t, d_t, rcond=None)

    X = np.eye(4)
    X[:3, :3] = R_X
    X[:3, 3] = t_X
    return X


# =============================================================================
#  方法三: Park (李代数线性化)
# =============================================================================


def solve_ax_xb_park(A_list: List[np.ndarray], B_list: List[np.ndarray]) -> np.ndarray:
    """
    Park 手眼标定方法 (Martin Park, 1994)

    核心思想: 对旋转方程取对数将 SO(3) 映射到 so(3),
    旋转和平移解耦为两个独立的线性最小二乘问题。

    步骤:
    -----
    1. 旋转求解:
       对方程 R_A @ R_X = R_X @ R_B 取矩阵对数:
           log(R_A) = R_X @ log(R_B) @ R_X^T
       记 α = log(R_A), β = log(R_B), 则有:
           α = R_X @ β @ R_X^T
       等价于: α = R_X @ β  (因为相似变换下李代数元素同向)
       这给出了 R_X 的最小二乘解: R_X = argmin ||α_i - R_X @ β_i||²

    2. 平移求解: 与 Tsai 方法相同。

    参考
    ----
    F. C. Park and B. J. Martin, "Robot sensor calibration:
    solving AX = XB on the Euclidean group", 1994.
    """
    n = len(A_list)
    assert n >= 2, "至少需要 2 组相对运动"

    # ---------- 第一步: 求对数映射 ----------
    alpha_list = []  # log(R_A) 的旋转向量
    beta_list = []  # log(R_B) 的旋转向量

    for i in range(n):
        R_A = A_list[i][:3, :3]
        R_B = B_list[i][:3, :3]

        # 旋转矩阵 -> 旋转向量 (对数映射)
        axis_A, angle_A = rotation_matrix_to_axis_angle(R_A)
        axis_B, angle_B = rotation_matrix_to_axis_angle(R_B)

        alpha_list.append(axis_A * angle_A)  # 3×1
        beta_list.append(axis_B * angle_B)

    alpha = np.column_stack(alpha_list)  # (3, n)
    beta = np.column_stack(beta_list)  # (3, n)

    # ---------- 求解 R_X ----------
    # 最小化 ||alpha - R_X @ beta||_F, 使用 SVD
    M = beta @ alpha.T  # (3, 3)
    U, _, Vt = np.linalg.svd(M)
    R_X = Vt.T @ U.T

    # 确保 det(R_X) = 1 (可能因数值原因得到 det = -1)
    if np.linalg.det(R_X) < 0:
        Vt[-1, :] *= -1
        R_X = Vt.T @ U.T

    # ---------- 第二步: 求解平移 ----------
    C_t_rows = []
    d_t_rows = []
    for i in range(n):
        R_A = A_list[i][:3, :3]
        t_A = A_list[i][:3, 3]
        t_B = B_list[i][:3, 3]
        C_t_rows.append(R_A - np.eye(3))
        d_t_rows.append(R_X @ t_B - t_A)

    C_t = np.vstack(C_t_rows)
    d_t = np.concatenate(d_t_rows)
    t_X, _, _, _ = np.linalg.lstsq(C_t, d_t, rcond=None)

    X = np.eye(4)
    X[:3, :3] = R_X
    X[:3, 3] = t_X
    return X


# =============================================================================
#  统一接口
# =============================================================================


def solve_hand_eye(
    A_list: List[np.ndarray],
    B_list: List[np.ndarray],
    method: str = "tsai",
) -> np.ndarray:
    """
    统一的标定接口, 根据 method 参数调用不同算法。

    参数
    ----
    A_list : 机器人相对运动列表, 每个元素为 (4, 4) 齐次矩阵
    B_list : 相机相对运动列表, 每个元素为 (4, 4) 齐次矩阵
    method : "tsai" | "navy" | "park"

    返回
    ----
    X : (4, 4) 手眼变换矩阵

    用法示例
    --------
    >>> from handeye import solve_hand_eye
    >>> X = solve_hand_eye(A_list, B_list, method="tsai")
    """
    methods = {
        "tsai": solve_ax_xb_tsai,
        "navy": solve_ax_xb_navy,
        "park": solve_ax_xb_park,
    }
    if method not in methods:
        raise ValueError(f"未知方法 '{method}', 可选: {list(methods.keys())}")

    return methods[method](A_list, B_list)
