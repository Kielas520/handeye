"""
变换工具模块 — 旋转矩阵、四元数、轴角、齐次变换矩阵的相互转换

手眼标定中所有计算均基于 4×4 齐次变换矩阵:
    T = [[R,  t],
         [0,  1]]
其中 R 是 3×3 旋转矩阵, t 是 3×1 平移向量。
"""

import numpy as np
from typing import Tuple


# =============================================================================
#  四元数 <-> 旋转矩阵
#  约定: 四元数为 (w, x, y, z), w 为实部
# =============================================================================


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    旋转矩阵 -> 四元数 (w, x, y, z)

    使用 Shepperd 方法, 数值稳定。基于旋转矩阵迹的分支判断,
    避免分母接近零时的精度损失。
    """
    assert R.shape == (3, 3), f"Expected (3,3), got {R.shape}"

    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    # 归一化, 防止数值误差累积
    return q / np.linalg.norm(q)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    四元数 (w, x, y, z) -> 旋转矩阵

    标准公式:
        R = I + 2*w*[v]× + 2*[v]×²
    其中 v = (x, y, z), [v]× 是 v 的反对称矩阵。
    """
    w, x, y, z = q / np.linalg.norm(q)
    R = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )
    return R


# =============================================================================
#  轴角 <-> 旋转矩阵
#  轴角: (axis, angle) — axis 是单位向量, angle 是弧度
# =============================================================================


def rotation_matrix_to_axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    旋转矩阵 -> 轴角表示 (axis, angle)

    Rodrigues 公式逆运算:
        cos(θ) = (trace(R) - 1) / 2
        axis = 1/(2*sin(θ)) * (R - Rᵀ) 的反对称提取

    当 θ 接近 0 或 π 时做特殊处理, 避免除零。
    """
    assert R.shape == (3, 3), f"Expected (3,3), got {R.shape}"

    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if np.isclose(theta, 0.0):
        return np.array([0.0, 0.0, 1.0]), 0.0

    if np.isclose(theta, np.pi):
        # 旋转角为 π 时, 反对称部分为零, 用特征向量求解
        # 找到 R + I 的非零列作为轴方向
        B = R + np.eye(3)
        for i in range(3):
            col = B[:, i]
            if np.linalg.norm(col) > 1e-6:
                axis = col / np.linalg.norm(col)
                return axis, theta
        return np.array([0.0, 0.0, 1.0]), theta

    # 一般情况: 从反对称部分提取轴
    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    ) / (2.0 * np.sin(theta))
    axis = axis / np.linalg.norm(axis)
    return axis, theta


def axis_angle_to_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    轴角 (axis, angle) -> 旋转矩阵

    Rodrigues 公式:
        R = I + sin(θ)·[k]× + (1 - cos(θ))·[k]×²
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R


# =============================================================================
#  齐次变换矩阵 (Homogeneous Transformation)
# =============================================================================


def compose_transforms(*Ts: np.ndarray) -> np.ndarray:
    """
    组合多个齐次变换矩阵: T = T1 @ T2 @ ... @ Tn
    即先应用 Tn, 再 T_{n-1}, ..., 最后 T1。
    """
    result = np.eye(4)
    for T in Ts:
        result = result @ T
    return result


def inverse_homogeneous(T: np.ndarray) -> np.ndarray:
    """
    齐次变换矩阵的逆

    利用分块结构快速求逆:
        T^{-1} = [[Rᵀ,  -Rᵀ·t],
                  [0,       1 ]]
    """
    R = T[:3, :3]
    t = T[:3, 3]
    inv_T = np.eye(4)
    inv_T[:3, :3] = R.T
    inv_T[:3, 3] = -R.T @ t
    return inv_T


def random_rotation_matrix(
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    """
    生成一个均匀分布的随机 3D 旋转矩阵

    方法: 随机四元数归一化后转旋转矩阵, 保证 SO(3) 上均匀分布。
    """
    if random_state is None:
        random_state = np.random
    q = random_state.randn(4)
    q = q / np.linalg.norm(q)
    return quaternion_to_rotation_matrix(q)


def random_transform(
    translation_scale: float = 0.5,
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    """
    生成一个随机的 4×4 齐次变换矩阵 (均匀旋转 + 随机平移)
    """
    if random_state is None:
        random_state = np.random
    T = np.eye(4)
    T[:3, :3] = random_rotation_matrix(random_state)
    T[:3, 3] = translation_scale * random_state.randn(3)
    return T


def _skew(v: np.ndarray) -> np.ndarray:
    """
    向量 v 的反对称矩阵 (用于叉乘)
        [v]× = [[  0, -vz,  vy],
                [ vz,   0, -vx],
                [-vy,  vx,   0]]
    """
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def log_rotation(R: np.ndarray) -> np.ndarray:
    """
    旋转矩阵的对数映射: SO(3) -> so(3)
    返回 3×1 的旋转向量 (轴角乘以角度)
    """
    axis, angle = rotation_matrix_to_axis_angle(R)
    return axis * angle
