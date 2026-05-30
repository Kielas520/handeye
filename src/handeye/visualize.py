"""
可视化工具 — 标定误差分析与坐标系绘制
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_calibration_error(
    A_list: np.ndarray,
    B_list: np.ndarray,
    X_est: np.ndarray,
    max_pairs: int = 50,
    title: str = "标定误差分布",
    save_path: str | None = None,
) -> None:
    """
    绘制每对 (A, B) 的残差: ||A@X - X@B||_F

    理想情况下该值接近零; 残差越大说明该对数据含噪或 X 不准。

    参数
    ----
    A_list, B_list : (N, 4, 4) 运动序列
    X_est : (4, 4) 标定出的手眼变换
    max_pairs : 最多绘制的数据对数量
    title : 图标题
    save_path : 若指定则保存图片
    """
    n = min(len(A_list), max_pairs)
    errors = []

    for i in range(n):
        residual = A_list[i] @ X_est - X_est @ B_list[i]
        errors.append(np.linalg.norm(residual, ord="fro"))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(n), errors, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axhline(
        y=np.mean(errors),
        color="red",
        linestyle="--",
        label=f"均值: {np.mean(errors):.4f}",
    )
    ax.set_xlabel("数据对索引")
    ax.set_ylabel("残差 Frobenius 范数")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    # 不在非交互环境下自动展示 (脚本中可能无 GUI)
    # plt.show()


def plot_coordinate_frames(
    transforms: list[np.ndarray],
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    axis_length: float = 0.15,
    title: str = "坐标系可视化",
    save_path: str | None = None,
) -> None:
    """
    在 3D 图中绘制一组齐次变换对应的坐标系

    每个坐标系以 origin 为原点, X/Y/Z 轴分别用红/绿/蓝箭头表示。

    参数
    ----
    transforms : 4×4 齐次变换矩阵列表
    labels : 每个坐标系的标签
    colors : 每个坐标系原点的颜色 (默认自动)
    axis_length : 坐标轴长度
    title : 图标题
    save_path : 若指定则保存图片
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    default_colors = plt.cm.tab10(np.linspace(0, 1, max(len(transforms), 10)))

    for i, T in enumerate(transforms):
        origin = T[:3, 3]
        R = T[:3, :3]
        color = colors[i] if colors else default_colors[i % len(default_colors)]
        label = labels[i] if labels else f"frame_{i}"

        # 绘制坐标轴箭头
        for d, c, name in zip(
            [0, 1, 2],
            ["red", "green", "blue"],
            ["X", "Y", "Z"],
        ):
            direction = R[:, d] * axis_length
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                direction[0],
                direction[1],
                direction[2],
                color=c,
                arrow_length_ratio=0.15,
                linewidth=1.5,
            )

        ax.scatter(*origin, color=color, s=50, label=label)

    # 设置等比例轴
    all_points = np.array([T[:3, 3] for T in transforms])
    center = all_points.mean(axis=0)
    max_range = np.max(np.ptp(all_points, axis=0)) / 2.0 + axis_length
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
