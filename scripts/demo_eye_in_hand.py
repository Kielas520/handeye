#!/usr/bin/env python3
"""
Eye-in-Hand 手眼标定完整 Demo
===============================

场景: 相机安装在机械臂末端法兰上 (眼在手上), 标定板固定在地面上。

运行:
    uv run python scripts/demo_eye_in_hand.py

流程概览:
    1. 生成合成标定数据 (含已知真值, 用于验证)
    2. 用 Tsai / Navy / Park 三种方法分别求解 X = T_ee^cam
    3. 与真值对比, 计算旋转误差和平移误差
    4. 输出对比表格
    5. 可视化误差分布 (保存图片到 outputs/)
"""

import sys
import numpy as np
from pathlib import Path

# 将项目根目录加入路径, 确保可以从任意位置导入 handeye 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from handeye import (
    CalibrationData,
    generate_eye_in_hand_data,
    solve_ax_xb_tsai,
    solve_ax_xb_navy,
    solve_ax_xb_park,
    solve_hand_eye,
    plot_calibration_error,
)


def compute_error(X_est: np.ndarray, X_gt: np.ndarray) -> tuple[float, float]:
    """
    计算标定结果与真值之间的误差

    旋转误差: ||log(R_est^{-1} @ R_gt)||  (弧度), 转为度
    平移误差: ||t_est - t_gt||  (欧氏距离)

    返回: (rotation_error_deg, translation_error)
    """
    # 旋转误差: 从 R_err = R_est^T @ R_gt 提取旋转角
    R_err = X_est[:3, :3].T @ X_gt[:3, :3]
    cos_theta = (np.trace(R_err) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    rot_err_rad = np.arccos(cos_theta)
    rot_err_deg = np.rad2deg(rot_err_rad)

    # 平移误差
    trans_err = np.linalg.norm(X_est[:3, 3] - X_gt[:3, 3])

    return rot_err_deg, trans_err


def main():
    print("=" * 60)
    print("  Eye-in-Hand 手眼标定 Demo")
    print("  相机在机械臂上 (眼在手上), 标定板固定")
    print("=" * 60)

    # ============================================================
    #  第一步: 生成合成数据
    # ============================================================
    print("\n[1/4] 生成合成标定数据 ...")
    n_poses = 12  # 机器人姿态数量 (越多精度越好)
    noise_rot = 0.15  # 相机检测旋转噪声 (度)
    noise_trans = 0.003  # 相机检测平移噪声 (米)

    data: CalibrationData = generate_eye_in_hand_data(
        n_poses=n_poses,
        noise_rotation_deg=noise_rot,
        noise_translation=noise_trans,
        seed=42,
    )
    print(f"  生成 {len(data.A_list)} 组相对运动对 (来自 {n_poses} 个姿态)")
    print(f"  相机观测噪声: 旋转 ±{noise_rot}°, 平移 ±{noise_trans}m")
    print(
        f"  真值 X_gt (T_ee^cam):\n{np.array2string(data.X_gt, precision=4, suppress_small=True)}"
    )

    # ============================================================
    #  第二步: 用三种方法分别标定
    # ============================================================
    print("\n[2/4] 执行标定 (三种方法) ...")

    A = list(data.A_list)
    B = list(data.B_list)

    # Tsai-Lenz 方法
    X_tsai = solve_ax_xb_tsai(A, B)
    r_tsai, t_tsai = compute_error(X_tsai, data.X_gt)

    # Navy 方法 (对偶四元数)
    X_navy = solve_ax_xb_navy(A, B)
    r_navy, t_navy = compute_error(X_navy, data.X_gt)

    # Park 方法 (李代数)
    X_park = solve_ax_xb_park(A, B)
    r_park, t_park = compute_error(X_park, data.X_gt)

    # ============================================================
    #  第三步: 结果对比
    # ============================================================
    print("\n[3/4] 标定结果对比")
    print("-" * 55)
    print(f"{'方法':<12} {'旋转误差(°)':<15} {'平移误差(m)':<15}")
    print("-" * 55)
    print(f"{'Tsai-Lenz':<12} {r_tsai:<15.6f} {t_tsai:<15.6f}")
    print(f"{'Navy':<12} {r_navy:<15.6f} {t_navy:<15.6f}")
    print(f"{'Park':<12} {r_park:<15.6f} {t_park:<15.6f}")
    print("-" * 55)

    # 标注最优
    errors = {
        "Tsai-Lenz": r_tsai,
        "Navy": r_navy,
        "Park": r_park,
    }
    best = min(errors, key=errors.get)
    print(f"  最优方法: {best} (旋转误差最小)")

    # ============================================================
    #  第四步: 可视化
    # ============================================================
    print("\n[4/4] 生成可视化 ...")
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # 使用统一接口也可标定 (验证 solve_hand_eye 封装)
    X_unified = solve_hand_eye(A, B, method="tsai")
    plot_calibration_error(
        A_list=data.A_list,
        B_list=data.B_list,
        X_est=X_unified,
        title=f"Eye-in-Hand 标定残差 (Tsai, 旋转误差={r_tsai:.4f}°)",
        save_path=str(output_dir / "eye_in_hand_error.png"),
    )
    print(f"  残差图已保存: {output_dir / 'eye_in_hand_error.png'}")

    print(f"\n{'=' * 60}")
    print("  Demo 完成! 查看 outputs/ 目录获取可视化结果。")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
