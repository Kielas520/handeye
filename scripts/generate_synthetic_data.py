#!/usr/bin/env python3
"""
生成合成标定数据脚本

可以作为独立工具生成测试数据, 保存为 .npz 文件,
供后续标定流程使用。

运行:
    uv run python scripts/generate_synthetic_data.py

输出:
    outputs/data_eye_in_hand.npz
    outputs/data_eye_to_hand.npz
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from handeye import (
    generate_eye_in_hand_data,
    generate_eye_to_hand_data,
    save_calibration_data,
)


def main():
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # ---- Eye-in-Hand ----
    print("[1/2] 生成 Eye-in-Hand 数据 ...")
    data_eih = generate_eye_in_hand_data(
        n_poses=12,
        noise_rotation_deg=0.1,
        noise_translation=0.002,
        seed=123,
    )
    save_calibration_data(
        filepath=output_dir / "data_eye_in_hand.npz",
        A_list=data_eih.A_list,
        B_list=data_eih.B_list,
        config_type="eye_in_hand",
        ground_truth_X=data_eih.X_gt,
    )
    print(f"  保存: {output_dir / 'data_eye_in_hand.npz'}")

    # ---- Eye-to-Hand ----
    print("[2/2] 生成 Eye-to-Hand 数据 ...")
    data_eth = generate_eye_to_hand_data(
        n_poses=15,
        noise_rotation_deg=0.1,
        noise_translation=0.002,
        seed=456,
    )
    save_calibration_data(
        filepath=output_dir / "data_eye_to_hand.npz",
        A_list=data_eth.A_list,
        B_list=data_eth.B_list,
        config_type="eye_to_hand",
        ground_truth_X=data_eth.X_gt,
    )
    print(f"  保存: {output_dir / 'data_eye_to_hand.npz'}")

    print("\n完成! 数据文件已生成在 outputs/ 目录。")


if __name__ == "__main__":
    main()
