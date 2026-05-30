"""
I/O 工具 — 标定数据的保存与加载

数据格式: 每个 .npz 文件包含:
    - A_list: 形状 (N, 4, 4) 机器人运动序列
    - B_list: 形状 (N, 4, 4) 相机运动序列
    - config_type: "eye_in_hand" 或 "eye_to_hand" (字符串)
"""

import numpy as np
from pathlib import Path


def save_calibration_data(
    filepath: str | Path,
    A_list: np.ndarray,
    B_list: np.ndarray,
    config_type: str,
    ground_truth_X: np.ndarray | None = None,
    metadata: dict | None = None,
) -> None:
    """
    保存标定数据到 .npz 文件

    参数
    ----
    filepath : 保存路径
    A_list : (N, 4, 4) 机器人相对运动
    B_list : (N, 4, 4) 相机相对运动
    config_type : "eye_in_hand" 或 "eye_to_hand"
    ground_truth_X : (可选) 真值变换, 用于验证
    metadata : (可选) 额外元信息字典
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "A_list": A_list,
        "B_list": B_list,
        "config_type": np.array(config_type, dtype=str),
    }
    if ground_truth_X is not None:
        save_dict["ground_truth_X"] = ground_truth_X
    if metadata is not None:
        save_dict["metadata"] = np.array(str(metadata))

    np.savez_compressed(filepath, **save_dict)


def load_calibration_data(filepath: str | Path) -> dict:
    """
    从 .npz 文件加载标定数据

    返回
    ----
    dict 包含 "A_list", "B_list", "config_type",
    以及可选的 "ground_truth_X", "metadata"
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    data = np.load(filepath, allow_pickle=True)
    result = {}
    for key in data.files:
        val = data[key]
        if key == "config_type":
            result[key] = str(val) if val.ndim == 0 else str(val[0])
        elif key == "metadata":
            result[key] = str(val) if val.ndim == 0 else str(val[0])
        else:
            result[key] = val
    data.close()
    return result


def save_result(
    filepath: str | Path,
    X_calibrated: np.ndarray,
    method: str,
    rotation_error_deg: float | None = None,
    translation_error: float | None = None,
) -> None:
    """
    保存标定结果

    参数
    ----
    filepath : 保存路径 (.npy 或 .npz)
    X_calibrated : (4, 4) 标定出的手眼变换
    method : 使用的算法名称 ("tsai", "navy", "park")
    rotation_error_deg : 旋转误差 (度), 如有真值
    translation_error : 平移误差, 如有真值
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.suffix == ".npz":
        np.savez_compressed(
            filepath,
            X=X_calibrated,
            method=np.array(method),
            rotation_error_deg=np.array(rotation_error_deg or 0.0),
            translation_error=np.array(translation_error or 0.0),
        )
    else:
        np.save(filepath.with_suffix(".npy"), X_calibrated)
        print(f"[INFO] 结果已保存: {filepath.with_suffix('.npy')}")
