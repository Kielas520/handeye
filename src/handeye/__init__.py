"""
handeye — 手眼标定工具库

从任意位置导入:
    from handeye import solve_hand_eye, generate_calibration_data
"""

from handeye.core import (
    solve_ax_xb_tsai,
    solve_ax_xb_navy,
    solve_ax_xb_park,
    solve_hand_eye,
)
from handeye.data import (
    CalibrationData,
    generate_eye_in_hand_data,
    generate_eye_to_hand_data,
)
from handeye.transform import (
    rotation_matrix_to_axis_angle,
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    inverse_homogeneous,
    random_rotation_matrix,
    random_transform,
    compose_transforms,
)
from handeye.io import (
    save_calibration_data,
    load_calibration_data,
    save_result,
)
from handeye.visualize import (
    plot_calibration_error,
    plot_coordinate_frames,
)

__all__ = [
    # core
    "solve_ax_xb_tsai",
    "solve_ax_xb_navy",
    "solve_ax_xb_park",
    "solve_hand_eye",
    # data
    "CalibrationData",
    "generate_eye_in_hand_data",
    "generate_eye_to_hand_data",
    # transform
    "rotation_matrix_to_axis_angle",
    "axis_angle_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "quaternion_to_rotation_matrix",
    "inverse_homogeneous",
    "random_rotation_matrix",
    "random_transform",
    "compose_transforms",
    # io
    "save_calibration_data",
    "load_calibration_data",
    "save_result",
    # visualize
    "plot_calibration_error",
    "plot_coordinate_frames",
]
