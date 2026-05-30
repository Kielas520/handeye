"""
标定数据生成模块 — 合成数据的生成与标定数据结构的定义

支持两种经典配置:
    1. Eye-in-Hand (眼在手上): 相机安装在机械臂末端, 标定板固定
    2. Eye-to-Hand (眼在手外): 相机固定, 标定板安装在机械臂末端

生成的合成数据包含已知的真值 X_gt, 用于验证标定算法的精度。
"""

import numpy as np
from dataclasses import dataclass
from handeye.transform import (
    random_transform,
    inverse_homogeneous,
    compose_transforms,
    random_rotation_matrix,
    axis_angle_to_rotation_matrix,
)


@dataclass
class CalibrationData:
    """
    标定数据容器

    属性
    ----
    A_list : 机器人相对运动 (N, 4, 4)
    B_list : 相机相对运动 (N, 4, 4)
    config_type : "eye_in_hand" 或 "eye_to_hand"
    X_gt : 真值手眼变换 (4, 4), 仅合成数据时有值
    robot_poses : 原始机器人姿态列表 (M, 4, 4)
    camera_observations : 原始相机观测列表 (M, 4, 4)
    """

    A_list: np.ndarray
    B_list: np.ndarray
    config_type: str
    X_gt: np.ndarray | None = None
    robot_poses: np.ndarray | None = None
    camera_observations: np.ndarray | None = None

    def __post_init__(self):
        assert self.A_list.shape[1:] == (4, 4), (
            f"A_list shape error: {self.A_list.shape}"
        )
        assert self.B_list.shape[1:] == (4, 4), (
            f"B_list shape error: {self.B_list.shape}"
        )
        assert len(self.A_list) == len(self.B_list), "A_list 和 B_list 长度必须一致"
        assert self.config_type in ("eye_in_hand", "eye_to_hand"), (
            f"config_type 必须为 'eye_in_hand' 或 'eye_to_hand', 收到: {self.config_type}"
        )

    def __repr__(self) -> str:
        return (
            f"CalibrationData(\n"
            f"  config_type={self.config_type!r},\n"
            f"  n_pairs={len(self.A_list)},\n"
            f"  has_ground_truth={self.X_gt is not None}\n"
            f")"
        )


# =============================================================================
#  Eye-in-Hand 合成数据生成
# =============================================================================


def generate_eye_in_hand_data(
    n_poses: int = 10,
    noise_rotation_deg: float = 0.1,
    noise_translation: float = 0.002,
    seed: int | None = 42,
) -> CalibrationData:
    """
    生成 Eye-in-Hand (眼在手上) 合成标定数据

    物理模型:
        相机固连在机械臂末端法兰上, 标定板固定在地面上。
        关系式: T_base^target = T_base^ee_i @ X @ T_cam_i^target (对任意 i 不变)
        其中 X = T_ee^cam (相机在末端坐标系下的位姿, 待求)

    数据生成流程:
    -----
    1. 生成随机真值 X_gt = T_ee^cam (恒定的手眼变换)
    2. 生成随机标定板位姿 T_base^target (固定不变)
    3. 生成 n_poses 个不同的机器人末端位姿 T_base^ee_i
    4. 反算相机观测 T_cam_i^target = X_gt^{-1} @ (T_base^ee_i)^{-1} @ T_base^target
    5. 对相机观测添加噪声
    6. 构造相对运动对 (A_i, B_i):
         A_i = (T_base^ee_{i+1})^{-1} @ T_base^ee_i  (机器人相对运动)
         B_i = T_cam_{i+1}^target @ (T_cam_i^target)^{-1}  (相机观测到的标定板相对运动)

    参数
    ----
    n_poses : 机器人姿态数量 (建议 >= 6, 更多可提高精度)
    noise_rotation_deg : 相机观测的旋转噪声标准差 (度)
    noise_translation : 相机观测的平移噪声标准差 (米)
    seed : 随机种子

    返回
    ----
    CalibrationData 包含 A_list, B_list, X_gt 等
    """
    rng = np.random.RandomState(seed)

    # 1. 生成真值 X_gt (任意位姿)
    X_gt = random_transform(translation_scale=0.3, random_state=rng)

    # 2. 生成固定的标定板在世界坐标系下的位姿
    T_base_target = random_transform(translation_scale=0.8, random_state=rng)
    # 确保标定板在机器人前方合理位置
    T_base_target[:3, 3] = np.array([0.5, 0.2, 0.0]) + 0.2 * rng.randn(3)

    # 3. 生成 n_poses 个不同的机器人姿态
    robot_poses = []  # T_base^ee 列表
    camera_obs = []  # T_cam^target 列表 (无噪声)
    camera_obs_noisy = []  # T_cam^target 列表 (有噪声)

    for i in range(n_poses):
        # 随机机器人末端位姿 (在球面上分布, 确保能看到标定板)
        # 基准位置 + 随机扰动
        base_pos = np.array([0.0, 0.0, 0.5]) + 0.15 * rng.randn(3)
        T_base_ee_i = np.eye(4)
        T_base_ee_i[:3, :3] = random_rotation_matrix(rng)
        T_base_ee_i[:3, 3] = base_pos
        robot_poses.append(T_base_ee_i)

        # 反算无噪声的相机观测
        # T_base_target = T_base_ee_i @ X_gt @ T_cam_target_i
        # => T_cam_target_i = X_gt^{-1} @ T_base_ee_i^{-1} @ T_base_target
        T_cam_target_i = compose_transforms(
            inverse_homogeneous(X_gt),
            inverse_homogeneous(T_base_ee_i),
            T_base_target,
        )
        camera_obs.append(T_cam_target_i)

        # 添加噪声: 微小旋转扰动 + 平移扰动
        noise_R = (
            axis_angle_to_rotation_matrix(
                rng.randn(3),
                np.deg2rad(noise_rotation_deg),
            )
            if noise_rotation_deg > 0
            else np.eye(3)
        )
        noise_t = noise_translation * rng.randn(3)

        T_noisy = np.eye(4)
        T_noisy[:3, :3] = noise_R
        T_noisy[:3, 3] = noise_t
        camera_obs_noisy.append(compose_transforms(T_cam_target_i, T_noisy))

    robot_poses = np.array(robot_poses)
    camera_obs_noisy = np.array(camera_obs_noisy)

    # 4. 构造相对运动对 (A_i, B_i), i = 0, ..., n_poses-2
    A_list = []
    B_list = []
    for i in range(n_poses - 1):
        # A_i = (T_base^ee_{i+1})^{-1} @ T_base^ee_i  (机器人从 i 到 i+1 的运动)
        A_i = compose_transforms(
            inverse_homogeneous(robot_poses[i + 1]),
            robot_poses[i],
        )
        # B_i = T_cam_{i+1}^target @ (T_cam_i^target)^{-1}  (标定板在相机系下的相对运动)
        B_i = compose_transforms(
            camera_obs_noisy[i + 1],
            inverse_homogeneous(camera_obs_noisy[i]),
        )

        A_list.append(A_i)
        B_list.append(B_i)

    return CalibrationData(
        A_list=np.array(A_list),
        B_list=np.array(B_list),
        config_type="eye_in_hand",
        X_gt=X_gt,
        robot_poses=robot_poses,
        camera_observations=camera_obs_noisy,
    )


# =============================================================================
#  Eye-to-Hand 合成数据生成
# =============================================================================


def generate_eye_to_hand_data(
    n_poses: int = 10,
    noise_rotation_deg: float = 0.1,
    noise_translation: float = 0.002,
    seed: int | None = 42,
) -> CalibrationData:
    """
    生成 Eye-to-Hand (眼在手外) 合成标定数据

    物理模型:
        相机固定在世界坐标系中, 标定板固连在机械臂末端。
        关系式: T_cam^target_i = X @ T_base^ee_i @ T_ee^target (对任意 i 成立)
        其中 X = T_cam^base (相机在世界坐标系下的位姿, 待求)

    数据生成流程:
    -----
    1. 生成随机真值 X_gt = T_cam^base
    2. 生成标定板在末端坐标系下位姿 T_ee^target (固定不变)
    3. 生成 n_poses 个不同的机器人末端位姿 T_base^ee_i
    4. 反算相机观测: T_cam^target_i = X_gt @ T_base^ee_i @ T_ee^target
    5. 添加噪声
     6. 构造相对运动对 (A_i, B_i):
         A_i = T_cam_{i+1}^target @ (T_cam_i^target)^{-1}  (相机观测变化)
         B_i = T_base^ee_{i+1} @ (T_base^ee_i)^{-1}        (机器人相对运动)

    参数
    ----
    n_poses : 机器人姿态数量
    noise_rotation_deg : 旋转噪声标准差 (度)
    noise_translation : 平移噪声标准差 (米)
    seed : 随机种子

    返回
    ----
    CalibrationData 包含 A_list, B_list, X_gt 等
    """
    rng = np.random.RandomState(seed)

    # 1. 真值 X_gt (相机在世界坐标系下)
    X_gt = random_transform(translation_scale=0.8, random_state=rng)
    # 相机通常放置在机器人前方
    X_gt[:3, 3] = np.array([0.8, -0.3, 1.2]) + 0.1 * rng.randn(3)

    # 2. 标定板在末端坐标系下的固定位姿
    T_ee_target = random_transform(translation_scale=0.15, random_state=rng)

    # 3. 生成机器人姿态
    robot_poses = []
    camera_obs_noisy = []

    for i in range(n_poses):
        base_pos = np.array([0.0, 0.0, 0.6]) + 0.2 * rng.randn(3)
        T_base_ee_i = np.eye(4)
        T_base_ee_i[:3, :3] = random_rotation_matrix(rng)
        T_base_ee_i[:3, 3] = base_pos
        robot_poses.append(T_base_ee_i)

        # 无噪声相机观测
        # T_cam^target_i = X_gt @ T_base^ee_i @ T_ee^target
        T_cam_target_i = compose_transforms(
            X_gt,
            T_base_ee_i,
            T_ee_target,
        )

        # 添加噪声
        noise_R = (
            axis_angle_to_rotation_matrix(
                rng.randn(3),
                np.deg2rad(noise_rotation_deg),
            )
            if noise_rotation_deg > 0
            else np.eye(3)
        )
        noise_t = noise_translation * rng.randn(3)

        T_noisy = np.eye(4)
        T_noisy[:3, :3] = noise_R
        T_noisy[:3, 3] = noise_t
        camera_obs_noisy.append(compose_transforms(T_noisy, T_cam_target_i))

    robot_poses = np.array(robot_poses)
    camera_obs_noisy = np.array(camera_obs_noisy)

    # 4. 构造相对运动对
    # Eye-to-hand 的推导:
    #   T_cam_target_i = X @ T_base_ee_i @ T_ee_target  (1)
    #   T_cam_target_j = X @ T_base_ee_j @ T_ee_target  (2)
    # 消去 T_ee_target 后得到: A @ X = X @ B
    #   A = T_cam_target_j @ inv(T_cam_target_i)  (相机观测变化)
    #   B = T_base_ee_j @ inv(T_base_ee_i)        (机器人运动)
    A_list = []
    B_list = []
    for i in range(n_poses - 1):
        A_i = compose_transforms(
            camera_obs_noisy[i + 1],
            inverse_homogeneous(camera_obs_noisy[i]),
        )
        B_i = compose_transforms(
            robot_poses[i + 1],
            inverse_homogeneous(robot_poses[i]),
        )
        A_list.append(A_i)
        B_list.append(B_i)

    return CalibrationData(
        A_list=np.array(A_list),
        B_list=np.array(B_list),
        config_type="eye_to_hand",
        X_gt=X_gt,
        robot_poses=robot_poses,
        camera_observations=camera_obs_noisy,
    )
