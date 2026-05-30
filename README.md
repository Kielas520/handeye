# handeye — 手眼标定

Robot Hand-Eye Calibration demo, 使用 Python 实现三种经典 AX=XB 求解算法。

## 快速开始

```bash
# 安装 uv (如未安装)
pip install uv

# 克隆仓库
git clone <this-repo> && cd handeye

# 同步依赖 & 安装包 (可编辑模式)
uv sync

# 运行 Eye-in-Hand 标定 Demo
uv run python scripts/demo_eye_in_hand.py

# 运行 Eye-to-Hand 标定 Demo
uv run python scripts/demo_eye_to_hand.py
```

## 项目结构

```
handeye/
├── pyproject.toml         # uv 项目配置, Python >= 3.10
├── CALIBRATION_FLOW.md    # 详细实现流程 & 数学推导
├── src/                   # 库函数 (安装后可全局导入)
│   ├── __init__.py        # 统一导出接口
│   ├── core.py            # 标定算法: Tsai, Navy, Park
│   ├── transform.py       # 坐标变换工具
│   ├── data.py            # 合成数据生成
│   ├── io.py              # 数据读写
│   └── visualize.py       # 可视化
├── scripts/               # 可执行演示脚本
│   ├── demo_eye_in_hand.py
│   ├── demo_eye_to_hand.py
│   └── generate_synthetic_data.py
└── outputs/               # 输出文件 (自动创建)
```

## 使用方式

安装后, 可在任意 Python 文件中导入:

```python
from handeye import (
    solve_hand_eye,
    generate_eye_in_hand_data,
    generate_eye_to_hand_data,
    plot_calibration_error,
)

# 生成 Eye-in-Hand 合成数据
data = generate_eye_in_hand_data(n_poses=12, noise_rotation_deg=0.1, seed=42)

# 标定
X = solve_hand_eye(list(data.A_list), list(data.B_list), method="tsai")

# 与真值对比
print("标定结果:\n", X)
print("真值:\n", data.X_gt)
```

## 算法对比

| 方法 | 核心思想 | 特点 |
|------|---------|------|
| **Tsai-Lenz** | 轴角分离法 | 经典、直观, 先解旋转再解平移 |
| **Navy** | 对偶四元数 SVD | 同时求解旋转平移, 数学优雅 |
| **Park** | 李代数线性化 | 旋转平移完全解耦, 数值稳定 |

## 两种标定模式

### Eye-in-Hand (眼在手上)
相机安装在机械臂末端, 标定板固定。求 `X = T_ee^cam`。

### Eye-to-Hand (眼在手外)
相机固定, 标定板在机械臂末端。求 `X = T_cam^base`。

详见 [CALIBRATION_FLOW.md](./CALIBRATION_FLOW.md)。

## 依赖

- Python >= 3.10
- numpy >= 1.24
- scipy >= 1.10
- matplotlib >= 3.7

## 许可

MIT
