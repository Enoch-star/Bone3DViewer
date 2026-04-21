# config.py
import os

# ================= 项目路径配置 =================
# 获取当前脚本所在目录 (假设 config.py 在 src 目录下)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) # 返回到 Bone3DViewer111 目录

# 定义输入输出文件夹
INPUT_FOLDER = os.path.join(PROJECT_ROOT, "data/temp_input")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data/temp_output")

# 确保目录存在
for folder in [INPUT_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ================= AI 模型默认配置 =================
DEFAULT_DATASET_ID = "603"
DEFAULT_CONFIGURATION = "3d_fullres"

# ================= UI 默认配置 =================
WINDOW_TITLE = "AI 医学图像分割与 3D 重建系统"
WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 950

# 默认材质参数
DEFAULT_MATERIAL_PARAMS = {
    "color": (0.9, 0.2, 0.2),
    "ambient": 0.4,
    "diffuse": 0.6,
    "specular": 0.2,
    "specular_power": 20,
    "roughness": 0.5,
    "metallic": 0.0,
    "opacity": 1.0
}

# ================= 持久化配置 (QSettings) =================
# 用于注册表或 .ini 文件的组织名和应用名
ORGANIZATION_NAME = "FAS-Medical"
APPLICATION_NAME = "Bone3DViewer"
SETTINGS_KEY_LAST_DICOM_PATH = "LastDicomPath"