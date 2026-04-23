# --- 基础库导入 ---
import os, sys, shutil, subprocess # 系统交互、文件操作、进程调用
import numpy as np                # 数值计算
import nibabel as nib             # 读写 NIfTI 医学图像格式
import vtk                        # 3D 可视化核心库
# --- PyQt6 GUI 库导入 ---
from PyQt6.QtWidgets import (QMainWindow, QPushButton, QFileDialog, QLabel, 
    QHBoxLayout, QVBoxLayout, QWidget, QGroupBox, QMessageBox, QColorDialog, 
    QProgressBar, QApplication, QSlider, QScrollArea, QListWidget, QListWidgetItem, 
    QFormLayout, QCheckBox, QFrame, QAbstractItemView)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings # 核心功能、多线程、信号槽、配置保存
from PyQt6.QtGui import QColor, QIcon, QPixmap, QPainter, QImage # 图形绘制

# --- 项目内部模块导入 ---
from config import (INPUT_FOLDER, OUTPUT_FOLDER, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT, 
    ORGANIZATION_NAME, APPLICATION_NAME, SETTINGS_KEY_LAST_DICOM_PATH, DEFAULT_DATASET_ID, DEFAULT_CONFIGURATION)
from dicom_loader import DicomLoader       # 自定义：DICOM 转 NIfTI 工具
from viewer_3d import Viewer3D             # 自定义：VTK 渲染窗口封装
from nii_to_stl import NiiToStlConverter   # 自定义：NIfTI 转 3D 网格工具

STYLESHEET = """
QMainWindow { background-color: #f5f6f7; }
QWidget { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; font-size: 13px; color: #333; }
QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 8px; margin-top: 12px; padding-top: 10px; background-color: #ffffff; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 8px; color: #2c3e50; }
QPushButton { background-color: #3498db; color: white; border: none; border-radius: 6px; padding: 8px 15px; font-weight: bold; }
QPushButton:hover { background-color: #2980b9; }
QPushButton:pressed { background-color: #21618c; }
QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d; }
QPushButton#btn-success { background-color: #2ecc71; }
QPushButton#btn-success:hover { background-color: #27ae60; }
QLabel { color: #2c3e50; }
QLabel#status-label { color: #7f8c8d; font-style: italic; }
QProgressBar { border: 1px solid #ddd; border-radius: 4px; text-align: center; background: #ecf0f1; height: 18px; }
QProgressBar::chunk { background-color: #3498db; border-radius: 3px; }
QSlider::groove:horizontal { border: 1px solid #ddd; height: 6px; background: #ecf0f1; border-radius: 3px; }
QSlider::handle:horizontal { background: #3498db; border: 1px solid #2980b9; width: 16px; margin: -6px 0; border-radius: 8px; }
QScrollArea { border: none; background: transparent; }
"""

# ================= 自定义列表行控件 =================
class LabelRowWidget(QFrame):
    # 定义两个自定义信号，用于与主窗口通信
    # 信号1：当整行被点击时发射，传递参数 label_id (int)
    row_clicked = pyqtSignal(int)
    # 信号2：当复选框状态改变时发射，传递参数 label_id (int) 和 状态 (bool)
    visibility_toggled = pyqtSignal(int, bool)

    def __init__(self, label_id, is_visible=True, parent=None):
        super().__init__(parent)
        self.label_id = label_id                # 保存当前行的 ID，方便后续查找
        self.setObjectName("label_row")         # 设置对象名称，用于 QSS 样式表定位
        # 设置鼠标悬停时的光标形状为“手型”，提示用户该行可点击
        self.setCursor(Qt.CursorShape.PointingHandCursor) 

        # --- 布局与组件初始化 ---
        # 创建水平布局，将组件从左到右排列
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5) # 设置外边距 (左, 上, 右, 下)
        
        # 1. 标签名称 (例如 "Label 1")
        self.lbl_name = QLabel(f"Label {label_id}")
        # 关键设置：设置属性 WA_TransparentForMouseEvents
        # 作用：让鼠标事件“穿透”标签，直接传递给父控件 (QFrame)
        # 这样点击文字时，也能触发下方的 mousePressEvent，而不是被文字挡住
        self.lbl_name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        
        # 2. 可见性复选框
        self.cb_visible = QCheckBox()
        self.cb_visible.setChecked(is_visible)  # 初始化勾选状态
        
        # 绑定信号：当复选框状态切换 (toggled) 时，发射自定义信号 visibility_toggled
        # 使用 lambda 捕获当前的 label_id 和勾选状态 checked
        self.cb_visible.toggled.connect(lambda checked: self.visibility_toggled.emit(self.label_id, checked))
        
        # 将组件添加到布局中
        layout.addWidget(self.lbl_name)
        layout.addWidget(self.cb_visible)

    # --- 事件重写 ---
    def mousePressEvent(self, event):
        # 核心逻辑：判断点击位置是否在复选框区域内
        # 如果点击位置 不包含在复选框的几何区域内，则视为点击了“行”
        if not self.cb_visible.geometry().contains(event.pos()):
            # 发射行点击信号，通知主窗口“用户选中了这一行”
            self.row_clicked.emit(self.label_id)
        
        # 调用父类的 mousePressEvent 以确保标准的事件处理流程继续
        super().mousePressEvent(event)

    # 设置选中状态（用于高亮显示当前选中的行）
    def set_selected(self, selected):
        if selected:
            # 如果选中，应用高亮样式 (蓝色背景 + 左侧蓝条)
            self.setStyleSheet("#label_row { background: #e3f2fd; border-left: 4px solid #3498db; border-radius: 6px; }")
        else:
            # 如果未选中，恢复透明背景
            self.setStyleSheet("#label_row { background: transparent; border-left: 4px solid transparent; border-radius: 6px; }")


# ================= 推理线程 =================
class InferenceWorker(QThread):
    # 定义完成信号，用于通知主窗口任务结束
    # 参数1: result_path (str) - 推理结果文件的路径
    # 参数2: error_msg (str) - 错误信息（如果为空字符串表示无错误）
    finished_signal = pyqtSignal(str, str)

    def __init__(self, input_nii_path, dataset_id=DEFAULT_DATASET_ID, configuration=DEFAULT_CONFIGURATION):
        super().__init__()
        self.input_nii_path = input_nii_path    # 输入的 NIfTI 文件路径
        self.dataset_id = dataset_id            # nnU-Net 数据集 ID
        self.configuration = configuration      # 配置名称 (如 3d_fullres)

    def run(self):
        # run() 方法是 QThread 的核心，线程启动时会自动执行这里的代码
        try:
            # 1. 准备输入数据
            # 获取文件名 (例如 patient_001.nii.gz)
            filename = os.path.basename(self.input_nii_path)
            # 构建目标路径：将文件复制到项目的 INPUT_FOLDER 中
            temp_input_file = os.path.join(INPUT_FOLDER, filename)
            
            # 如果目标位置不存在该文件，则进行复制
            # 注意：nnU-Net 通常要求输入是一个文件夹，这里简化处理直接复制文件进去
            if not os.path.exists(temp_input_file):
                shutil.copy(self.input_nii_path, temp_input_file)
            
            # 2. 构建并执行 nnU-Net 命令行指令
            # 组装命令列表：nnUNetv2_predict -i [输入文件夹] -o [输出文件夹] ...
            cmd = [
                "nnUNetv2_predict", 
                "-i", INPUT_FOLDER, 
                "-o", OUTPUT_FOLDER, 
                "-d", str(self.dataset_id), 
                "-f", "all", 
                "-c", self.configuration
            ]
            # 使用 subprocess 运行命令
            # capture_output=True: 捕获标准输出和错误输出
            # text=True: 以字符串形式而非字节形式返回输出
            # check=False: 即使返回码非0也不抛出异常，而是由我们手动检查
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            # 3. 检查执行结果
            # returncode != 0 表示命令执行出错
            if result.returncode != 0:
                # 发射信号，返回空路径和错误信息
                self.finished_signal.emit("", f"命令执行失败:\n{result.stderr}")
                return # 结束线程

            # 4. 查找输出文件
            found_file = None
            # 策略 A：尝试根据原文件名查找对应的输出文件
            for f in os.listdir(OUTPUT_FOLDER):
                # 检查是否是 nii 文件，且文件名包含原始文件名的主体部分
                if (f.endswith(".nii.gz") or f.endswith(".nii")) and os.path.splitext(filename)[0] in f:
                    found_file = os.path.join(OUTPUT_FOLDER, f)
                    break
            
            # 策略 B：如果策略 A 没找到，取输出文件夹中最新修改的 nii 文件
            if not found_file:
                nii_files = [os.path.join(OUTPUT_FOLDER, f) for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.nii.gz') or f.endswith('.nii')]
                if nii_files: 
                    # max(..., key=os.path.getmtime) 获取修改时间最新的文件
                    found_file = max(nii_files, key=os.path.getmtime)

            # 5. 发送最终结果
            if found_file:
                self.finished_signal.emit(found_file, "") # 成功：发送路径，错误信息为空
            else:
                self.finished_signal.emit("", "未找到结果文件。") # 失败：路径为空，发送错误提示
                
        except Exception as e: 
            # 捕获代码执行过程中的任何 Python 异常
            self.finished_signal.emit("", str(e)) 


# ================= 主窗口 =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- 1. 窗口基础设置 ---
        self.setWindowTitle(WINDOW_TITLE)               # 设置窗口标题 (来自 config.py)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)        # 设置窗口初始大小
        self.setStyleSheet(STYLESHEET)                  # 应用全局 CSS 样式表
        self.setWindowIcon(self._create_logo_image())   # 设置窗口图标 (调用下方自定义方法生成蓝色圆圈)
        
        # --- 2. 配置管理 ---
        # QSettings 用于在注册表或 ini 文件中保存用户设置 (如上次打开的路径)
        self.settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)

        # --- 3. 状态变量初始化 ---
        self.input_nii_path = None      # 输入的 NIfTI 文件路径
        self.output_nii_path = None     # AI 推理输出的文件路径
        self.worker = None              # 推理线程对象
        self.label_data_cache = {}      # 缓存：存储每个标签的材质参数 (颜色、透明度等)
        self.current_selected_label = None # 当前在列表中点击选中的标签 ID
        self.label_row_widgets = {}     # 缓存：存储列表中的行控件对象
        self.slider_scales = {}         # 缓存：存储滑块的缩放比例 (因为 QSlider 只支持整数)

        # --- 4. 界面构建与恢复 ---
        self.init_ui()                  # 调用方法构建界面布局
        self._restore_last_path_hint()  # 从配置中读取上次打开的路径并显示提示
        self.statusBar().showMessage("👋 就绪。", 5000) # 在底部状态栏显示欢迎信息，5秒后消失

    def _create_logo_image(self, width=64, height=64, color="#3498db"):
        # 创建一个 ARGB32 格式的空白图像 (支持透明通道)
        image = QImage(width, height, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent) # 背景填充为透明
        
        painter = QPainter(image)              # 创建画家对象
        painter.setRenderHint(QPainter.RenderHint.Antialiasing) # 开启抗锯齿，使边缘平滑
        painter.setBrush(QColor(color))        # 设置画刷颜色 (蓝色)
        painter.setPen(Qt.PenStyle.NoPen)      # 设置画笔为无 (不绘制边框)
        
        # 绘制椭圆 (x, y, w, h)，这里留了 2px 边距
        painter.drawEllipse(2, 2, width-4, height-4)
        painter.end()                          # 结束绘制
        
        # 将 QImage 转换为 QIcon 返回
        return QIcon(QPixmap.fromImage(image))

    def init_ui(self):
        # --- 主布局：水平布局 (左控制栏 | 右3D视图) ---
        central_widget = QWidget()             # 创建中心控件
        self.setCentralWidget(central_widget)  # 设置为主窗口的中心控件
        main_layout = QHBoxLayout(central_widget) # 创建水平布局管理器
        main_layout.setSpacing(15)             # 设置控件间距
        main_layout.setContentsMargins(15, 15, 15, 15) # 设置外边距

        # --- 左侧：滚动区域 (Scroll Area) ---
        # 使用滚动区域是为了当控制面板内容过多时，可以滚动查看，而不拉伸窗口
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)   # 内容随区域大小自动调整
        # 禁止水平滚动条，强制内容换行或适应宽度
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # 滚动区域的内容容器
        scroll_content = QWidget()
        # 内容容器使用垂直布局，将各个 GroupBox 从上到下排列
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setSpacing(15)
        
        # 面板1：数据与推理
        # 1. 创建 GroupBox 容器
        file_group = self._create_group_box("📂 数据与推理")
        file_layout = QVBoxLayout() # 内部使用垂直布局

        # --- 导入按钮 ---
        self.btn_import = QPushButton("📂 导入 DICOM 序列")
        self.btn_import.clicked.connect(self.load_dicom)  # 绑定点击事件
        self.btn_import.setMinimumHeight(40)              # 设置最小高度，方便点击
        file_layout.addWidget(self.btn_import)
        
        # --- 路径提示标签 ---
        self.lbl_path_hint = QLabel()
        self.lbl_path_hint.setObjectName("path-hint")     # 设置对象名以便在 CSS 中定制样式
        file_layout.addWidget(self.lbl_path_hint)

        # --- 文件状态标签 ---
        self.lbl_file = QLabel("❌ 未选择文件")
        self.lbl_file.setWordWrap(True)                   # 允许文字自动换行
        # 设置初始样式：红色文字，淡红背景，圆角
        self.lbl_file.setStyleSheet("color: #e74c3c; padding: 5px; background: #fdf2f2; border-radius: 4px;")
        file_layout.addWidget(self.lbl_file)

        # --- 推理按钮 ---
        self.btn_infer = QPushButton("🚀 开始 AI 推理")
        self.btn_infer.clicked.connect(self.run_inference)
        self.btn_infer.setObjectName("btn-success")      # 使用 CSS 中定义的成功按钮样式 (绿色)
        self.btn_infer.setEnabled(False)                 # 初始禁用，导入文件后才启用
        self.btn_infer.setMinimumHeight(40)
        file_layout.addWidget(self.btn_infer)
        
        # --- 进度条 ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)              # 初始隐藏，推理开始时显示
        self.progress_bar.setRange(0, 0)                 # 设置为 0-0，显示为“忙碌”动画而非具体进度
        file_layout.addWidget(self.progress_bar)

        # 应用布局并添加到主滚动布局
        file_group.setLayout(file_layout)
        self.scroll_layout.addWidget(file_group)

        # 面板 2：3D 显示控制
        # 2. 3D 显示控制面板
        view_group = self._create_group_box("👁️ 3D 可视化")
        view_layout = QVBoxLayout()

        # --- 生成 3D 按钮 ---
        self.btn_show_3d = QPushButton("🧊 生成并显示 3D 模型")
        self.btn_show_3d.clicked.connect(self.show_3d_model)
        self.btn_show_3d.setEnabled(True)
        self.btn_show_3d.setMinimumHeight(40)
        view_layout.addWidget(self.btn_show_3d)

        # --- 状态标签 ---
        self.lbl_status_3d = QLabel("⏳ 状态：等待推理结果")
        self.lbl_status_3d.setObjectName("status-label")      # 对应 CSS 中的灰色斜体样式
        view_layout.addWidget(self.lbl_status_3d)
        view_group.setLayout(view_layout)
        self.scroll_layout.addWidget(view_group)

        # 3. 面板 3：标签列表 
        # 3. 标签列表面板
        self.layer_group = self._create_group_box("📑 结构列表 (点击选中 / 勾选显示)")
        layer_layout = QVBoxLayout()

        self.label_list = QListWidget()
        self.label_list.setMinimumHeight(180)     # 设置最小高度
        self.label_list.setSpacing(1)             # 项之间的间距
        self.label_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # 关键设置：禁用默认的选中高亮，因为我们用自定义控件的高亮样式
        self.label_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)

        layer_layout.addWidget(self.label_list)
        self.layer_group.setLayout(layer_layout)
        self.scroll_layout.addWidget(self.layer_group)

        # 面板 4：材质属性编辑器
        # 4. 材质属性编辑器
        self.mat_group = self._create_group_box("🎨 材质属性编辑器")
        mat_layout = QFormLayout()        # 表单布局：左侧标签，右侧控件
        mat_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        
        # --- 目标提示 ---
        self.lbl_mat_target = QLabel("请在上方列表中点击一个标签")
        self.lbl_mat_target.setStyleSheet("color: #7f8c8d; font-style: italic; margin-bottom: 10px;")
        mat_layout.addRow(self.lbl_mat_target)      # 占满一行

        # --- 颜色选择按钮 ---
        self.btn_color = QPushButton("选择基础颜色")
        self.btn_color.setEnabled(False)      # 初始禁用
        self.btn_color.clicked.connect(self.pick_color)
        mat_layout.addRow(self.btn_color)

        # --- 滑块组 ---
        self.sliders = {}
        # 定义属性配置：(显示名, 键名, 最小值, 最大值, 步长, 默认值)
        props = [
            ("透明度", "opacity", 0.0, 1.0, 0.01, 1.0),
            ("环境光", "ambient", 0.0, 1.0, 0.01, 0.4),
            ("漫反射", "diffuse", 0.0, 1.0, 0.01, 0.6),
            ("高光强度", "specular", 0.0, 1.0, 0.01, 0.2),
            ("光泽度", "specular_power", 1.0, 100.0, 1.0, 20.0),
        ]

        for label_text, key, min_v, max_v, step, default_v in props:
            slider = QSlider(Qt.Orientation.Horizontal) # 创建水平滑块
            slider.setEnabled(False)           # 初始禁用
            
            # QSlider 只能处理整数，所以需要缩放 (例如 0.0-1.0 映射到 0-100)
            scale = 100 if max_v <= 1.0 else 1
            self.slider_scales[key] = scale
            
            slider.setMinimum(int(min_v * scale))
            slider.setMaximum(int(max_v * scale))
            slider.setValue(int(default_v * scale))
            # 绑定值改变信号，使用 lambda 传递键名 key 和当前值 v
            slider.valueChanged.connect(lambda v, k=key: self.on_slider_changed(k, v))
            
            mat_layout.addRow(label_text, slider) # 添加到表单
            self.sliders[key] = slider            # 缓存滑块对象

        self.mat_group.setLayout(mat_layout)
        self.scroll_layout.addWidget(self.mat_group)

        # 面板 5：导出结果
        # 5. 导出面板
        export_group = self._create_group_box("💾 导出结果")
        export_layout = QVBoxLayout()
        
        # --- 导出 STL ---
        self.btn_export_stl = QPushButton("📐 导出 STL (合并可见)")
        # 使用 lambda 传递参数 "stl" 给通用的 export_file 方法
        self.btn_export_stl.clicked.connect(lambda: self.export_file("stl"))
        self.btn_export_stl.setEnabled(False)
        export_layout.addWidget(self.btn_export_stl)

        # --- 导出 OBJ ---
        self.btn_export_obj = QPushButton("🎬 导出 OBJ (合并可见)")
        self.btn_export_obj.clicked.connect(lambda: self.export_file("obj"))
        self.btn_export_obj.setEnabled(False)
        export_layout.addWidget(self.btn_export_obj)

        # --- 导出 NIfTI (新增功能) ---
        self.btn_export_nii = QPushButton("🧠 导出 NIfTI (合并可见)")
        self.btn_export_nii.clicked.connect(self.export_nii_file)
        self.btn_export_nii.setEnabled(False)
        export_layout.addWidget(self.btn_export_nii)

        export_group.setLayout(export_layout)
        self.scroll_layout.addWidget(export_group)

        self.scroll_layout.addStretch() # 在底部添加伸缩因子，将所有控件推向顶部
        
        # 将内容容器放入滚动区域
        scroll_area.setWidget(scroll_content)

        # --- 右侧：3D 视图 ---
        self.viewer = Viewer3D()          # 实例化自定义的 VTK 窗口类 (来自 viewer_3d.py)
        self.viewer.setMinimumWidth(600)  # 设置最小宽度，保证 3D 视图有足够的空间
        
        # 将左侧滚动区和右侧 3D 视图添加到主水平布局中
        # 参数 1 和 3 是拉伸因子，意味着 3D 视图的宽度将是左侧控制栏的 3 倍
        main_layout.addWidget(scroll_area, 1) 
        main_layout.addWidget(self.viewer, 3)

    def _create_group_box(self, title):
        gb = QGroupBox(title)
        gb.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #e0e0e0; border-radius: 8px; margin-top: 12px; padding-top: 10px; background-color: #ffffff; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 10px; color: #2c3e50; background-color: #ffffff; }
        """)
        return gb

    def _restore_last_path_hint(self):
        last_path = self.settings.value(SETTINGS_KEY_LAST_DICOM_PATH, "")
        if last_path and os.path.exists(last_path):
            self.lbl_path_hint.setText(f"💡 上次位置：.../{os.path.basename(last_path)}")
        else:
            self.lbl_path_hint.setText("💡 暂无历史记录")

    def _save_current_path(self, path):
        self.settings.setValue(SETTINGS_KEY_LAST_DICOM_PATH, path)
        self.lbl_path_hint.setText(f"💡 上次位置：.../{os.path.basename(path)}")

    # ================= 交互逻辑 =================
    def on_row_clicked(self, label_id):
        if label_id not in self.label_data_cache: return
        
        if self.current_selected_label in self.label_row_widgets:
            self.label_row_widgets[self.current_selected_label].set_selected(False)
            
        self.current_selected_label = label_id
        self.label_row_widgets[label_id].set_selected(True)
        
        self.lbl_mat_target.setText(f"正在编辑：Label {label_id}")
        self.lbl_mat_target.setStyleSheet("color: #2c3e50; font-weight: bold;")
        self.btn_color.setEnabled(True)
        for slider in self.sliders.values(): 
            slider.setEnabled(True)
            
        params = self.label_data_cache[label_id]["params"]
        for key, slider in self.sliders.items():
            val = params.get(key, 0.0)
            scale = self.slider_scales[key]
            slider.blockSignals(True)
            slider.setValue(int(val * scale))
            slider.blockSignals(False)

    def on_label_visibility_toggled(self, label_id, is_visible):
        self.viewer.toggle_label_visibility(label_id, is_visible)

    def on_slider_changed(self, key, value):
        if self.current_selected_label is None: return
        scale = self.slider_scales[key]
        real_value = value / scale
        
        self.label_data_cache[self.current_selected_label]["params"][key] = real_value
        self.viewer.update_actor_property(self.current_selected_label, key, real_value)

    def pick_color(self):
        if self.current_selected_label is None: return
        current_color = self.label_data_cache[self.current_selected_label]["params"].get("color", (1,1,1))
        qcolor = QColor(int(current_color[0]*255), int(current_color[1]*255), int(current_color[2]*255))
        color = QColorDialog.getColor(qcolor, self, "选择颜色")
        if color.isValid():
            rgb = (color.redF(), color.greenF(), color.blueF())
            self.label_data_cache[self.current_selected_label]["params"]["color"] = rgb
            self.viewer.update_actor_property(self.current_selected_label, "color", rgb)

    # ================= 业务逻辑 =================
    def load_dicom(self):
        start_dir = self.settings.value(SETTINGS_KEY_LAST_DICOM_PATH, os.path.expanduser("~"))
        if not os.path.isdir(start_dir): start_dir = os.path.expanduser("~")
        folder_path = QFileDialog.getExistingDirectory(self, "选择 DICOM 序列文件夹", start_dir)
        if not folder_path: return
        self._save_current_path(folder_path)

        try:
            self.lbl_file.setText("⏳ 正在加载 DICOM...")
            self.lbl_file.setStyleSheet("color: #f39c12; background: #fef9e7;")
            self.repaint()
            nii_img, nii_path, sitk_img = DicomLoader.load_series(folder_path)
            if not nii_path or not os.path.exists(nii_path):
                raise Exception("DicomLoader 未能生成有效的 NIfTI 文件。")
            self.input_nii_path = nii_path
            self.lbl_file.setText(f"✅ 已加载：{os.path.basename(folder_path)}")
            self.lbl_file.setStyleSheet("color: #27ae60; background: #eafaf1;")
            self._reset_state()
            self.btn_infer.setEnabled(True)
            self.statusBar().showMessage("✅ DICOM 加载成功。")
        except Exception as e:
            QMessageBox.critical(self, "❌ 加载失败", f"读取 DICOM 时发生错误:\n{str(e)}")
            self.lbl_file.setText("❌ 加载失败")
            self.lbl_file.setStyleSheet("color: #e74c3c; background: #fdf2f2;")

    def _reset_state(self):
        self.output_nii_path = None
        self.btn_show_3d.setEnabled(False)
        self.lbl_status_3d.setText("⏳ 状态：等待推理")
        self.viewer.clear_scene()
        self.label_list.clear()
        self.label_data_cache = {}
        self.label_row_widgets = {}
        self.current_selected_label = None
        self.lbl_mat_target.setText("请在上方列表中点击一个标签")
        self.lbl_mat_target.setStyleSheet("color: #7f8c8d; font-style: italic;")
        self.btn_color.setEnabled(False)
        for slider in self.sliders.values():
            slider.setEnabled(False)
            slider.setValue(slider.minimum())
        self.btn_export_stl.setEnabled(False)
        self.btn_export_obj.setEnabled(False)
        self.btn_export_nii.setEnabled(False)  # 【新增】重置状态

    def run_inference(self):
        if not self.input_nii_path or not os.path.exists(self.input_nii_path):
            QMessageBox.warning(self, "⚠️ 警告", "请先导入有效的 DICOM 序列。")
            return
        self.btn_infer.setEnabled(False)
        self.btn_import.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.lbl_status_3d.setText("🚀 正在运行 AI 推理... 请勿关闭窗口")
        self.lbl_status_3d.setStyleSheet("color: #2980b9; font-weight: bold;")
        self.statusBar().showMessage("⏳ 正在执行 nnU-Net 推理...")

        self.worker = InferenceWorker(self.input_nii_path, dataset_id=DEFAULT_DATASET_ID, configuration=DEFAULT_CONFIGURATION)
        self.worker.finished_signal.connect(self.on_inference_finished)
        self.worker.start()

    def on_inference_finished(self, result_path, error_msg):
        self.progress_bar.setVisible(False)
        self.btn_import.setEnabled(True)
        if error_msg:
            self.btn_infer.setEnabled(True)
            self.lbl_status_3d.setText("❌ 推理失败")
            self.lbl_status_3d.setStyleSheet("color: #e74c3c;")
            QMessageBox.critical(self, "推理错误", error_msg)
        else:
            self.output_nii_path = result_path
            self.btn_infer.setEnabled(True)
            self.lbl_status_3d.setText("✅ 推理完成，准备生成 3D 视图")
            self.lbl_status_3d.setStyleSheet("color: #27ae60; font-weight: bold;")
            self.btn_show_3d.setEnabled(True)
            self.statusBar().showMessage("✅ 推理成功，请点击“生成并显示 3D 模型”", 5000)

    def show_3d_model(self):
        if not self.output_nii_path:
            self.output_nii_path = r"D:\Work\FAS-Model\function\Bone3DViewer111\data\temp_output\Ribs_1111.nii.gz"
        if not self.output_nii_path or not os.path.exists(self.output_nii_path):
            QMessageBox.warning(self, "⚠️ 警告", "没有可用的推理结果文件。")
            return

        self.btn_show_3d.setEnabled(False)
        self.lbl_status_3d.setText("🏗️ 正在构建 3D 网格...")
        self.statusBar().showMessage("⏳ 正在读取 NIfTI 并提取表面...")
        self.repaint()

        try:
            results = NiiToStlConverter.load_and_convert_multi(self.output_nii_path)
            if not results:
                raise ValueError("未能从 NIfTI 提取到有效网格。")

            self.viewer.clear_scene()
            self.label_list.clear()
            self.label_data_cache = {}
            self.label_row_widgets = {}

            colors = [(1.0, 0.7, 0.7), (0.7, 1.0, 0.7), (0.7, 0.7, 1.0), (1.0, 1.0, 0.7), (0.7, 1.0, 1.0)]

            for idx, (lid, p_data) in enumerate(results.items()):
                color = colors[idx % len(colors)]
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(p_data)
                mapper.ScalarVisibilityOff()
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                prop = actor.GetProperty()
                prop.SetColor(*color)
                prop.SetAmbient(0.4); prop.SetDiffuse(0.6); prop.SetSpecular(0.2)
                self.viewer.renderer.AddActor(actor)
                self.viewer.actors[lid] = actor
                self.viewer.mappers[lid] = mapper
                self.viewer.poly_data_map[lid] = p_data

                row_widget = LabelRowWidget(lid, is_visible=True)
                row_widget.row_clicked.connect(self.on_row_clicked)
                row_widget.visibility_toggled.connect(self.on_label_visibility_toggled)
                
                item = QListWidgetItem()
                item.setSizeHint(row_widget.sizeHint())
                self.label_list.addItem(item)
                self.label_list.setItemWidget(item, row_widget)
                self.label_row_widgets[lid] = row_widget

                self.label_data_cache[lid] = {
                    "poly_data": p_data,
                    "params": {"color": color, "opacity": 1.0, "ambient": 0.4, "diffuse": 0.6, "specular": 0.2, "specular_power": 20}
                }

            self.viewer.renderer.ResetCamera()
            self.viewer.vtk_widget.GetRenderWindow().Render()
            self.viewer.vtk_widget.update()

            self.lbl_status_3d.setText(f"✅ 显示成功 (共 {len(self.label_data_cache)} 个结构)")
            self.lbl_status_3d.setStyleSheet("color: #27ae60;")
            self.btn_export_stl.setEnabled(True)
            self.btn_export_obj.setEnabled(True)
            self.btn_export_nii.setEnabled(True)  # 【新增】启用 NIfTI 导出
            self.statusBar().showMessage("✅ 3D 模型构建完成")
            
            if self.label_row_widgets:
                first_id = list(self.label_row_widgets.keys())[0]
                self.on_row_clicked(first_id)

        except Exception as e:
            QMessageBox.critical(self, "❌ 3D 生成失败", f"处理 NIfTI 文件时出错:\n{str(e)}")
            self.lbl_status_3d.setText("❌ 3D 生成失败")
            self.lbl_status_3d.setStyleSheet("color: #e74c3c;")
        finally:
            self.btn_show_3d.setEnabled(True)

    # 【新增】导出 NIfTI 逻辑
    def export_nii_file(self):
        if not self.output_nii_path or not os.path.exists(self.output_nii_path):
            QMessageBox.warning(self, "⚠️ 警告", "没有可用的原始 NIfTI 文件作为空间参考。")
            return

        visible_labels = [lid for lid, row_w in self.label_row_widgets.items() if row_w.cb_visible.isChecked()]
        if not visible_labels:
            QMessageBox.warning(self, "⚠️ 警告", "没有勾选任何可见标签，无法导出。")
            return

        try:
            default_name = "segmentation_visible.nii.gz"
            save_path, _ = QFileDialog.getSaveFileName(
                self, "导出 NIfTI 文件",
                os.path.join(os.path.dirname(self.output_nii_path), default_name),
                "NIfTI Files (*.nii.gz *.nii)"
            )
            if not save_path: return

            self.statusBar().showMessage("⏳ 正在合并并导出 NIfTI...")
            self.repaint()

            # 读取原始数据与空间信息
            nii_img = nib.load(self.output_nii_path)
            original_data = nii_img.get_fdata()
            affine = nii_img.affine
            header = nii_img.header.copy()

            # 构建新掩膜（保留原始标签值，未选中的标签自动清零）
            combined_data = np.zeros_like(original_data, dtype=original_data.dtype)
            for lid in visible_labels:
                combined_data[original_data == lid] = lid

            # 保存
            new_nii = nib.Nifti1Image(combined_data, affine, header=header)
            nib.save(new_nii, save_path)

            QMessageBox.information(self, "✅ 导出成功", f"文件已保存至:\n{save_path}")
            self.statusBar().showMessage(f"✅ 导出成功: {save_path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "❌ 导出失败", f"导出 NIfTI 时发生错误:\n{str(e)}")
            self.statusBar().showMessage("❌ 导出失败", 5000)

    def export_file(self, file_type):
        if not self.viewer.actors: return
        try:
            file_filter = "STL Files (*.stl)" if file_type == "stl" else "OBJ Files (*.obj)"
            default_name = f"segmentation_visible.{file_type}"
            save_path, _ = QFileDialog.getSaveFileName(self, f"导出 {file_type.upper()} 文件", os.path.expanduser(f"~/Desktop/{default_name}"), file_filter)
            if not save_path: return

            self.statusBar().showMessage(f"⏳ 正在合并并导出 {file_type.upper()}...")
            self.repaint()

            append_filter = vtk.vtkAppendPolyData()
            has_data = False
            for lid, row_w in self.label_row_widgets.items():
                if row_w.cb_visible.isChecked() and lid in self.viewer.poly_data_map:
                    append_filter.AddInputData(self.viewer.poly_data_map[lid])
                    has_data = True

            if not has_data:
                QMessageBox.warning(self, "⚠️ 警告", "没有勾选任何可见标签，无法导出。")
                return

            append_filter.Update()
            NiiToStlConverter.export_polydata_to_file(append_filter.GetOutput(), save_path)
            
            QMessageBox.information(self, "✅ 导出成功", f"文件已保存至:\n{save_path}")
            self.statusBar().showMessage(f"✅ 导出成功: {save_path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "❌ 导出失败", f"导出文件时发生错误:\n{str(e)}")
            self.statusBar().showMessage("❌ 导出失败", 5000)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, '确认退出', "推理正在进行中，确定要退出吗?", 
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.terminate()
                event.accept()
            else: event.ignore()
        else: event.accept()

if __name__ == "__main__":
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())