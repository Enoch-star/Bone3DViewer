import os
import sys
import shutil
import subprocess
import numpy as np
import nibabel as nib
import vtk

from PyQt6.QtWidgets import (QMainWindow, QPushButton, QFileDialog, QLabel, 
                             QHBoxLayout, QVBoxLayout, QWidget, QGroupBox, 
                             QMessageBox, QColorDialog, QProgressBar, QApplication,
                             QSlider, QScrollArea, QListWidget, QListWidgetItem,
                             QFormLayout, QCheckBox, QFrame, QAbstractItemView)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QColor, QIcon, QPixmap, QPainter, QImage

from config import (
    INPUT_FOLDER, OUTPUT_FOLDER, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT,
    ORGANIZATION_NAME, APPLICATION_NAME, SETTINGS_KEY_LAST_DICOM_PATH, DEFAULT_DATASET_ID, DEFAULT_CONFIGURATION
)
from dicom_loader import DicomLoader
from viewer_3d import Viewer3D
from nii_to_stl import NiiToStlConverter

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
    row_clicked = pyqtSignal(int)
    visibility_toggled = pyqtSignal(int, bool)

    def __init__(self, label_id, is_visible=True, parent=None):
        super().__init__(parent)
        self.label_id = label_id
        self.setObjectName("label_row")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(12)
        
        self.lbl_name = QLabel(f"Label {label_id}")
        self.lbl_name.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.lbl_name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        
        self.cb_visible = QCheckBox()
        self.cb_visible.setChecked(is_visible)
        self.cb_visible.setStyleSheet("margin-left: auto; spacing: 5px;")
        self.cb_visible.toggled.connect(lambda checked: self.visibility_toggled.emit(self.label_id, checked))
        
        layout.addWidget(self.lbl_name)
        layout.addWidget(self.cb_visible)
        self.setLayout(layout)

    def mousePressEvent(self, event):
        if not self.cb_visible.geometry().contains(event.pos()):
            self.row_clicked.emit(self.label_id)
        super().mousePressEvent(event)

    def set_selected(self, selected):
        if selected:
            self.setStyleSheet("""
                #label_row { 
                    background: #e3f2fd; 
                    border-left: 4px solid #3498db; 
                    border-radius: 6px; 
                }
            """)
        else:
            self.setStyleSheet("#label_row { background: transparent; border-left: 4px solid transparent; border-radius: 6px; }")


# ================= 推理线程 =================
class InferenceWorker(QThread):
    finished_signal = pyqtSignal(str, str) 
    def __init__(self, input_nii_path, dataset_id=DEFAULT_DATASET_ID, configuration=DEFAULT_CONFIGURATION):
        super().__init__()
        self.input_nii_path = input_nii_path
        self.dataset_id = dataset_id
        self.configuration = configuration

    def run(self):
        try:
            filename = os.path.basename(self.input_nii_path)
            temp_input_file = os.path.join(INPUT_FOLDER, filename)
            if not os.path.exists(temp_input_file):
                shutil.copy(self.input_nii_path, temp_input_file)
            
            cmd = ["nnUNetv2_predict", "-i", INPUT_FOLDER, "-o", OUTPUT_FOLDER, "-d", str(self.dataset_id), "-f", "all", "-c", self.configuration]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode != 0:
                self.finished_signal.emit("", f"命令执行失败:\n{result.stderr}")
                return

            found_file = None
            for f in os.listdir(OUTPUT_FOLDER):
                if (f.endswith(".nii.gz") or f.endswith(".nii")) and os.path.splitext(filename)[0] in f:
                    found_file = os.path.join(OUTPUT_FOLDER, f)
                    break
            if not found_file:
                nii_files = [os.path.join(OUTPUT_FOLDER, f) for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.nii.gz') or f.endswith('.nii')]
                if nii_files: found_file = max(nii_files, key=os.path.getmtime)

            if found_file: self.finished_signal.emit(found_file, "")
            else: self.finished_signal.emit("", "未找到结果文件。")
        except Exception as e: self.finished_signal.emit("", str(e))


# ================= 主窗口 =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(STYLESHEET)
        self.setWindowIcon(self._create_logo_image())
        
        self.settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)
        self.input_nii_path = None
        self.output_nii_path = None 
        self.worker = None
        self.label_data_cache = {}
        self.current_selected_label = None
        self.label_row_widgets = {}
        self.slider_scales = {}

        self.init_ui()
        self._restore_last_path_hint()
        self.statusBar().showMessage("👋 就绪。", 5000)

    def _create_logo_image(self, width=64, height=64, color="#3498db"):
        image = QImage(width, height, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, width-4, height-4)
        painter.end()
        return QIcon(QPixmap.fromImage(image))

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setSpacing(15)
        
        # 1. 文件与推理
        file_group = self._create_group_box("📂 数据与推理")
        file_layout = QVBoxLayout()
        self.btn_import = QPushButton("📂 导入 DICOM 序列")
        self.btn_import.clicked.connect(self.load_dicom)
        self.btn_import.setMinimumHeight(40)
        file_layout.addWidget(self.btn_import)
        
        self.lbl_path_hint = QLabel()
        self.lbl_path_hint.setObjectName("path-hint")
        file_layout.addWidget(self.lbl_path_hint)

        self.lbl_file = QLabel("❌ 未选择文件")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color: #e74c3c; padding: 5px; background: #fdf2f2; border-radius: 4px;")
        file_layout.addWidget(self.lbl_file)

        self.btn_infer = QPushButton("🚀 开始 AI 推理")
        self.btn_infer.clicked.connect(self.run_inference)
        self.btn_infer.setObjectName("btn-success")
        self.btn_infer.setEnabled(False)
        self.btn_infer.setMinimumHeight(40)
        file_layout.addWidget(self.btn_infer)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)
        file_layout.addWidget(self.progress_bar)
        file_group.setLayout(file_layout)
        self.scroll_layout.addWidget(file_group)

        # 2. 3D 显示控制
        view_group = self._create_group_box("👁️ 3D 可视化")
        view_layout = QVBoxLayout()
        self.btn_show_3d = QPushButton("🧊 生成并显示 3D 模型")
        self.btn_show_3d.clicked.connect(self.show_3d_model)
        self.btn_show_3d.setEnabled(True)
        self.btn_show_3d.setMinimumHeight(40)
        view_layout.addWidget(self.btn_show_3d)
        self.lbl_status_3d = QLabel("⏳ 状态：等待推理结果")
        self.lbl_status_3d.setObjectName("status-label")
        view_layout.addWidget(self.lbl_status_3d)
        view_group.setLayout(view_layout)
        self.scroll_layout.addWidget(view_group)

        # 3. 标签列表
        self.layer_group = self._create_group_box("📑 结构列表 (点击选中 / 勾选显示)")
        layer_layout = QVBoxLayout()
        self.label_list = QListWidget()
        self.label_list.setMinimumHeight(180)
        self.label_list.setSpacing(1)
        self.label_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.label_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        layer_layout.addWidget(self.label_list)
        self.layer_group.setLayout(layer_layout)
        self.scroll_layout.addWidget(self.layer_group)

        # 4. 材质属性编辑器
        self.mat_group = self._create_group_box("🎨 材质属性编辑器")
        mat_layout = QFormLayout()
        mat_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        
        self.lbl_mat_target = QLabel("请在上方列表中点击一个标签")
        self.lbl_mat_target.setStyleSheet("color: #7f8c8d; font-style: italic; margin-bottom: 10px;")
        mat_layout.addRow(self.lbl_mat_target)

        self.btn_color = QPushButton("选择基础颜色")
        self.btn_color.setEnabled(False)
        self.btn_color.clicked.connect(self.pick_color)
        mat_layout.addRow(self.btn_color)

        self.sliders = {}
        props = [
            ("透明度", "opacity", 0.0, 1.0, 0.01, 1.0),
            ("环境光", "ambient", 0.0, 1.0, 0.01, 0.4),
            ("漫反射", "diffuse", 0.0, 1.0, 0.01, 0.6),
            ("高光强度", "specular", 0.0, 1.0, 0.01, 0.2),
            ("光泽度", "specular_power", 1.0, 100.0, 1.0, 20.0),
        ]

        for label_text, key, min_v, max_v, step, default_v in props:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setEnabled(False)
            scale = 100 if max_v <= 1.0 else 1
            self.slider_scales[key] = scale
            
            slider.setMinimum(int(min_v * scale))
            slider.setMaximum(int(max_v * scale))
            slider.setValue(int(default_v * scale))
            slider.valueChanged.connect(lambda v, k=key: self.on_slider_changed(k, v))
            mat_layout.addRow(label_text, slider)
            self.sliders[key] = slider

        self.mat_group.setLayout(mat_layout)
        self.scroll_layout.addWidget(self.mat_group)

        # 5. 导出
        export_group = self._create_group_box("💾 导出结果")
        export_layout = QVBoxLayout()
        
        self.btn_export_stl = QPushButton("📐 导出 STL (合并可见)")
        self.btn_export_stl.clicked.connect(lambda: self.export_file("stl"))
        self.btn_export_stl.setEnabled(False)
        export_layout.addWidget(self.btn_export_stl)
        
        self.btn_export_obj = QPushButton("🎬 导出 OBJ (合并可见)")
        self.btn_export_obj.clicked.connect(lambda: self.export_file("obj"))
        self.btn_export_obj.setEnabled(False)
        export_layout.addWidget(self.btn_export_obj)

        # 【新增】导出 NIfTI 按钮
        self.btn_export_nii = QPushButton("🧠 导出 NIfTI (合并可见)")
        self.btn_export_nii.clicked.connect(self.export_nii_file)
        self.btn_export_nii.setEnabled(False)
        export_layout.addWidget(self.btn_export_nii)
        
        export_group.setLayout(export_layout)
        self.scroll_layout.addWidget(export_group)

        self.scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)

        # 右侧 3D 视图
        self.viewer = Viewer3D()
        self.viewer.setMinimumWidth(600)
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