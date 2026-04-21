import SimpleITK as sitk
import numpy as np
import nibabel as nib
import os
from config import INPUT_FOLDER

class DicomLoader:
    @staticmethod
    def load_series(folder_path):
        """
        读取 DICOM 序列并转换为 NIfTI (nnUNet 需要 NIfTI 输入)
        """
        print(f"正在读取 DICOM 序列: {folder_path}")
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(folder_path)
        
        if not series_IDs:
            raise ValueError("未在文件夹中找到有效的 DICOM 序列。")
        
        # 获取第一个序列 (实际项目中可能需要让用户选择)
        series_file_names = reader.GetGDCMSeriesFileNames(folder_path, series_IDs[0])
        reader.SetFileNames(series_file_names)
        
        try:
            image = reader.Execute()
        except Exception as e:
            raise RuntimeError(f"读取 DICOM 失败: {e}")

        # 转换为 NIfTI 供 nnUNet 使用
        temp_nii_path = os.path.join(INPUT_FOLDER, "Ribs_1111_0000.nii.gz")
        sitk.WriteImage(image, temp_nii_path)
        
        # 重新加载为 nibabel 对象以便获取数据数组用于后续处理（如果需要）
        nii_img = nib.load(temp_nii_path)
        return nii_img, temp_nii_path, image # 返回 nibabel对象, 路径, 和原始的sitk图像(保留元数据)