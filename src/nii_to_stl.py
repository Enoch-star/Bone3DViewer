import vtk
import numpy as np
from vtk.util import numpy_support
import os

class NiiToStlConverter:
    @staticmethod
    def load_and_convert_multi(nii_path, output_dir=None):
        """读取 NIfTI，提取所有非零标签，分别生成平滑后的 PolyData"""
        try:
            if not os.path.exists(nii_path):
                raise FileNotFoundError(f"文件不存在: {nii_path}")

            reader = vtk.vtkNIFTIImageReader()
            reader.SetFileName(nii_path)
            reader.Update()
            image_data = reader.GetOutput()
            if image_data is None:
                raise ValueError("读取失败：VTK ImageData 为空。")

            scalars = image_data.GetPointData().GetScalars()
            np_array = numpy_support.vtk_to_numpy(scalars)
            unique_labels = np.unique(np_array)
            valid_labels = [int(l) for l in unique_labels if l != 0 and np.isfinite(l)]

            if not valid_labels:
                raise ValueError("文件中没有非零标签。")

            results = {}
            for label in valid_labels:
                temp_array = np.zeros_like(np_array)
                temp_array[np_array == label] = 1
                
                temp_image = vtk.vtkImageData()
                temp_image.CopyStructure(image_data)
                vtk_array = numpy_support.numpy_to_vtk(temp_array.ravel(), deep=True)
                vtk_array.SetName("Scalars")
                temp_image.GetPointData().SetScalars(vtk_array)

                surf = vtk.vtkDiscreteMarchingCubes()
                surf.SetInputData(temp_image)
                surf.SetValue(0, 1)
                surf.Update()
                if surf.GetOutput().GetNumberOfPoints() == 0:
                    continue

                smoother = vtk.vtkWindowedSincPolyDataFilter()
                smoother.SetInputConnection(surf.GetOutputPort())
                smoother.SetNumberOfIterations(20)
                smoother.NonManifoldSmoothingOn()
                smoother.NormalizeCoordinatesOn()
                smoother.Update()

                cleaner = vtk.vtkCleanPolyData()
                cleaner.SetInputConnection(smoother.GetOutputPort())
                cleaner.Update()

                if cleaner.GetOutput().GetNumberOfPoints() > 0:
                    results[label] = smoother.GetOutput()

            return results
        except Exception as e:
            print(f"❌ [转换错误] {type(e).__name__}: {e}")
            return None

    @staticmethod
    def export_polydata_to_file(poly_data, file_path):
        """将 PolyData 导出为 STL 或 OBJ"""
        if not poly_data or poly_data.GetNumberOfPoints() == 0:
            raise ValueError("无效的 PolyData，无法导出。")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".stl":
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(file_path)
            writer.SetInputData(poly_data)
            writer.Write()
        elif ext == ".obj":
            writer = vtk.vtkOBJWriter()
            writer.SetFileName(file_path)
            writer.SetInputData(poly_data)
            writer.Write()
        else:
            raise ValueError(f"不支持的导出格式: {ext}。仅支持 .stl 或 .obj")