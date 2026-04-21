import vtk
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

class Viewer3D(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.layout.addWidget(self.vtk_widget)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1.0, 1.0, 1.0)
        
        self.main_light = vtk.vtkLight()
        self.main_light.SetLightTypeToHeadlight()
        self.main_light.SetIntensity(1.5)
        self.renderer.AddLight(self.main_light)
        
        render_window = self.vtk_widget.GetRenderWindow()
        render_window.AddRenderer(self.renderer)
        
        interactor = render_window.GetInteractor()
        style = vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)
        
        self.camera = self.renderer.GetActiveCamera()
        self.actors = {}
        self.mappers = {}
        self.poly_data_map = {}

        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        self.vtk_widget.setFocus()

    def clear_scene(self):
        self.renderer.RemoveAllViewProps()
        self.actors.clear()
        self.mappers.clear()
        self.poly_data_map.clear()
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.vtk_widget.update()

    def render_multi_polydata(self, poly_data_dict):
        if not poly_data_dict: return
        self.clear_scene()
        colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0), (1.0, 1.0, 0.2), (1.0, 0.2, 1.0)]
        
        for i, (label_id, p_data) in enumerate(poly_data_dict.items()):
            if p_data.GetNumberOfPoints() == 0: continue

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(p_data)
            mapper.ScalarVisibilityOff()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            r, g, b = colors[i % len(colors)]
            prop.SetColor(r, g, b)
            prop.SetAmbient(0.4)
            prop.SetDiffuse(0.6)
            prop.SetSpecular(0.2)
            prop.SetSpecularPower(20)
            prop.SetOpacity(1.0)

            self.renderer.AddActor(actor)
            self.actors[label_id] = actor
            self.mappers[label_id] = mapper
            self.poly_data_map[label_id] = p_data

        self.renderer.ResetCamera()
        self.camera.Zoom(1.1)
        self.vtk_widget.GetRenderWindow().Render()
        self.vtk_widget.update()

    def toggle_label_visibility(self, label_id, visible):
        if label_id not in self.actors: return
        self.actors[label_id].SetVisibility(visible)
        self.vtk_widget.GetRenderWindow().Render()
        self.vtk_widget.update()  # 【修复】强制 Qt 刷新渲染窗口

    def update_actor_property(self, label_id, key, value):
        if label_id not in self.actors: return
        actor = self.actors[label_id]
        prop = actor.GetProperty()

        if key == "color":
            prop.SetColor(value)
            prop.SetAmbientColor(value)
        elif key == "opacity": prop.SetOpacity(value)
        elif key == "ambient": prop.SetAmbient(value)
        elif key == "diffuse": prop.SetDiffuse(value)
        elif key == "specular": prop.SetSpecular(value)
        elif key == "specular_power": prop.SetSpecularPower(int(value))

        actor.Modified()
        self.vtk_widget.GetRenderWindow().Render()
        self.vtk_widget.update()