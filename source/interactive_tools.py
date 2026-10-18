"""Interactive constraints and a separate indexed-mesh mode for the existing Studio."""
import copy
import hashlib
import json
import threading
import uuid
import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot, Qt
from PySide6.QtWidgets import (QWidget, QFormLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QListWidget, QScrollArea, QMessageBox, QFileDialog)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser
from vtkmodules.vtkRenderingCore import vtkCellPicker
from interactive_geometry import (deform_control_surface, deform_mesh_vertices,
    ManipulationCancelled, surface_samples)


class ManipulationJob(QThread):
    result_ready = Signal(object)
    progress_ready = Signal(object)

    def __init__(self, generation, function, arguments, parent):
        super().__init__(parent)
        self.generation, self.function, self.arguments = generation, function, arguments
        self.cancelled = threading.Event()

    def run(self):
        def progress(current,total):
            if self.cancelled.is_set():
                raise ManipulationCancelled()
            self.progress_ready.emit((self.generation,current,total))
        try:
            result = self.function(**self.arguments, progress=progress)
            if self.cancelled.is_set():
                raise ManipulationCancelled()
            payload = (self.generation,result,None)
        except ManipulationCancelled:
            payload = (self.generation,None,'cancelled')
        except Exception as error:
            payload = (self.generation,None,str(error))
        self.result_ready.emit(payload)


class ManipulationBridge(QObject):
    """Explicit QObject slots keep all widget/VTK calls on the GUI thread."""
    def __init__(self, owner):
        super().__init__(owner)
        self.owner=owner

    @Slot(object)
    def result(self,payload):
        self.owner._irm_receive_result(payload)

    @Slot(object)
    def progress(self,payload):
        generation,current,total=payload
        if generation==self.owner._irm_generation:
            self.owner.irm_status.setText(f'Расчёт: {round(100*current/max(total,1))}% · можно отменить')


class InteractiveManipulationMixin:
    def __init__(self,*args,**kwargs):
        self._irm_controls=[]
        self._irm_pins=set()
        self._irm_selected=set()
        self._irm_mask_uids=[]
        self._irm_options={}
        self._irm_job=None
        self._irm_generation=0
        self._irm_result=None
        self._irm_drag=None
        self._irm_overlay=[]
        self._irm_enabled=False
        self._irm_token=None
        super().__init__(*args,**kwargs)

    def _install_manipulation(self):
        self._irm_bridge=ManipulationBridge(self)
        self._irm_timer=QTimer(self); self._irm_timer.setSingleShot(True)
        self._irm_timer.timeout.connect(self._irm_start_preview)
        page=QWidget(); form=QFormLayout(page); form.setRowWrapPolicy(QFormLayout.WrapLongRows); self._irm_form=form
        intro=QLabel('Interactive Relief Manipulation\nРельеф: щелчок создаёт точку; тяните точку вверх/вниз для изменения Z. Закреплённые точки неподвижны.\nZ отсчитывается от дна: при основании 5 мм ноль рельефа — Z=5 мм.\nСетка: выбирайте вершины, рёбра или грани; смещения X/Y/Z сохраняют нависающую геометрию. Shift — добавить к выделению.'); intro.setWordWrap(True)
        form.addRow(intro)
        self.irm_target=QComboBox(); self.irm_target.addItem('Рельеф · контрольные точки','relief'); self.irm_target.addItem('Mesh Edit · X/Y/Z','mesh')
        form.addRow('Режим',self.irm_target)
        row=QHBoxLayout()
        for label,slot in [('STL как сетка…',self.open_stl_mesh),('Рельеф → сетка',self.relief_to_mesh)]:
            button=QPushButton(label); button.clicked.connect(slot); row.addWidget(button)
        form.addRow(row)
        self.irm_enable_button=QPushButton('Выбирать и тянуть в 3D'); self.irm_enable_button.setCheckable(True)
        self.irm_enable_button.toggled.connect(self._irm_toggle)
        form.addRow(self.irm_enable_button)
        self.irm_selection_kind=QComboBox(); self.irm_selection_kind.addItems(['Вершины','Рёбра','Грани'])
        form.addRow('Выбор в сетке',self.irm_selection_kind)
        self.irm_list=QListWidget(); self.irm_list.setMaximumHeight(140)
        self.irm_list.currentRowChanged.connect(self._irm_sync_point_fields); form.addRow(self.irm_list)
        self.irm_z=self._spin(5,.05,100000,4,' мм',.1)
        self._irm_z_dirty=False
        self.irm_z.valueChanged.connect(self._irm_mark_z_edit)
        self.irm_z.editingFinished.connect(self._irm_numeric_point)
        form.addRow('Z точки / поверхности',self.irm_z)
        for label,slot in [('Закрепить / снять',self._irm_pin),('Удалить точку',self._irm_delete_point),('Снять выделение',self._irm_clear_selection)]:
            button=QPushButton(label); button.clicked.connect(slot); form.addRow(button)
            if slot==self._irm_delete_point: self.irm_delete_button=button
        self.irm_surface_mode=QComboBox(); self.irm_surface_mode.addItem('Local Deformation','local'); self.irm_surface_mode.addItem('Full Surface','full')
        self.irm_radius=self._spin(15,.01,10000,3,' мм',1)
        self.irm_strength=self._spin(1,0,2,3,'',.05)
        self.irm_gain=self._spin(.05,.001,10,3,' мм/px',.01)
        self.irm_feather=self._spin(3,.01,1000,3,' мм',.5)
        self.irm_fixed=QCheckBox('Закрепить границу области'); self.irm_fixed.setChecked(True)
        self.irm_mask=QCheckBox('Ограничить выбранными контурами 2D'); self.irm_mask.setChecked(True)
        self.irm_mask.setToolTip('Контуры области запоминаются в проекте. Для смены области выделите другие замкнутые векторы. Для всей плиты отключите ограничение.')
        for label,widget in [('Деформация',self.irm_surface_mode),('Радиус / мягкий выбор',self.irm_radius),('Сила (1 = заданная высота)',self.irm_strength),('Перетаскивание',self.irm_gain),('Плавность края',self.irm_feather)]:
            form.addRow(label,widget)
        form.addRow(self.irm_fixed); form.addRow(self.irm_mask)
        self.irm_dx,self.irm_dy,self.irm_dz=[self._spin(0,-100000,100000,4,' мм',.1) for _ in range(3)]
        for label,widget in [('Сетка: смещение X',self.irm_dx),('Смещение Y',self.irm_dy),('Смещение Z',self.irm_dz)]: form.addRow(label,widget)
        self.irm_soft=QCheckBox('Мягкое пропорциональное выделение'); self.irm_soft.setChecked(True); form.addRow(self.irm_soft)
        self.irm_drag_axis=QComboBox(); self.irm_drag_axis.addItems(['Z','X','Y']); form.addRow('Тянуть в сетке по оси',self.irm_drag_axis)
        row=QHBoxLayout()
        self.irm_preview_button=QPushButton('Предпросмотр'); self.irm_preview_button.clicked.connect(self._irm_start_preview)
        self.irm_apply_button=QPushButton('Применить'); self.irm_apply_button.setProperty('primary',True); self.irm_apply_button.setEnabled(False); self.irm_apply_button.clicked.connect(self._irm_apply)
        cancel=QPushButton('Отменить просмотр'); cancel.clicked.connect(self._irm_cancel)
        for button in (self.irm_preview_button,self.irm_apply_button): row.addWidget(button)
        form.addRow(row)
        form.addRow(cancel)
        self.irm_status=QLabel('Изменения применяются только кнопкой «Применить». Один результат — один шаг Undo.'); self.irm_status.setWordWrap(True); form.addRow(self.irm_status)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(page)
        self.irm_tab=self.inspector_tabs.addTab(scroll,'Контроль 3D')
        self.irm_action=self._action('Контроль рельефа / Mesh Edit','point',lambda:self.show_inspector(self.irm_tab))
        for toolbar in self.findChildren(type(self.snapbar)):
            if toolbar.windowTitle()=='Создание и редактирование рельефа': toolbar.addAction(self.irm_action)
        for menu in self.menuBar().findChildren(type(self.recent_menu)):
            if menu.title()=='Рельефы': menu.addAction(self.irm_action)
        self.irm_target.currentIndexChanged.connect(self._irm_target_changed)
        interactor=self.plotter.iren.interactor
        for event,callback in [('LeftButtonPressEvent',self._irm_press),('MouseMoveEvent',self._irm_move),('LeftButtonReleaseEvent',self._irm_release)]: interactor.AddObserver(event,callback,1.1)
        self._irm_restore_state(self._irm_state())

    def _irm_update_rows(self):
        relief=self.irm_target.currentData()=='relief'
        groups=[(relief,[self.irm_z,self.irm_delete_button,self.irm_surface_mode,self.irm_feather,self.irm_fixed,self.irm_mask]),
                (not relief,[self.irm_selection_kind,self.irm_dx,self.irm_dy,self.irm_dz,self.irm_soft,self.irm_drag_axis])]
        for visible,widgets in groups:
            for widget in widgets:
                widget.setVisible(visible)
                label=self._irm_form.labelForField(widget)
                if label is not None: label.setVisible(visible)

    def _irm_state(self):
        options=dict(self._irm_options)
        if hasattr(self,'irm_target'):
            options.update(target=self.irm_target.currentData(),surface_mode=self.irm_surface_mode.currentData(),
                radius=self.irm_radius.value(),strength=self.irm_strength.value(),gain=self.irm_gain.value(),
                feather=self.irm_feather.value(),fixed=self.irm_fixed.isChecked(),use_mask=self.irm_mask.isChecked(),
                mesh_delta=[self.irm_dx.value(),self.irm_dy.value(),self.irm_dz.value()],soft=self.irm_soft.isChecked(),
                selection_kind=self.irm_selection_kind.currentIndex(),drag_axis=self.irm_drag_axis.currentIndex())
        bindings=list(self._irm_mask_uids)
        if self._irm_controls and options.get('use_mask',True) and hasattr(self,'view'):
            selected=[v.data.uid for v in self.sel_vectors() if v.data.closed]
            if selected: bindings=selected
        return dict(controls=copy.deepcopy(self._irm_controls),pins=sorted(self._irm_pins),selected=sorted(self._irm_selected),mask_vectors=bindings,options=options)

    def _irm_restore_state(self,state):
        self._irm_controls=copy.deepcopy(state.get('controls',[])); self._irm_pins=set(state.get('pins',[]))
        self._irm_selected=set(state.get('selected',[])); self._irm_options=dict(state.get('options',{}))
        self._irm_mask_uids=list(state.get('mask_vectors',[]))
        if not hasattr(self,'irm_target'): return
        for key,widget in [('radius',self.irm_radius),('strength',self.irm_strength),('gain',self.irm_gain),('feather',self.irm_feather)]:
            widget.blockSignals(True); widget.setValue(self._irm_options.get(key,widget.value())); widget.blockSignals(False)
        for key,widget in [('fixed',self.irm_fixed),('use_mask',self.irm_mask)]: widget.setChecked(self._irm_options.get(key,True))
        for key,widget,default in [('target',self.irm_target,'relief'),('surface_mode',self.irm_surface_mode,'local')]:
            widget.blockSignals(True); widget.setCurrentIndex(max(0,widget.findData(self._irm_options.get(key,default)))); widget.blockSignals(False)
        for widget,value in zip((self.irm_dx,self.irm_dy,self.irm_dz),self._irm_options.get('mesh_delta',[0,0,0])): widget.setValue(value)
        self.irm_soft.setChecked(self._irm_options.get('soft',True))
        self.irm_selection_kind.setCurrentIndex(self._irm_options.get('selection_kind',0))
        self.irm_drag_axis.setCurrentIndex(self._irm_options.get('drag_axis',0))
        self._irm_update_rows()
        self._irm_refresh_list()

    def _settings(self):
        return dict(super()._settings(),interactive=self._irm_state())

    def after_change(self):
        if self.stl_mesh is None:
            self._irm_pins.clear(); self._irm_selected.clear()
        return super().after_change()

    def state_sig(self):
        return hashlib.sha256((super().state_sig()+json.dumps(self._irm_state(),sort_keys=True)).encode()).hexdigest()

    def snapshot(self):
        return dict(super().snapshot(),interactive=self._irm_state())

    def restore(self,state):
        self._irm_cancel(silent=True)
        self._irm_restore_state(state.get('interactive',{}))
        super().restore(state)
        # A mesh-to-mesh undo has z=None on both sides: the old core redraws only
        # when the Z reference changes. Explicitly rebuild this actor as well.
        if self.z is None and self.stl_mesh is not None: self.show_mesh()
        self._irm_refresh_overlay()

    def _clear_all(self):
        self._irm_cancel(silent=True)
        if hasattr(self,'irm_enable_button'): self.irm_enable_button.setChecked(False)
        self._irm_restore_state({})
        super()._clear_all()

    def _write(self,path):
        self._irm_release()
        return super()._write(path)

    def confirm_discard(self):
        self._irm_release()
        self._irm_cancel(silent=False)
        if self._irm_job is not None and self._irm_job.isRunning():
            self._irm_job.cancelled.set()
            if not self._irm_job.wait(3000):
                self.statusBar().showMessage('Дождитесь завершения отмены расчёта перед закрытием.'); return False
        return super().confirm_discard()

    def cancel_preview(self,silent=False):
        self._irm_cancel(silent=silent)
        return super().cancel_preview(silent)

    def show_mesh(self,*args,**kwargs):
        result=super().show_mesh(*args,**kwargs)
        if hasattr(self,'irm_target'): self._irm_refresh_overlay()
        return result

    def _irm_target_changed(self,*_):
        if self._irm_enabled: self.irm_enable_button.setChecked(False)
        self._irm_update_rows()
        self._irm_cancel(); self._irm_refresh_list()

    def _irm_toggle(self,enabled):
        self._irm_release()
        if enabled:
            if (self.irm_target.currentData()=='relief' and self.z is None) or (self.irm_target.currentData()=='mesh' and self.stl_mesh is None):
                self.irm_enable_button.setChecked(False)
                self.irm_status.setText('Создайте рельеф или откройте STL кнопкой «STL как сетка».'); return
            if self.sculpt_action.isChecked():
                self.sculpt_action.setChecked(False); self.toggle_sculpt(False)
            self._irm_camera_style=self.plotter.iren.interactor.GetInteractorStyle()
            self._irm_style=vtkInteractorStyleUser()
            self.plotter.iren.interactor.SetInteractorStyle(self._irm_style)
            self.tabs.setCurrentIndex(1); self.show_inspector(self.irm_tab)
        elif hasattr(self,'_irm_camera_style'):
            self.plotter.iren.interactor.SetInteractorStyle(self._irm_camera_style)
        self._irm_enabled=enabled
        self._irm_refresh_overlay()

    def _irm_refresh_list(self,selected=None):
        if not hasattr(self,'irm_list'): return
        current=self.irm_list.currentRow() if selected is None else selected
        self.irm_list.blockSignals(True); self.irm_list.clear()
        if self.irm_target.currentData()=='relief':
            for i,point in enumerate(self._irm_controls): self.irm_list.addItem(f'{i+1}. Z {point["z"]:.4f} мм'+(' · закреплена' if point.get('pinned') else ''))
        else:
            self.irm_list.addItem(f'Выбрано вершин: {len(self._irm_selected)}; закреплено: {len(self._irm_pins)}')
        self.irm_list.setCurrentRow(min(max(current,0),self.irm_list.count()-1)); self.irm_list.blockSignals(False)
        self._irm_sync_point_fields(self.irm_list.currentRow()); self._irm_refresh_overlay()

    def _irm_sync_point_fields(self,row):
        valid=self.irm_target.currentData()=='relief' and 0<=row<len(self._irm_controls)
        self.irm_z.setEnabled(valid and not self._irm_controls[row].get('pinned'))
        if valid:
            self.irm_z.blockSignals(True); self.irm_z.setValue(self._irm_controls[row]['z']); self.irm_z.blockSignals(False)
        self._irm_z_dirty=False

    def _irm_mark_z_edit(self,*_):
        self._irm_z_dirty=True

    def _irm_numeric_point(self):
        if not self._irm_z_dirty: return
        self._irm_z_dirty=False
        row=self.irm_list.currentRow()
        if self.irm_target.currentData()!='relief' or not 0<=row<len(self._irm_controls): return
        point=self._irm_controls[row]
        if point.get('pinned') or point['z']==self.irm_z.value(): return
        snap=self.snapshot(); point['z']=self.irm_z.value()
        self.push_history('Высота контрольной точки',snap); self._irm_refresh_list(row); self._irm_schedule_preview()

    def _irm_pin(self):
        snap=self.snapshot()
        if self.irm_target.currentData()=='mesh':
            if not self._irm_selected: return
            if self._irm_selected<=self._irm_pins: self._irm_pins-=self._irm_selected
            else: self._irm_pins|=self._irm_selected
        else:
            row=self.irm_list.currentRow()
            if not 0<=row<len(self._irm_controls): return
            point=self._irm_controls[row]; point['pinned']=not point.get('pinned')
        self.push_history('Закрепление контрольных точек',snap); self._irm_refresh_list(); self._irm_schedule_preview()

    def _irm_delete_point(self):
        row=self.irm_list.currentRow()
        if self.irm_target.currentData()!='relief' or not 0<=row<len(self._irm_controls): return
        snap=self.snapshot(); self._irm_controls.pop(row)
        self.push_history('Удаление контрольной точки',snap); self._irm_cancel(); self._irm_refresh_list()

    def _irm_clear_selection(self):
        self._irm_selected.clear(); self.irm_list.setCurrentRow(-1); self._irm_refresh_overlay()

    def _irm_refresh_overlay(self):
        if not hasattr(self,'plotter') or not hasattr(self,'irm_target'): return
        for actor in self._irm_overlay:
            try: self.plotter.remove_actor(actor,render=False)
            except (RuntimeError,AttributeError): pass
        self._irm_overlay=[]
        if not self._irm_enabled: return
        import pyvista as pv
        if self.irm_target.currentData()=='relief':
            groups=[([p for p in self._irm_controls if p.get('pinned')], '#69b8ed'),([p for p in self._irm_controls if not p.get('pinned')], '#f1b04c')]
            for points,color in groups:
                if points:
                    xyz=np.array([[p['x'],p['y'],p['z']] for p in points])
                    self._irm_overlay.append(self.plotter.add_mesh(pv.PolyData(xyz),color=color,point_size=15,render_points_as_spheres=True,lighting=False,pickable=False,reset_camera=False,render=False))
        elif self.stl_mesh is not None:
            vertices=self._irm_result[1].vertices if self._irm_result is not None and self._irm_result[0]=='mesh' else self.stl_mesh.vertices
            for indices,color in [(self._irm_pins,'#69b8ed'),(self._irm_selected,'#f1b04c')]:
                valid=[i for i in indices if 0<=i<len(vertices)]
                if valid: self._irm_overlay.append(self.plotter.add_mesh(pv.PolyData(vertices[valid]),color=color,point_size=13,render_points_as_spheres=True,lighting=False,pickable=False,reset_camera=False,render=False))

    def _irm_press(self,*_):
        if not self._irm_enabled: return
        self._irm_release()
        interactor=self.plotter.iren.interactor; x,y=interactor.GetEventPosition()
        if self.irm_target.currentData()=='relief':
            renderer=self.plotter.renderer; closest=None
            for i,point in enumerate(self._irm_controls):
                renderer.SetWorldPoint(point['x'],point['y'],point['z'],1); renderer.WorldToDisplay()
                sx,sy,_=renderer.GetDisplayPoint(); distance=(sx-x)**2+(sy-y)**2
                if distance<18**2 and (closest is None or distance<closest[0]): closest=(distance,i)
            if closest is not None:
                row=closest[1]; self.irm_list.setCurrentRow(row)
                if not self._irm_controls[row].get('pinned'):
                    self._irm_drag=dict(row=row,y=y,z=self._irm_controls[row]['z'],snapshot=self.snapshot())
                return
        picker=vtkCellPicker(); picker.SetTolerance(.005); picker.PickFromListOn()
        if self.actor is None: return
        picker.AddPickList(self.actor)
        if not picker.Pick(x,y,0,self.plotter.renderer): return
        position=np.asarray(picker.GetPickPosition())
        if self.irm_target.currentData()=='relief' and self.z is not None:
            if len(self._irm_controls)>=128: self.irm_status.setText('Достигнут предел 128 точек.'); return
            if abs(position[0])>self.sp_w.value()/2 or abs(position[1])>self.sp_h.value()/2: return
            snap=self.snapshot(); position[2]=surface_samples(self.z,[position[:2]],self.sp_w.value(),self.sp_h.value())[0]
            self._irm_controls.append(dict(id=uuid.uuid4().hex,x=float(position[0]),y=float(position[1]),z=float(position[2]),pinned=False))
            self.push_history('Контрольная точка на поверхности',snap)
            row=len(self._irm_controls)-1; self._irm_refresh_list(row)
            self._irm_drag=dict(row=row,y=y,z=float(position[2]),snapshot=self.snapshot())
        elif self.stl_mesh is not None:
            face=picker.GetCellId()
            if not 0<=face<len(self.stl_mesh.faces): return
            ids=np.asarray(self.stl_mesh.faces[face]); xyz=np.asarray(self.stl_mesh.vertices[ids])
            mode=self.irm_selection_kind.currentIndex()
            if mode==0: picked={int(ids[np.argmin(np.linalg.norm(xyz-position,axis=1))])}
            elif mode==2: picked=set(map(int,ids))
            else:
                pairs=[(0,1),(1,2),(2,0)]
                def distance(pair):
                    a,b=xyz[list(pair)]; d=b-a; t=np.clip(np.dot(position-a,d)/max(np.dot(d,d),1e-20),0,1)
                    return np.linalg.norm(position-a-t*d)
                picked=set(map(int,ids[list(min(pairs,key=distance))]))
            if not interactor.GetShiftKey(): self._irm_selected.clear()
            self._irm_selected|=picked; self._irm_refresh_list()
            self._irm_drag=dict(y=y,axis=self.irm_drag_axis.currentIndex(),offset=0.)
        self.plotter.render()

    def _irm_move(self,*_):
        if self._irm_drag is None: return
        _,y=self.plotter.iren.interactor.GetEventPosition()
        offset=(y-self._irm_drag['y'])*self.irm_gain.value()
        if self.irm_target.currentData()=='relief':
            row=self._irm_drag['row']; point=self._irm_controls[row]
            point['z']=max(.05,self._irm_drag['z']+offset)
            self._irm_sync_point_fields(row); self._irm_refresh_overlay(); self.plotter.render()
        else:
            self._irm_drag['offset']=offset
            # Mesh axis list is Z,X,Y; updates are previews, never in-place edits.
            widgets=[self.irm_dz,self.irm_dx,self.irm_dy]
            widgets[self._irm_drag['axis']].setValue(offset)
        self._irm_schedule_preview()

    def _irm_release(self,*_):
        if self._irm_drag is None: return
        drag,self._irm_drag=self._irm_drag,None
        if 'row' in drag and self._irm_controls[drag['row']]['z']!=drag['z']:
            self.push_history('Перемещение контрольной точки',drag['snapshot']); self._irm_refresh_list(drag['row']); self._irm_schedule_preview()
        elif 'axis' in drag and drag['offset']!=0: self._irm_schedule_preview()

    def _irm_schedule_preview(self):
        if self._irm_job is not None and self._irm_job.isRunning(): self._irm_job.cancelled.set()
        self._irm_timer.start(180)

    def _irm_start_preview(self):
        if self._irm_job is not None and self._irm_job.isRunning():
            self._irm_job.cancelled.set(); self._irm_timer.start(100); return
        self._irm_timer.stop()
        super().cancel_preview(silent=True)
        self._irm_result=None; self.irm_apply_button.setEnabled(False)
        target=self.irm_target.currentData()
        try:
            if target=='relief':
                if self.z is None: raise ValueError('Создайте карту высот. Для исходного STL выберите Mesh Edit.')
                mask=None
                if self.irm_mask.isChecked():
                    selected=[v for v in self.sel_vectors() if v.data.closed]
                    if selected: self._irm_mask_uids=[v.data.uid for v in selected]
                    if self._irm_mask_uids:
                        by_uid={v.data.uid:v for v in self.view.vectors()}
                        if any(uid not in by_uid for uid in self._irm_mask_uids):
                            raise ValueError('Контур сохранённой области удалён. Выделите новую область или отключите ограничение.')
                        from studio_core import raster_coverage
                        polys=[[self.scene_to_grid(x,y) for x,y in by_uid[uid].scene_points()] for uid in self._irm_mask_uids]
                        mask=raster_coverage(polys,self.z.shape[1],self.z.shape[0])>.5
                function=deform_control_surface
                arguments=dict(z=self.z,controls=copy.deepcopy(self._irm_controls),width=self.sp_w.value(),height=self.sp_h.value(),mask=mask,
                    fixed=self.irm_fixed.isChecked(),mode=self.irm_surface_mode.currentData(),radius=self.irm_radius.value(),strength=self.irm_strength.value(),feather=self.irm_feather.value())
            else:
                if self.stl_mesh is None: raise ValueError('Откройте STL как сетку или скопируйте рельеф в сетку.')
                function=deform_mesh_vertices
                arguments=dict(vertices=np.asarray(self.stl_mesh.vertices),selected=list(self._irm_selected),pinned=list(self._irm_pins),
                    delta=[self.irm_dx.value(),self.irm_dy.value(),self.irm_dz.value()],radius=self.irm_radius.value() if self.irm_soft.isChecked() else 0,strength=self.irm_strength.value())
            self._irm_generation+=1; self._irm_token=self.state_sig(); self._irm_job_target=target
            if self._irm_job is not None: self._irm_job.deleteLater()
            self._irm_job=ManipulationJob(self._irm_generation,function,arguments,self)
            self._irm_job.result_ready.connect(self._irm_bridge.result,Qt.QueuedConnection)
            self._irm_job.progress_ready.connect(self._irm_bridge.progress,Qt.QueuedConnection)
            self.irm_status.setText('Расчёт в рабочем потоке…'); self._irm_job.start()
        except Exception as error:
            self.irm_status.setText(str(error))

    def _irm_receive_result(self,payload):
        generation,result,error=payload
        if generation!=self._irm_generation: return
        if error:
            self.irm_status.setText('Расчёт отменён; исходная геометрия сохранена.' if error=='cancelled' else error); return
        if self.state_sig()!=self._irm_token:
            self.irm_status.setText('Модель или параметры изменились. Повторите предпросмотр.'); return
        if self._irm_job_target=='mesh':
            import trimesh
            result=trimesh.Trimesh(vertices=result,faces=np.asarray(self.stl_mesh.faces).copy(),process=False)
        self._irm_result=(self._irm_job_target,result)
        self._irm_draw_preview()
        self.irm_apply_button.setEnabled(True)
        self.irm_status.setText('Предпросмотр готов. Исходная геометрия не изменена.'+(' После деформации сетки проверьте замкнутость STL.' if self._irm_job_target=='mesh' else ''))

    def _irm_draw_preview(self):
        target,result=self._irm_result
        if target=='relief': self.show_mesh(result)
        else:
            original=self.stl_mesh
            try: self.stl_mesh=result; self.show_mesh()
            finally: self.stl_mesh=original

    def _irm_apply(self):
        self._irm_release()
        self._irm_timer.stop()
        if self._irm_result is None or self.state_sig()!=self._irm_token:
            self.irm_status.setText('Сначала выполните актуальный предпросмотр.'); return False
        target,result=self._irm_result; snap=self.snapshot()
        self.push_history('Interactive Relief Manipulation' if target=='relief' else 'Mesh Edit · X/Y/Z',snap)
        if target=='relief': self.z=result
        else: self.stl_mesh=result
        self._irm_result=None; self.irm_apply_button.setEnabled(False)
        self.show_mesh(); self.after_change(); self.irm_status.setText('Применено. Ctrl+Z — отменить всю деформацию.')
        return True

    def _irm_cancel(self,checked=False,silent=False):
        if hasattr(self,'_irm_timer'): self._irm_timer.stop()
        if self._irm_job is not None and self._irm_job.isRunning(): self._irm_job.cancelled.set()
        self._irm_generation+=1
        had_preview=self._irm_result is not None; self._irm_result=None
        if hasattr(self,'irm_apply_button'): self.irm_apply_button.setEnabled(False)
        if had_preview and not silent: self.show_mesh()
        if not silent and hasattr(self,'irm_status'):
            self.irm_status.setText('Предпросмотр отменён; исходная геометрия сохранена.')

    def open_stl_mesh(self,path=None):
        if not isinstance(path,str): path,_=QFileDialog.getOpenFileName(self,'STL без преобразования в карту высот','','STL (*.stl)')
        if not path: return False
        try:
            import trimesh
            from input_limits import read_checked_file
            import io
            mesh=trimesh.load(io.BytesIO(read_checked_file(path,256*1024**2)),file_type='stl',force='mesh',process=False)
            vertices,inverse=np.unique(np.asarray(mesh.vertices),axis=0,return_inverse=True)
            faces=inverse[np.asarray(mesh.faces)]
            if not len(faces) or not np.isfinite(vertices).all() or np.any(np.ptp(vertices,axis=0)<=0): raise ValueError('STL пуст, содержит неверные координаты или нулевой объём габаритов.')
            mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
            if np.any(mesh.extents<1e-6) or np.any(mesh.extents>1e9):
                raise ValueError('Размер STL вне диапазона 0,000001…1e9 мм. Проверьте единицы и масштаб исходной модели.')
        except Exception as error:
            QMessageBox.warning(self,'STL не открыт',str(error)); return False
        self._irm_cancel(silent=True); self._sculpt_release()
        snap=self.snapshot(); self.push_history('Импорт STL как исходной сетки',snap)
        self.z=None; self.stl_mesh=mesh; self._irm_restore_state({})
        for widget,value in ((self.sp_w,float(mesh.extents[0])),(self.sp_h,float(mesh.extents[1]))):
            widget.blockSignals(True); widget.setDecimals(6)
            widget.setRange(min(widget.minimum(),value),max(widget.maximum(),value)); widget.setValue(value); widget.blockSignals(False)
        self.irm_target.setCurrentIndex(1)
        self.show_mesh(reset=True); self.after_change(); self.show_inspector(self.irm_tab)
        self.irm_status.setText('Исходная XYZ-сетка открыта без карты высот. Единицы STL интерпретируются как мм.')
        return True

    def relief_to_mesh(self):
        if self.z is None: self.irm_status.setText('Сначала создайте рельеф.'); return False
        from studio_core import build_mesh_arrays
        import trimesh
        self._irm_cancel(silent=True); self._sculpt_release(); snap=self.snapshot()
        vertices,faces=build_mesh_arrays(self.z,self.sp_w.value(),self.sp_h.value())
        self.push_history('Рельеф → Mesh Edit',snap)
        self.stl_mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False); self.z=None
        self._irm_restore_state({}); self.irm_target.setCurrentIndex(1)
        self.show_mesh(); self.after_change(); self.show_inspector(self.irm_tab)
        self.irm_status.setText('Создана сетка полного разрешения. Ctrl+Z возвращает исходную карту высот.')
        return True
