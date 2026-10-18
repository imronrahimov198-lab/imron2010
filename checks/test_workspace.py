import io
import json
import sys
import zipfile

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPointF, Qt, QSettings
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

import stone_relief_studio_fixed as core
from stone_relief_studio_fixed import StoneStudio


@pytest.fixture(scope='session')
def app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    messages = []
    monkeypatch.setattr(QMessageBox, 'information', lambda *a, **k: messages.append(a[1:]))
    monkeypatch.setattr(QMessageBox, 'critical', lambda *a, **k: messages.append(a[1:]))
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: messages.append(a[1:]))
    preferences = QSettings(str(tmp_path/'workspace.ini'), QSettings.IniFormat)
    monkeypatch.setattr(sys.modules[StoneStudio.__module__], 'QSettings', lambda *a: preferences)
    w = StoneStudio()
    w.resize(1360, 900)
    w.sp_w.setValue(60); w.sp_h.setValue(60); w.sp_step.setValue(.5)
    w.show(); w.activateWindow(); w.raise_(); app.processEvents(); app.processEvents()
    yield w
    w._saved_sig = w.state_sig()
    w.close()
    app.processEvents()
    assert not messages, messages


def add_sweep(w):
    # Rails in scene coordinates, with a cubic arch as the cross-section.
    a = w._add_vec(core.polyline_path([(10*w.K, 10*w.K), (10*w.K, 50*w.K)]))
    b = w._add_vec(core.polyline_path([(40*w.K, 10*w.K), (40*w.K, 50*w.K)]))
    profile = core.PathData([core.Node(0,0,hout=(10*w.K,-10*w.K)),
                             core.Node(30*w.K,0,hin=(20*w.K,-10*w.K))])
    p = w._add_vec(profile)
    w.sw_rails=[a,b]; w.sw_sections=[p]
    return a,b,p


def test_circle_mouse_drawing_and_undo(window, app):
    w=window
    w.choose_shape('circle')
    a=w.view.mapFromScene(QPointF(10*w.K,10*w.K))
    b=w.view.mapFromScene(QPointF(30*w.K,35*w.K))
    QTest.mousePress(w.view.viewport(),Qt.LeftButton,pos=a)
    QTest.mouseMove(w.view.viewport(),b)
    QTest.mouseRelease(w.view.viewport(),Qt.LeftButton,pos=b)
    app.processEvents()
    assert len(w.view.vectors())==1
    p=w.view.vectors()[0].data
    assert p.closed and len(p.nodes)==4
    assert any(n.hin for n in p.nodes)
    w.undo()
    assert not w.view.vectors()


def test_node_nudge_tab_delete_and_history(window, app):
    w=window
    item=w._add_vec(core.polyline_path([(10,10),(40,10),(40,40),(10,40)],True),True)
    uid=item.data.uid
    item.overlay.keyboard_handle=('n',0)
    w.view.setFocus()
    x=item.data.nodes[0].x
    QTest.keyClick(w.view,Qt.Key_Right,Qt.ControlModifier)
    assert item.data.nodes[0].x == pytest.approx(x+.01*w.K)
    w.undo()
    item=next(v for v in w.view.vectors() if v.data.uid==uid)
    assert item.data.nodes[0].x==x
    QTest.keyClick(w.view,Qt.Key_Tab)
    assert item.overlay.keyboard_handle==('n',0)
    QTest.keyClick(w.view,Qt.Key_Delete)
    assert len(item.data.nodes)==3
    w.undo()
    assert len(w.view.vectors()[0].data.nodes)==4


def test_svg_import_preserves_size(window, tmp_path):
    w=window
    path=tmp_path/'pattern.svg'
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="50mm" height="50mm" viewBox="0 0 50 50"><circle cx="20" cy="20" r="10"/></svg>')
    w.import_svg(str(path))
    data=w.view.vectors()[0].data
    bounds=core.build_qpath(data).boundingRect()
    assert bounds.width()/w.K==pytest.approx(20,abs=1e-4)


def test_dxf_inches_are_converted_to_mm(window, tmp_path):
    import ezdxf
    from ezdxf import units
    w=window
    doc=ezdxf.new(); doc.units=units.IN
    doc.modelspace().add_line((0,0),(1,0))
    path=tmp_path/'inch.dxf'; doc.saveas(path)
    w.import_dxf(str(path))
    p=w.view.vectors()[0].data
    assert abs(p.nodes[1].x-p.nodes[0].x)/w.K==pytest.approx(25.4)


def test_photo_changes_are_dirty_and_undo_restores_previous(window, tmp_path):
    w=window
    red=tmp_path/'red.png'; blue=tmp_path/'blue.png'
    Image.new('RGB',(30,30),'red').save(red)
    Image.new('RGB',(30,30),'blue').save(blue)
    w.open_photo(str(red)); first=w.photo_bytes
    signature=w.state_sig()
    w.open_photo(str(blue))
    assert w.state_sig()!=signature
    w.undo()
    assert w.photo_bytes==first


def test_sweep_preview_save_restore_and_stl(window, tmp_path, app):
    w=window; w.create_plate()
    a,b,p=add_sweep(w)
    w.sw_positions[p.data.uid]=37
    w.sw_start.setValue(.75)
    base=w.z.copy()
    w.sw_run(True)
    assert w.preview_active
    assert w.z==pytest.approx(base)
    assert w.preview_z.max()>base.max()+1
    w.d_dim.setValue(30); w._refresh_preview()
    assert w._preview_tool=='sweep'
    w.sw_run(False)
    assert w.z.max()>base.max()+1
    path=tmp_path/'petal.srelief'
    assert w._write(str(path))
    result=w.z.copy()
    w.open_project(str(path))
    assert w.z==pytest.approx(result)
    assert [v.data.uid for v in w.sw_rails]==[a.data.uid,b.data.uid]
    assert w.sw_positions[p.data.uid]==37
    assert w._saved_sig==w.state_sig()
    vertices,faces=core.build_mesh_arrays(w.z,60,60)
    output=tmp_path/'petal.stl'; core.write_stl(output,vertices,faces)
    import trimesh
    mesh=trimesh.load(output)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.extents[:2]==pytest.approx([60,60])
    w.plotter.render()
    image=w.plotter.screenshot(return_img=True)
    assert image is not None and image.std()>1


def test_model_controls_and_resizable_modeless_dialogs(window, app):
    w=window
    assert w.sp_w.isVisible()
    assert w.sp_step.isVisible()
    assert w.snapbar.isVisible()
    for dialog in (w.sweep_dialog,w.relief_dialog):
        assert not dialog.isModal()
        dialog.resize(650,760); dialog.show(); app.processEvents()
        assert dialog.width()>=650
        if dialog is w.relief_dialog:
            assert w.sh_angle.isVisible()
            assert w.sh_start.isVisible()
            assert w.sh_view.isVisible()
        dialog.hide()
    assert w.settings.value('sweep/geometry')


def test_sculpt_gesture_changes_height_and_undo(window):
    w=window; w.create_plate()
    original=w.z.copy()
    weight=np.zeros(w.z.shape,dtype=np.float32); weight[20:40,20:40]=1
    w._stroke=dict(y=0,weight=weight,base=w.z,snapshot=w.snapshot(),changed=False)
    w.plotter.iren.interactor.SetEventInformation(0,20)
    w._sculpt_move()
    assert w.z[25,25]>original[25,25]
    w._sculpt_release(); w.undo()
    assert w.z==pytest.approx(original)


@pytest.mark.parametrize('protect', [True, False])
def test_sculpt_zero_relief_and_fixed_neighbours(window, protect):
    w=window; w.create_plate()
    original=w.z.copy(); original[20:40,20:40]+=2; original[0,0]=1
    w.z=original
    w.sculpt_protect_base.setChecked(protect)
    weight=np.zeros(w.z.shape,dtype=np.float32); weight[20:40,20:40]=1
    w._stroke=dict(y=0,weight=weight,base=w.z,snapshot=w.snapshot(),changed=False)
    w.plotter.iren.interactor.SetEventInformation(0,-1000)
    w._sculpt_move(); w._sculpt_release()
    assert w.z[25,25]==pytest.approx(w.sp_base.value() if protect else .05)
    assert w.z[0,0]==1
    assert w.z[60,60]==original[60,60]
    w.undo(); assert np.array_equal(w.z,original)


def test_sculpt_clamped_noop_does_not_add_undo(window):
    w=window; w.create_plate(); count=len(w.undo_stack)
    w._stroke=dict(y=0,weight=np.ones(w.z.shape),base=w.z,snapshot=w.snapshot(),changed=False)
    w.plotter.iren.interactor.SetEventInformation(0,-1000)
    w._sculpt_move(); w._sculpt_release()
    assert len(w.undo_stack)==count


@pytest.mark.parametrize('answer', [QMessageBox.Save, QMessageBox.Discard, QMessageBox.Cancel])
def test_close_during_sculpt_finishes_gesture_before_prompt(window, tmp_path, monkeypatch, answer):
    w=window; w.create_plate(); w._saved_sig=w.state_sig()
    original=w.z.copy(); path=tmp_path/'last-gesture.srelief'; w.project_path=str(path)
    w._stroke=dict(y=0,weight=np.ones(w.z.shape),base=w.z,snapshot=w.snapshot(),changed=False)
    w.plotter.iren.interactor.SetEventInformation(0,20); w._sculpt_move()
    expected=w.z.copy(); prompts=[]
    def choose(*args):
        assert w._stroke is None
        assert w.undo_stack[-1][0]=='Лепка поверхности'
        prompts.append(True)
        return answer
    monkeypatch.setattr(QMessageBox,'question',choose)
    w.close()
    assert len(prompts)==1
    if answer==QMessageBox.Save:
        _,_,data=core.unpack_project(path)
        assert np.array_equal(data['z'],expected)
    elif answer==QMessageBox.Cancel:
        assert w.isVisible()
        assert np.array_equal(w.z,expected)
        w.undo(); assert np.array_equal(w.z,original)
    assert w._stroke is None


def test_raw_stl_mesh_survives_project_roundtrip_and_is_dirty(window, tmp_path, monkeypatch):
    import trimesh
    w=window; baseline=w.state_sig()
    w.stl_mesh=trimesh.creation.box(extents=(2.54,3.25,4.5))
    assert w.state_sig()!=baseline
    for widget,value in ((w.sp_w,2.54),(w.sp_h,3.25)):
        widget.blockSignals(True); widget.setDecimals(6); widget.setMinimum(.001)
        widget.setValue(value); widget.blockSignals(False)
    vertices,faces=w.stl_mesh.vertices.copy(),w.stl_mesh.faces.copy()
    path=tmp_path/'raw-mesh.srelief'; assert w._write(str(path))
    # Restore standard ranges to reproduce opening a small model in a new session.
    for widget in (w.sp_w,w.sp_h):
        widget.blockSignals(True); widget.setRange(10,3000); widget.setDecimals(1); widget.blockSignals(False)
    monkeypatch.setattr(w,'confirm_discard',lambda:True)
    assert w.open_project(str(path))
    assert w.z is None
    assert np.array_equal(w.stl_mesh.vertices,vertices)
    assert np.array_equal(w.stl_mesh.faces,faces)
    assert w.sp_w.value()==2.54 and w.sp_h.value()==3.25
    assert w._saved_sig==w.state_sig()


@pytest.mark.parametrize('problem',['node','workspace','photo','settings','heightmap','integer_setting','form_overflow'])
def test_broken_project_preserves_current_document(window, tmp_path, monkeypatch, problem):
    w=window; w.create_plate(); add_sweep(w)
    path=tmp_path/'broken.srelief'; assert w._write(str(path))
    before=w.state_sig(); original=w.z.copy(); notices=[]
    with zipfile.ZipFile(path) as archive:
        entries={name:archive.read(name) for name in archive.namelist()}
    meta=json.loads(entries['project.json'])
    if problem=='node': meta['vec'][0]['n'][0][2]=[1]
    elif problem=='workspace': meta['settings']['workspace']['sweep']['rails']=12
    elif problem=='settings': meta['settings']['w']=float('nan')
    elif problem=='integer_setting': meta['settings']['bright']=12.0
    elif problem=='form_overflow': meta['settings']['workspace']['form']['angle']=2**64
    elif problem=='photo':
        meta['photo_file']='photo.bin'; entries['photo.bin']=b'not an image'
    else:
        buffer=io.BytesIO(); np.savez_compressed(buffer,z=np.full((3,3),np.nan))
        entries['arrays.npz']=buffer.getvalue()
    entries['project.json']=json.dumps(meta).encode()
    with zipfile.ZipFile(path,'w') as archive:
        for name,data in entries.items(): archive.writestr(name,data)
    monkeypatch.setattr(QMessageBox,'critical',lambda *a,**k:notices.append(a[2]))
    assert w.open_project(str(path)) is False
    assert len(notices)==1
    assert w.state_sig()==before and np.array_equal(w.z,original)
    assert w.project_path==str(path)


def test_surface_picking_through_vtk_mouse_events(window, app):
    w=window; w.create_plate()
    w.sculpt_action.setChecked(True); w.toggle_sculpt(True)
    app.processEvents(); w.plotter.render()
    renderer=w.plotter.renderer
    renderer.SetWorldPoint(0,0,float(w.z[60,60]),1)
    renderer.WorldToDisplay()
    x,y,_=renderer.GetDisplayPoint()
    interactor=w.plotter.iren.interactor
    interactor.SetEventInformation(round(x),round(y))
    interactor.InvokeEvent('LeftButtonPressEvent')
    assert w._stroke is not None
    before=w.z.copy()
    interactor.SetEventInformation(round(x),round(y)+30)
    interactor.InvokeEvent('MouseMoveEvent')
    interactor.InvokeEvent('LeftButtonReleaseEvent')
    assert w._stroke is None
    assert w.z.max()>before.max()
    assert w.undo_stack[-1][0]=='Лепка поверхности'
    w.toggle_sculpt(False); w.sculpt_action.setChecked(False)


def test_f12_preview_apply_and_undo_with_hole(window):
    w=window; w.create_plate()
    for points in ([(8,8),(52,8),(52,52),(8,52)],[(24,24),(36,24),(36,36),(24,36)]):
        w._add_vec(core.polyline_path([(x*w.K,y*w.K) for x,y in points],True),True)
    initial=w.z.copy()
    w.shape_apply(True)
    assert w.preview_active
    assert w.preview_z.max()>initial.max()
    assert w.preview_z[60,60]==initial[60,60]
    assert w.z==pytest.approx(initial)
    w.shape_apply(False)
    assert w.z.max()>initial.max()
    w.undo()
    assert w.z==pytest.approx(initial)


def test_gui_export_uses_streaming_path(window, tmp_path, monkeypatch):
    w=window; w.create_plate()
    path=tmp_path/'gui.stl'; notices=[]
    monkeypatch.setattr(core.QFileDialog,'getSaveFileName',lambda *a,**k:(str(path),'STL'))
    monkeypatch.setattr(QMessageBox,'information',lambda *a,**k:notices.append(a[2]))
    def full_mesh_forbidden(*args,**kwargs):
        raise AssertionError('Full geometry allocation during STL export')
    monkeypatch.setattr(core,'build_mesh_arrays',full_mesh_forbidden)
    w.export_stl()
    assert path.exists()
    assert len(notices)==1 and '60 × 60 × 5 мм' in notices[0]
    import trimesh
    mesh=trimesh.load(path)
    assert mesh.is_watertight and mesh.is_winding_consistent


def test_gui_blocks_open_mesh_without_overwriting(window, tmp_path, monkeypatch):
    import trimesh
    w=window
    mesh=trimesh.creation.box(); mesh.update_faces(np.arange(len(mesh.faces)-1))
    w.stl_mesh=mesh; w.z=None
    path=tmp_path/'preserve.stl'; path.write_bytes(b'original file'); warnings=[]
    monkeypatch.setattr(core.QFileDialog,'getSaveFileName',lambda *a,**k:(str(path),'STL'))
    monkeypatch.setattr(QMessageBox,'warning',lambda *a,**k:warnings.append(a[2]))
    w.export_stl()
    assert len(warnings)==1 and 'незамкнута' in warnings[0]
    assert path.read_bytes()==b'original file'


def test_gui_export_can_be_cancelled(window, tmp_path, monkeypatch, app):
    from PySide6.QtCore import QTimer
    w=window; w.create_plate()
    path=tmp_path/'cancel.stl'; path.write_bytes(b'existing')
    monkeypatch.setattr(core.QFileDialog,'getSaveFileName',lambda *a,**k:(str(path),'STL'))
    def cancel():
        for dialog in app.topLevelWidgets():
            if isinstance(dialog,core.QProgressDialog): dialog.cancel()
    QTimer.singleShot(0,cancel)
    w.export_stl()
    assert path.read_bytes()==b'existing'
    assert 'отменён' in w.statusBar().currentMessage()


def test_history_budget_preserves_nearest_undo_and_redo(window):
    from stone_relief_studio_fixed import history_storage_bytes
    w=window; w.create_plate()
    w.history_max_bytes=w.z.nbytes*2
    for index in range(5):
        w.push_history(str(index)); w.z=w.z+1
    assert history_storage_bytes(w.undo_stack,w.redo_stack)<=w.history_max_bytes
    assert len(w.undo_stack)==2
    w.undo(); assert np.all(w.z==9)
    w.redo(); assert np.all(w.z==10)


def test_small_stl_dimensions_are_not_clipped_to_ten_mm(window, tmp_path):
    import trimesh
    w=window
    path=tmp_path/'small.stl'
    trimesh.creation.box(extents=(2.54,3.25,4.5)).export(path)
    w.open_stl(str(path))
    assert w.sp_w.value()==pytest.approx(2.54,abs=1e-6)
    assert w.sp_h.value()==pytest.approx(3.25,abs=1e-6)


def test_home_frames_active_view_without_changing_geometry(window, app):
    w=window; w.create_plate(); before=w.state_sig()
    w.tabs.setCurrentIndex(0); app.processEvents(); w.view.fit_all()
    fitted=w.view.transform().m11()
    w.view.scale(5,5); w.view.setFocus()
    QTest.keyClick(w.view,Qt.Key_Home)
    assert w.view.transform().m11()==pytest.approx(fitted)
    assert w.tabs.currentIndex()==0
    w.tabs.setCurrentIndex(1); app.processEvents()
    w.plotter.reset_camera(); distance=w.plotter.camera.distance
    w.plotter.camera.position=(1000,1000,1000)
    w._shell_actions['fit'].trigger()
    assert w.plotter.camera.distance==pytest.approx(distance,rel=.01)
    assert w.tabs.currentIndex()==1
    assert w.state_sig()==before


def test_fit_selection_and_dimensions_include_straight_lines(window, app):
    w=window; w.tabs.setCurrentIndex(0)
    for points in ([(10,10),(20,10)],[(40,20),(40,50)]):
        w._add_vec(core.polyline_path([(x*w.K,y*w.K) for x,y in points]),True)
    w.update_selection_ui(); before=w.state_sig(); w.view.fit_all()
    zoom=w.view.transform().m11()
    assert '30.00 × 40.00 мм' in w.selection_info.text()
    assert 'X 10.00; Y 10.00 мм' in w.selection_info.text()
    w.view.setFocus(); QTest.keyClick(w.view,Qt.Key_Home,Qt.ControlModifier)
    assert w.view.transform().m11()>zoom
    assert w.state_sig()==before
    w.scene.clearSelection()
    assert not w._shell_actions['fit_selection'].isEnabled()


def test_brush_nudge_and_snap_preferences_restore_without_dirtying_project(window, app):
    w=window; before=w.state_sig()
    w.sculpt_radius.setValue(23.75); w.sculpt_gain.setValue(.125)
    w.sculpt_fixed.setChecked(False); w.sculpt_protect_base.setChecked(False)
    w.nudge.setCurrentIndex(0)
    w.snap_actions['nodes'].setChecked(True); w.snap_actions['showgrid'].setChecked(True)
    assert w.state_sig()==before
    other=StoneStudio()
    try:
        assert other.sculpt_radius.value()==23.75 and other.sculpt_gain.value()==.125
        assert not other.sculpt_fixed.isChecked() and not other.sculpt_protect_base.isChecked()
        assert other.view.keyboard_nudge_mm==.01
        assert other.view.snap_nodes and other.view.show_grid
        assert not other.sculpt_action.isChecked()
    finally:
        other._saved_sig=other.state_sig(); other.close(); app.processEvents()


@pytest.mark.parametrize('payload',['broken json','{"radius": NaN, "gain": -1, "nudge": 99, "fixed": "false"}'])
def test_invalid_preferences_use_safe_defaults(window, app, payload):
    window.settings.setValue('userPreferences',payload)
    other=StoneStudio()
    try:
        assert other.sculpt_radius.value()==15 and other.sculpt_gain.value()==.05
        assert other.sculpt_fixed.isChecked() and other.sculpt_protect_base.isChecked()
        assert other.nudge.currentIndex()==1
    finally:
        other._saved_sig=other.state_sig(); other.close(); app.processEvents()


def test_recent_project_menu_opens_and_remembers_only_success(window, tmp_path, monkeypatch):
    w=window; paths=[tmp_path/'first.srelief',tmp_path/'second.srelief']
    for path in paths: assert w._write(str(path))
    assert w._recent_projects==[str(paths[1]),str(paths[0])]
    first=next(a for a in w.recent_menu.actions() if a.data()==str(paths[0]))
    first.trigger()
    assert w.project_path==str(paths[0])
    assert w._recent_projects==[str(paths[0]),str(paths[1])]
    before=w.state_sig(); notices=[]; paths[1].unlink()
    missing=next(a for a in w.recent_menu.actions() if a.data()==str(paths[1]))
    monkeypatch.setattr(QMessageBox,'critical',lambda *a,**k:notices.append(a[2]))
    missing.trigger()
    assert len(notices)==1 and w.state_sig()==before
    assert w.project_path==str(paths[0])
    assert w._recent_projects==[str(paths[0]),str(paths[1])]


def test_recent_list_is_limited_and_clearing_preserves_files(window, tmp_path):
    w=window
    paths=[tmp_path/f'project{i}.srelief' for i in range(8)]
    for path in paths: assert w._write(str(path))
    assert len(w._recent_projects)==6
    assert w._recent_projects[0]==str(paths[-1])
    w._clear_recent_projects()
    assert not w._recent_projects
    assert all(path.exists() for path in paths)
    assert not w.recent_menu.actions()[0].isEnabled()


def test_nonfinite_photo_reports_error_without_changing_relief(window, monkeypatch):
    w=window
    buffer=io.BytesIO()
    Image.fromarray(np.array([[0,np.nan],[1,2]],np.float32)).save(buffer,format='TIFF')
    w.photo_pil=Image.new('RGB',(2,2)); w.photo_bytes=buffer.getvalue()
    warnings=[]
    monkeypatch.setattr(QMessageBox,'warning',lambda *a,**k:warnings.append(a[2]))
    w.create_relief()
    assert w.z is None and len(warnings)==1
