"""Regression tests for exact F12 millimetres, dimensions and one-click rails."""
import numpy as np
import pytest
import trimesh
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QPushButton, QScrollArea, QMessageBox

import stone_relief_studio_fixed as core
from test_workspace import app, window


@pytest.mark.parametrize('height', [2, 7])
@pytest.mark.parametrize('angle', [5, 30, 45, 75, 90, -45])
@pytest.mark.parametrize('kind', ['round', 'angular', 'plateau'])
def test_exact_shape_height_ignores_legacy_multiplier(height, angle, kind):
    yy, xx = np.indices((61, 91))
    cov = (((xx-45)/34)**2 + ((yy-30)/24)**2 < 1).astype(float)
    shape = core.build_shape(cov, .5, .8, kind, angle, 'exact', height, 0, 5)
    assert np.abs(shape*cov).max() == pytest.approx(height, abs=1e-10)
    for mode in ('add', 'replace'):
        z = core.combine_relief(np.full(cov.shape, 5.), shape, cov, 5., mode)
        assert np.max(np.abs(z-5)) == pytest.approx(min(height,4.95) if angle<0 else height, abs=1e-10)


@pytest.mark.parametrize('mode', ['limit','scale','const'])
def test_explicit_mm_modes_do_not_multiply_height_a_second_time(mode):
    cov=np.zeros((81,81)); cov[5:76,5:76]=1
    shape=core.build_shape(cov,1,1,'angular',80,mode,7,start_h=3,scale=5)
    assert shape.max()==pytest.approx(10)
    assert np.max((shape-3)[cov>0])==pytest.approx(7)


@pytest.mark.parametrize('kind', ['round','angular','plateau'])
def test_exact_profile_angle_changes_shape_but_not_height(kind):
    cov=np.zeros((81,81)); cov[5:76,5:76]=1
    shallow=core.build_shape(cov,1,1,kind,15,'exact',7)
    steep=core.build_shape(cov,1,1,kind,80,'exact',7)
    assert shallow.max()==pytest.approx(7) and steep.max()==pytest.approx(7)
    assert np.max(np.abs(shallow-steep))>.1


def selected_rectangle(w):
    return w._add_vec(core.polyline_path([(10*w.K,10*w.K),(50*w.K,10*w.K),
                                          (50*w.K,50*w.K),(10*w.K,50*w.K)], True), True)


def mode_button(w, mode):
    return next(b for (_,key),b in zip(w.SH_MODES,w.sh_mode_btns) if key == mode)


def click_form_button(w, text, app):
    button = next(b for b in w.relief_dialog.findChildren(QPushButton) if b.text()==text)
    w.relief_dialog.findChild(QScrollArea).ensureWidgetVisible(button)
    app.processEvents()
    QTest.mouseClick(button, Qt.LeftButton)
    app.processEvents()


def test_f12_real_buttons_exact_add_replace_history_and_stl(window, app, tmp_path):
    w = window; w.sp_base.setValue(5); w.create_plate(); selected_rectangle(w)
    initial = w.z.copy(); xy = (w.sp_w.value(), w.sp_h.value())
    w.open_relief_editor(); app.processEvents()
    assert w.sh_height_mode()[0] == 'exact'
    w.sh_scale_sl.setValue(50)
    assert not w.sh_scale_sl.isEnabled()
    for height,expected in [(2,7), (7,14)]:
        w.sh_exact.setValue(height)
        click_form_button(w, 'Добавить', app)
        assert mode_button(w,'add').isChecked()
        assert 'Добавить' in w.sh_mode_label.text()
        click_form_button(w, 'Применить', app)
        assert w.z.max() == pytest.approx(expected, abs=1e-9)
        assert w.sh_last_report['shape_height'] == pytest.approx(height)
    accumulated = w.z.copy()
    w.sh_angle.setValue(75)
    click_form_button(w,'Заменить',app)
    click_form_button(w,'Применить',app)
    assert w.z.max() == pytest.approx(12)
    replaced = w.z.copy()
    w.undo(); assert w.z == pytest.approx(accumulated)
    w.redo(); assert w.z == pytest.approx(replaced)
    assert (w.sp_w.value(),w.sp_h.value()) == xy
    assert w.actor.GetScale() == (1,1,1)
    assert w.actor.GetBounds()[5] == pytest.approx(12)
    path = tmp_path/'exact.stl'
    core.export_heightmap_stl(path,w.z,*xy)
    mesh = trimesh.load(path)
    assert mesh.extents == pytest.approx((*xy,12), abs=1e-5)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert not np.array_equal(initial,w.z)


@pytest.mark.parametrize('mode', ['add','sub','high','low','replace','zero_in','zero_out'])
def test_f12_modes_apply_preview_and_undo(window, mode):
    w=window; w.create_plate(); selected_rectangle(w)
    w.z = w.z + 3
    before=w.z.copy(); cov=w.region_cov(); base=w.sp_base.value()
    w.sh_exact.setValue(2); mode_button(w,mode).setChecked(True)
    shape=core.build_shape(cov,*w.cell(),'round',45,'exact',2)
    if mode=='zero_out':
        expected=core.combine_relief(before,np.zeros_like(before),1-cov,base,'zero')
    else:
        expected=core.combine_relief(before,np.zeros_like(before) if mode=='zero_in' else shape,
                                     cov,base,'zero' if mode=='zero_in' else mode)
    w.shape_apply(True)
    assert w.z == pytest.approx(before)
    assert w.preview_z == pytest.approx(expected)
    w.shape_apply(False)
    assert w.z == pytest.approx(expected)
    w.undo(); assert w.z == pytest.approx(before)
    w.redo(); assert w.z == pytest.approx(expected)


def test_add_exceeding_height_requires_confirmation(window, monkeypatch):
    w=window; w.sp_base.setValue(5); w.create_plate(); selected_rectangle(w)
    w.sh_exact.setValue(7); w.sp_depth.setValue(2)
    before=w.z.copy(); history=len(w.undo_stack); questions=[]
    def reject(*args):
        questions.append(args); return QMessageBox.No
    monkeypatch.setattr(QMessageBox,'question',reject)
    w.shape_apply(True)
    assert not questions and 'ВНИМАНИЕ' in w.sh_report.text()
    w.shape_apply(False)
    assert len(questions)==1 and w.z==pytest.approx(before)
    assert len(w.undo_stack)==history
    monkeypatch.setattr(QMessageBox,'question',lambda *args:QMessageBox.Yes)
    w.shape_apply(False)
    assert w.z.max()==pytest.approx(12)


def test_dimensions_preserve_relief_vectors_controls_and_undo(window, tmp_path):
    w=window; w.sp_base.setValue(5); w.create_plate(); v=selected_rectangle(w)
    yy,xx=np.indices(w.z.shape); w.z=w.z+xx*.01+yy*.02
    w._irm_controls=[dict(id='p',x=30.,y=30.,z=6.,pinned=True)]
    before=w.z.copy(); original=np.array(v.scene_points())/w.K
    assert w.apply_model_dimensions(120,30,8,7,.5,confirm=False)
    assert w.z.shape == (61,241)
    assert w.z[0,0]==pytest.approx(before[0,0]+3)
    assert w.z[-1,-1]==pytest.approx(before[-1,-1]+3)
    assert np.array(v.scene_points())/w.K==pytest.approx(original*[2,.5])
    assert (w._irm_controls[0]['x'],w._irm_controls[0]['y'],w._irm_controls[0]['z'])==(60,15,9)
    core.export_heightmap_stl(tmp_path/'resized.stl',w.z,120,30)
    assert trimesh.load(tmp_path/'resized.stl').extents==pytest.approx((120,30,before.max()+3))
    w.undo()
    assert (w.sp_w.value(),w.sp_h.value(),w.sp_base.value(),w.sp_depth.value())==(60,60,5,10)
    assert w.z==pytest.approx(before)
    assert np.array(w.view.vectors()[0].scene_points())/w.K==pytest.approx(original)
    w.redo()
    assert w.z.shape==(61,241) and w.sp_depth.value()==7


def test_dimensions_rejection_keeps_project(window, monkeypatch):
    w=window; w.create_plate(); before=w.state_sig()
    monkeypatch.setattr(QMessageBox,'question',lambda *args:QMessageBox.No)
    assert not w.apply_model_dimensions(80,40,7,10,.2)
    assert w.state_sig()==before


def test_dimensions_button_and_new_project_dialog_cancel(window, app):
    w=window; w.create_plate(); before=w.state_sig(); seen=[]
    def cancel():
        seen.append(w.dimensions_dialog.windowTitle())
        assert all(f.suffix().strip()=='мм' for f in w.dimensions_fields)
        w.dimensions_dialog.reject()
    QTimer.singleShot(0,cancel)
    QTest.mouseClick(w.dimensions_button,Qt.LeftButton)
    assert seen==['РАЗМЕРЫ МОДЕЛИ'] and w.state_sig()==before
    QTimer.singleShot(0,cancel)
    assert not w.new_project()
    assert 'Новый проект' in seen[-1] and w.state_sig()==before


def test_new_project_dimensions_accepted(window, monkeypatch):
    w=window; w.create_plate()
    monkeypatch.setattr(QMessageBox,'question',lambda *args:QMessageBox.Discard)
    def accept():
        for field,value in zip(w.dimensions_fields,(120,200,5,7,.5)): field.setValue(value)
        w.dimensions_dialog.accept()
    QTimer.singleShot(0,accept)
    assert w.new_project()
    assert not w.has_content() and w.sp_w.value()==120 and w.sp_h.value()==200
    w.create_plate()
    assert w.z.max()==5 and w.z.shape==(401,241)


def test_single_click_picks_lines_beziers_sections_and_preserves_other_rail(window, app):
    w=window
    a=w._add_vec(core.polyline_path([(10*w.K,10*w.K),(10*w.K,50*w.K)]))
    b=w._add_vec(core.PathData([core.Node(40*w.K,10*w.K,hout=(30*w.K,20*w.K)),
                               core.Node(40*w.K,50*w.K,hin=(50*w.K,40*w.K))]))
    p=w._add_vec(core.polyline_path([(15*w.K,55*w.K),(35*w.K,55*w.K)]))
    w.open_sweep_dialog(); app.processEvents()
    for which,item in [('top',a),('bottom',b),('section',p)]:
        w.sw_begin_pick(which); w.view.fit_all(); app.processEvents()
        points=item.scene_points(); x,y=points[len(points)//2] if len(points)>2 else np.mean(points,axis=0)
        QTest.mouseClick(w.view.viewport(),Qt.LeftButton,pos=w.view.mapFromScene(QPointF(x,y)))
        app.processEvents()
        assert w._sw_pending_pick is None and w.sweep_dialog.isVisible()
    assert w.sw_rails==[a,b] and w.sw_sections==[p]
    assert len({item.pen().color().name() for item in (a,b,p)})==3
    before=[v.data.to_dict() for v in (a,b,p)]
    w.sw_begin_pick('top'); app.processEvents()
    x,y=b.scene_points()[len(b.scene_points())//2]
    QTest.mouseClick(w.view.viewport(),Qt.LeftButton,pos=w.view.mapFromScene(QPointF(x,y)))
    app.processEvents()
    assert w.sw_rails==[a,b] and w.sw_sections==[p]
    assert before==[v.data.to_dict() for v in (a,b,p)]


def test_single_click_miss_closed_and_escape_do_not_erase_rails(window, app):
    w=window
    a=w._add_vec(core.polyline_path([(5*w.K,5*w.K),(5*w.K,50*w.K)]))
    selected_rectangle(w); w.sw_rails[0]=a
    w.sw_begin_pick('bottom'); app.processEvents()
    for x,y in [(30,30),(10,30)]:
        QTest.mouseClick(w.view.viewport(),Qt.LeftButton,pos=w.view.mapFromScene(QPointF(x*w.K,y*w.K)))
        assert w._sw_pending_pick=='bottom' and w.sw_rails==[a,None]
    QTest.keyClick(w.view,Qt.Key_Escape); app.processEvents()
    assert w._sw_pending_pick is None and w.sweep_dialog.isVisible()
    assert w.sw_rails==[a,None]


def test_exact_parameters_and_resized_project_roundtrip(window, tmp_path):
    w=window; w.sp_base.setValue(5); w.create_plate(); selected_rectangle(w)
    w.sh_exact.setValue(7); mode_button(w,'replace').setChecked(True); w.shape_apply()
    w.apply_model_dimensions(80,40,5,8,.5,confirm=False)
    before=w.z.copy(); path=str(tmp_path/'exact.srelief')
    assert w._write(path)
    assert w.open_project(path)
    assert w.sh_height_mode()==('exact',7) and w.sh_mode_key()=='replace'
    assert (w.sp_w.value(),w.sp_h.value())==(80,40)
    assert w.z==pytest.approx(before)


def test_mesh_dimensions_keep_xyz_and_connectivity_with_undo(window):
    w=window
    w.stl_mesh=trimesh.creation.box(extents=(60,60,10))
    original=w.stl_mesh.vertices.copy(); faces=w.stl_mesh.faces.copy()
    w.show_mesh()
    assert w.apply_model_dimensions(120,30,w.sp_base.value(),10,.5,confirm=False)
    expected=original.copy()
    expected[:,:2]=-30+(original[:,:2]+30)*[2,.5]
    assert w.stl_mesh.vertices==pytest.approx(expected)
    assert np.array_equal(w.stl_mesh.faces,faces) and w.z is None
    assert w.actor.GetBounds()==pytest.approx((-30,90,-30,0,-5,5))
    w.undo(); assert w.stl_mesh.vertices==pytest.approx(original)
    w.redo(); assert w.stl_mesh.vertices==pytest.approx(expected)


def test_apply_cancels_queued_preview_without_extra_history(window, app):
    w=window; w.create_plate(); selected_rectangle(w)
    w.shape_apply(True); w.sh_angle.setValue(60)
    assert w.sh_timer.isActive()
    count=len(w.undo_stack)
    w.shape_apply(False); applied=w.z.copy()
    QTest.qWait(350)
    assert not w.preview_active and not w.sh_timer.isActive()
    assert w.z==pytest.approx(applied) and len(w.undo_stack)==count+1
