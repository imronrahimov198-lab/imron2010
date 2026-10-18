import time
import numpy as np
import pytest
import trimesh
from PySide6.QtTest import QTest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox
from test_workspace import app, window
from test_manipulation_geometry import point
import stone_relief_studio_fixed as core


def wait_preview(w,app):
    deadline=time.monotonic()+12
    while w._irm_job is not None and w._irm_job.isRunning() and time.monotonic()<deadline:
        QTest.qWait(20); app.processEvents()
    app.processEvents(); QTest.qWait(30); app.processEvents()
    assert w._irm_result is not None, w.irm_status.text()


def test_real_vtk_control_drag_previews_then_applies_with_one_undo(window,app):
    w=window; w.create_plate(); original=w.z.copy()
    w.irm_enable_button.setChecked(True); app.processEvents(); w.plotter.render()
    renderer=w.plotter.renderer; renderer.SetWorldPoint(0,0,5,1); renderer.WorldToDisplay()
    x,y,_=renderer.GetDisplayPoint(); interactor=w.plotter.iren.interactor
    interactor.SetEventInformation(round(x),round(y)); interactor.InvokeEvent('LeftButtonPressEvent')
    assert len(w._irm_controls)==1
    interactor.SetEventInformation(round(x),round(y)+20); interactor.InvokeEvent('MouseMoveEvent'); interactor.InvokeEvent('LeftButtonReleaseEvent')
    assert w._irm_controls[0]['z']>5.9
    assert np.array_equal(w.z,original)
    w._irm_start_preview(); wait_preview(w,app)
    assert np.array_equal(w.z,original)
    assert w._irm_result[1].max()>5.8
    assert w._irm_apply()
    applied=w.z.copy(); w.undo(); assert np.array_equal(w.z,original)
    w.redo(); assert np.array_equal(w.z,applied)


def test_existing_sculpt_and_manipulation_modes_are_mutually_exclusive(window):
    w=window; w.create_plate()
    w.irm_enable_button.setChecked(True); assert w._irm_enabled
    w.sculpt_action.setChecked(True); w.toggle_sculpt(True)
    assert not w._irm_enabled and not w.irm_enable_button.isChecked()
    w.irm_enable_button.setChecked(True)
    assert not w.sculpt_action.isChecked()
    w.irm_enable_button.setChecked(False)


def test_numeric_control_height_pin_and_project_roundtrip(window,tmp_path,app):
    w=window; w.create_plate(); w._irm_controls=[point(0,0,5)]; w._irm_refresh_list(0)
    w.irm_z.setValue(7); w._irm_numeric_point()
    w._irm_timer.stop(); w._irm_start_preview(); wait_preview(w,app)
    assert w._irm_apply()
    w._irm_pin(); w._irm_timer.stop()
    assert w._irm_controls[0]['pinned'] and not w.irm_z.isEnabled()
    controls=w._irm_state()
    path=tmp_path/'controls.srelief'; assert w._write(str(path))
    assert w.open_project(str(path))
    assert w._irm_state()==controls
    assert w.z.max()>6.9
    assert w._saved_sig==w.state_sig()


def test_stale_preview_cannot_overwrite_a_newer_model(window,app):
    w=window; w.create_plate(); w._irm_controls=[point(0,0,7)]
    w._irm_start_preview(); w.z=w.z+1
    deadline=time.monotonic()+10
    while w._irm_job.isRunning() and time.monotonic()<deadline: QTest.qWait(20)
    app.processEvents(); QTest.qWait(30)
    assert w._irm_result is None and not w.irm_apply_button.isEnabled()
    assert np.all(w.z==6)


def test_mesh_xyz_preview_preserves_faces_overhangs_and_undo_actor(window,tmp_path,app):
    w=window
    mesh=trimesh.creation.box(extents=(20,20,8)); mesh.apply_translation((2,3,10))
    path=tmp_path/'mesh.stl'; mesh.export(path)
    assert w.open_stl_mesh(str(path))
    before=w.stl_mesh; vertices=before.vertices.copy(); faces=before.faces.copy()
    assert w.z is None
    assert before.bounds==pytest.approx(mesh.bounds)
    w._irm_selected={0,1}; w._irm_pins={1}; w.irm_soft.setChecked(False)
    w.irm_dx.setValue(2); w.irm_dy.setValue(3); w.irm_dz.setValue(4)
    w._irm_start_preview(); wait_preview(w,app)
    assert w.stl_mesh is before
    assert w._irm_apply()
    assert np.array_equal(w.stl_mesh.faces,faces)
    assert w.stl_mesh.vertices[0]==pytest.approx(vertices[0]+[2,3,4])
    assert np.array_equal(w.stl_mesh.vertices[1:],vertices[1:])
    w.undo(); assert np.array_equal(w.stl_mesh.vertices,vertices)
    actor_points=np.asarray(w.actor.mapper.dataset.points)
    assert actor_points.min(axis=0)==pytest.approx(vertices.min(axis=0))
    w.redo(); assert w.stl_mesh.vertices[0]==pytest.approx(vertices[0]+[2,3,4])
    project=tmp_path/'mesh.srelief'; assert w._write(str(project)); assert w.open_project(str(project))
    assert w.z is None and w._irm_pins=={1}
    assert np.array_equal(w.stl_mesh.faces,faces)


def test_real_mesh_face_selection_and_xyz_drag(window,app):
    w=window; w.create_plate(); assert w.relief_to_mesh()
    w.irm_selection_kind.setCurrentIndex(2); w.irm_soft.setChecked(False)
    w.irm_enable_button.setChecked(True); app.processEvents(); w.plotter.render()
    renderer=w.plotter.renderer; renderer.SetWorldPoint(0,0,5,1); renderer.WorldToDisplay()
    x,y,_=renderer.GetDisplayPoint(); iren=w.plotter.iren.interactor
    iren.SetEventInformation(round(x),round(y)); iren.InvokeEvent('LeftButtonPressEvent')
    assert len(w._irm_selected)==3
    iren.SetEventInformation(round(x),round(y)+20); iren.InvokeEvent('MouseMoveEvent'); iren.InvokeEvent('LeftButtonReleaseEvent')
    assert w.irm_dz.value()==pytest.approx(1)
    w._irm_start_preview(); wait_preview(w,app); assert w._irm_apply()


def test_cancelling_project_save_preserves_old_file_and_dirty_state(window,tmp_path,monkeypatch,app):
    w=window; w.create_plate(); path=tmp_path/'preserve.srelief'; assert w._write(str(path))
    previous=path.read_bytes(); w.z=w.z+1
    def cancel():
        for widget in app.topLevelWidgets():
            if isinstance(widget,core.QProgressDialog): widget.cancel()
    QTimer.singleShot(0,cancel)
    assert not w._write(str(path))
    assert path.read_bytes()==previous
    assert w.state_sig()!=w._saved_sig


def test_cancel_preview_and_relief_mesh_conversion_restore_original(window,app):
    w=window; w.create_plate(); original=w.z.copy(); w._irm_controls=[point(0,0,7)]
    w._irm_start_preview(); wait_preview(w,app); w._irm_cancel()
    assert w._irm_result is None and np.array_equal(w.z,original)
    assert w.relief_to_mesh() and w.z is None
    w.undo(); assert np.array_equal(w.z,original) and w.stl_mesh is None


def test_undo_finishes_an_active_control_drag_without_losing_history(window):
    w=window; w.create_plate(); w._irm_controls=[point(0,0,5)]
    snap=w.snapshot(); w._irm_controls[0]['z']=7
    w._irm_drag=dict(row=0,y=0,z=5,snapshot=snap)
    w.undo()
    assert w._irm_drag is None and w._irm_controls[0]['z']==5
    w.redo(); assert w._irm_controls[0]['z']==7


def test_save_during_control_drag_captures_final_point_height(window,tmp_path):
    w=window; w.create_plate(); w._irm_controls=[point(0,0,5)]
    snap=w.snapshot(); w._irm_controls[0]['z']=7
    w._irm_drag=dict(row=0,y=0,z=5,snapshot=snap)
    path=tmp_path/'drag.srelief'; assert w._write(str(path))
    assert w._irm_drag is None
    meta,_,_=core.unpack_project(path)
    assert meta['settings']['interactive']['controls'][0]['z']==7


def test_relief_mask_binding_survives_reopening_without_active_selection(window,tmp_path,app):
    w=window; w.create_plate()
    w._add_vec(core.polyline_path([(10*w.K,10*w.K),(50*w.K,10*w.K),(50*w.K,50*w.K),(10*w.K,50*w.K)],True),True)
    w._irm_controls=[point(0,0,7)]; w.irm_fixed.setChecked(False); w.irm_surface_mode.setCurrentIndex(1)
    path=tmp_path/'mask.srelief'; assert w._write(str(path)); assert w.open_project(str(path))
    assert not w.sel_vectors() and w._irm_mask_uids
    w._irm_start_preview(); wait_preview(w,app)
    result=w._irm_result[1]
    assert result[60,60]==pytest.approx(7,abs=1e-5)
    assert result[60,10]==5


def test_export_precision_and_thickness_are_saved_in_project(window,tmp_path):
    w=window; w.create_plate(); w.stl_min_wall.setValue(.8); w.stl_precision.setValue(.002)
    path=tmp_path/'export-options.srelief'; assert w._write(str(path))
    w._clear_all(); w.stl_min_wall.setValue(.5); w.stl_precision.setValue(.001)
    assert w.open_project(str(path))
    assert w.stl_min_wall.value()==.8 and w.stl_precision.value()==.002


def test_book_button_opens_russian_help_with_real_generated_diagrams(window,app):
    w=window; seen=[]
    def close_book():
        for dialog in app.topLevelWidgets():
            if isinstance(dialog,core.QDialog) and dialog.windowTitle().startswith('Книга'):
                browser=dialog.findChild(core.QTextBrowser)
                seen.append(browser.toPlainText()); dialog.accept()
    assert w.help_button.isVisible()
    QTimer.singleShot(30,close_book)
    w.help_button.click()
    assert seen and 'Interactive Relief Manipulation' in seen[0]
    assert w.book_data['interactive'] and w.book_data['mesh_edit']


def test_display_rounding_does_not_edit_a_control_point_on_focus_loss(window):
    w=window; w.create_plate(); w._irm_controls=[point(0,0,5.123456789)]; w._irm_refresh_list(0)
    count=len(w.undo_stack); w._irm_numeric_point()
    assert w._irm_controls[0]['z']==5.123456789
    assert len(w.undo_stack)==count
