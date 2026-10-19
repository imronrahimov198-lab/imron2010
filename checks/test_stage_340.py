"""Stage P0: edit the existing photo relief, keep the picking panel visible."""
import io
import numpy as np
import pytest
import trimesh
from PIL import Image
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDockWidget, QPushButton, QMessageBox

import stone_relief_studio_fixed as core
from test_workspace import app, window
from test_urgent_331 import selected_rectangle, mode_button, click_form_button


def photo_relief(w,tmp_path):
    yy,xx=np.indices((121,121))
    pixels=np.clip(130+22*np.sin(xx*np.pi/3)+8*np.cos(yy*np.pi/4),0,255).astype(np.uint8)
    path=tmp_path/'veins.png'; Image.fromarray(pixels).save(path)
    w.sp_base.setValue(5); w.sp_depth.setValue(10); w.sp_blur.setValue(0)
    assert w.open_photo(str(path))
    w.create_relief(); w.sp_depth.setValue(100)
    selected_rectangle(w)
    return pixels


@pytest.mark.parametrize('outside',[False,True])
def test_zero_photo_relief_exact_base_and_other_cells_bit_identical(window,tmp_path,outside):
    w=window; photo_relief(w,tmp_path); before=w.z.copy()
    coverage=w.region_cov(); mask=coverage>.5
    assert np.any((coverage>0)&(coverage<1))
    if outside: mask=~mask
    mode_button(w,'zero_out' if outside else 'zero_in').setChecked(True)
    w.shape_apply(True)
    assert np.array_equal(w.z,before)
    assert np.all(w.preview_z[mask]==5)
    assert np.array_equal(w.preview_z[~mask],before[~mask])
    w.shape_apply(); after=w.z.copy()
    assert np.all(after[mask]==5)
    assert np.array_equal(after[~mask],before[~mask])
    w.undo(); assert np.array_equal(w.z,before)
    w.redo(); assert np.array_equal(w.z,after)


@pytest.mark.parametrize('mode,sign',[('raise',1),('lower',-1)])
@pytest.mark.parametrize('amount',[.125,5,7])
def test_raise_lower_photo_keeps_texture_and_outside(window,tmp_path,mode,sign,amount):
    w=window; photo_relief(w,tmp_path); before=w.z.copy(); cov=w.region_cov()
    w.sh_delta.setValue(amount); mode_button(w,mode).setChecked(True)
    w.shape_apply(True)
    assert np.array_equal(w.z,before)
    expected=before.astype(np.float64)+sign*amount*cov.astype(np.float64)
    assert w.preview_z==pytest.approx(expected,abs=1e-12)
    w.shape_apply(); after=w.z.copy()
    assert after==pytest.approx(expected,abs=1e-12)
    assert np.array_equal(after[cov==0],before[cov==0])
    # Differences between neighbouring full-mask samples retain every vein.
    assert np.diff(after[40:80,40:80],axis=1)==pytest.approx(np.diff(before[40:80,40:80].astype(np.float64),axis=1),abs=1e-12)
    assert 'Z min' in w.sh_report.text() and 'Z max' in w.sh_report.text()
    w.undo(); assert np.array_equal(w.z,before)
    w.redo(); assert np.array_equal(w.z,after)


@pytest.mark.parametrize('mode',['tilt','convex','concave'])
def test_macro_deformation_preserves_existing_photo_relief(window,tmp_path,mode):
    w=window; photo_relief(w,tmp_path); before=w.z.copy(); cov=w.region_cov()
    w.sh_exact.setValue(7); w.sh_delta.setValue(4); w.sh_direction.setValue(90)
    mode_button(w,mode).setChecked(True)
    w.shape_apply(True)
    assert np.array_equal(w.z,before)
    assert np.array_equal(w.preview_z[cov==0],before[cov==0])
    delta=w.preview_z-before
    if mode=='tilt':
        assert delta[40,60]<delta[80,60]
        assert np.max(np.abs(np.diff(delta[35:85,60],n=2)))<1e-10
    else:
        assert np.max(np.abs(delta))==pytest.approx(7)
        assert np.sign(delta[60,60])==(-1 if mode=='concave' else 1)
    w.shape_apply(); applied=w.z.copy()
    w.undo(); assert np.array_equal(w.z,before)
    w.redo(); assert np.array_equal(w.z,applied)


def test_replace_with_details_preserves_veins_instead_of_flattening(window,tmp_path):
    w=window; photo_relief(w,tmp_path); before=w.z.copy()
    mode_button(w,'replace').setChecked(True)
    w.sh_r_const.setChecked(True); w.sh_const.setValue(3)
    w.sh_preserve.setChecked(False); w.shape_apply(True)
    assert np.std(w.preview_z[40:80,40:80])<1e-12
    w.sh_preserve.setChecked(True); w.sh_detail_radius.setValue(2)
    w.shape_apply(True)
    assert np.array_equal(w.z,before)
    assert np.std(w.preview_z[40:80,40:80])>.4
    correlation=np.corrcoef(before[40:80,40:80].ravel(),w.preview_z[40:80,40:80].ravel())[0,1]
    assert correlation>.99
    w.shape_apply(); after=w.z.copy()
    w.undo(); assert np.array_equal(w.z,before)
    w.redo(); assert np.array_equal(w.z,after)


def test_add_and_apply_and_cancel_buttons_actually_work(window,app):
    w=window; w.create_plate(); selected_rectangle(w); before=w.z.copy()
    w.open_relief_editor(); app.processEvents()
    w.sh_exact.setValue(2); mode_button(w,'sub').setChecked(True)
    count=len(w.undo_stack)
    QTest.mouseClick(w.sh_add_apply_button,Qt.LeftButton); app.processEvents()
    assert w.sh_mode_key()=='add' and w.z.max()==pytest.approx(7)
    assert len(w.undo_stack)==count+1
    w.shape_apply(True)
    assert w.preview_active
    button=next(b for b in w.relief_dialog.findChildren(QPushButton) if b.text()=='Отмена')
    QTest.mouseClick(button,Qt.LeftButton); app.processEvents()
    assert not w.preview_active and w.z.max()==pytest.approx(7)
    assert w.sh_add_apply_button.isVisible()
    assert not w.relief_dialog.findChild(core.QScrollArea).isAncestorOf(w.sh_mode_label)
    assert not w.relief_dialog.findChild(core.QScrollArea).isAncestorOf(w.sh_report)
    w.undo(); assert np.array_equal(w.z,before)


def test_workspace_resize_preserves_mm_photo_and_history(window,tmp_path):
    w=window; photo_relief(w,tmp_path); before=w.z.copy()
    def world_points():
        points=np.asarray(w.view.vectors()[0].scene_points())
        return np.column_stack((points[:,0]/w.K,w.sp_h.value()-points[:,1]/w.K))
    points=world_points(); photo=w.photo_values(121,121)
    assert w.apply_model_dimensions(120,90,5,100,.5,confirm=False,resize_mode='workspace')
    assert w.z.shape==(181,241)
    assert w.z[:121,:121]==pytest.approx(before,abs=1e-12)
    assert np.all(w.z[121:,:]==5) and np.all(w.z[:,121:]==5)
    assert world_points()==pytest.approx(points)
    assert w.photo_frame==[0,0,60,60]
    assert w.photo_values(241,181)[:121,:121]==pytest.approx(photo,abs=1e-12)
    assert w.view.photo_item.sceneBoundingRect().width()/w.K==pytest.approx(60,abs=.1)
    assert '120 × 90' in w.model_status.text()
    path=str(tmp_path/'workspace.srelief')
    assert w._write(path)
    assert w.open_project(path)
    assert w.photo_frame==[0,0,60,60] and world_points()==pytest.approx(points)


def test_workspace_undo_restores_photo_placement_and_exact_surface(window,tmp_path):
    w=window; photo_relief(w,tmp_path); before=w.z.copy()
    original_photo=w.photo_values(121,121)
    w.apply_model_dimensions(40,30,5,100,.5,confirm=False,resize_mode='workspace')
    assert w.z==pytest.approx(before[:61,:81],abs=1e-12)
    assert w.photo_values(81,61)==pytest.approx(original_photo[:61,:81],abs=1e-12)
    w.undo()
    assert np.array_equal(w.z,before) and w.photo_frame is None
    assert (w.sp_w.value(),w.sp_h.value())==(60,60)
    w.redo()
    assert w.photo_frame==[0,0,60,60] and w.z.shape==(61,81)


def test_dimensions_show_resolution_memory_and_persistent_status(window):
    w=window; seen=[]
    def inspect():
        for f,value in zip(w.dimensions_fields,(1200,600,20,50,2)): f.setValue(value)
        text=w.dimensions_grid_info.text(); seen.append(text)
        assert '601 × 301' in text and 'МиБ' in text
        assert w.dimensions_resize_mode.count()==2
        w.dimensions_dialog.accept()
    QTimer.singleShot(0,inspect)
    assert w.model_dimensions_dialog()
    assert seen and '1200 × 600' in w.model_status.text()
    w.statusBar().showMessage('Любое другое сообщение')
    assert '1200 × 600' in w.model_status.text()


def test_large_mm_project_exact_relief_and_stl(window,tmp_path):
    w=window; w.apply_model_dimensions(1200,600,20,50,2,confirm=False)
    w.create_plate()
    w._add_vec(core.polyline_path([(100*w.K,100*w.K),(1100*w.K,100*w.K),
                                   (1100*w.K,500*w.K),(100*w.K,500*w.K)],True),True)
    w.sh_exact.setValue(50); w.shape_apply()
    assert w.z.max()==pytest.approx(70) and w.z.min()==20
    path=tmp_path/'millimetres.stl'
    report=core.export_heightmap_stl(path,w.z,1200,600)
    mesh=trimesh.load(path)
    assert report.dimensions==pytest.approx((1200,600,70))
    assert mesh.extents==pytest.approx((1200,600,70),abs=1e-5)
    assert mesh.is_watertight and mesh.is_winding_consistent


def test_docked_sweep_stays_visible_during_real_bezier_picks(window,app):
    w=window; w.create_plate()
    rails=[]
    for x in (15,45):
        rails.append(w._add_vec(core.PathData([core.Node(x*w.K,10*w.K,hout=((x-3)*w.K,20*w.K)),
                  core.Node(x*w.K,50*w.K,hin=((x+3)*w.K,40*w.K))])))
    profiles=[]
    for y in (53,56,59):
        profiles.append(w._add_vec(core.PathData([core.Node(10*w.K,y*w.K,hout=(20*w.K,(y-5)*w.K)),
                             core.Node(40*w.K,y*w.K,hin=(30*w.K,(y-5)*w.K))])))
    w.open_sweep_dialog(); app.processEvents()
    assert isinstance(w.sweep_dialog,QDockWidget) and not w.sweep_dialog.isFloating()
    for role,item in [('top',rails[0]),('bottom',rails[1])]+[('section',p) for p in profiles]:
        w.sw_begin_pick(role); app.processEvents()
        assert w.sweep_dialog.isVisible()
        points=item.scene_points(); x,y=points[len(points)//2]
        QTest.mouseClick(w.view.viewport(),Qt.LeftButton,pos=w.view.mapFromScene(QPointF(x,y)))
        app.processEvents()
        assert w._sw_pending_pick is None and w.sweep_dialog.isVisible()
    assert w.sw_rails==rails and w.sw_sections==profiles
    assert rails[0].data.uid[:8] in w.sw_pick_buttons['top'].text()
    assert rails[1].data.uid[:8] in w.sw_pick_buttons['bottom'].text()
    before=w.z.copy(); w.sw_run(True)
    assert np.array_equal(w.z,before) and w.preview_z.max()>before.max()
    w.sw_run(False); applied=w.z.copy()
    w.undo(); assert np.array_equal(w.z,before)
    w.redo(); assert np.array_equal(w.z,applied)


def test_new_f12_options_roundtrip_and_history(window,tmp_path):
    w=window; photo_relief(w,tmp_path)
    w.sh_delta.setValue(5); w.sh_direction.setValue(-30)
    w.sh_preserve.setChecked(True); w.sh_detail_radius.setValue(1.25)
    mode_button(w,'raise').setChecked(True); w.shape_apply()
    before=w.z.copy(); path=str(tmp_path/'photo-edited.srelief')
    assert w._write(path) and w.open_project(path)
    assert w.sh_delta.value()==5 and w.sh_direction.value()==-30
    assert w.sh_preserve.isChecked() and w.sh_detail_radius.value()==1.25
    assert w.sh_mode_key()=='raise' and np.array_equal(w.z,before)


@pytest.mark.parametrize('height_mode',['exact','scale','limit','const'])
def test_photo_source_obeys_all_explicit_mm_modes(window,tmp_path,height_mode):
    w=window; photo_relief(w,tmp_path)
    w.sh_source.setCurrentIndex(1); w.sp_depth.setValue(100)
    radio,field={'exact':(w.sh_r_exact,w.sh_exact),'scale':(w.sh_r_scale,w.sh_scaleh),
                 'limit':(w.sh_r_limit,w.sh_limit),'const':(w.sh_r_const,w.sh_const)}[height_mode]
    radio.setChecked(True); field.setValue(7); w.sh_scale_sl.setValue(50)
    mode_button(w,'replace').setChecked(True)
    before=w.z.copy(); w.shape_apply(True)
    assert w.preview_z.max()==pytest.approx(12,abs=1e-10)
    assert np.array_equal(w.z,before)
    w.shape_apply()
    assert w.z.max()==pytest.approx(12,abs=1e-10)


@pytest.mark.parametrize('frame',[[0,0,0,60],[0,0,np.nan,60],[0,0,60],{'width':60}])
def test_invalid_photo_placement_cannot_replace_existing_project(tmp_path,frame):
    path=tmp_path/'safe.srelief'; path.write_bytes(b'previous project')
    with pytest.raises(ValueError):
        core.pack_project(path,{'settings':{'workspace':{'photo_frame':frame}}},arrays={'z':np.full((3,3),5.)})
    assert path.read_bytes()==b'previous project'
