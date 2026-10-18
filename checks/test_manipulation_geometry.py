import io
import itertools
import json
import struct
import zipfile
import zlib
import numpy as np
import pytest
from PIL import Image
from stone_relief_studio_fixed import (deform_control_surface, deform_mesh_vertices,
    ManipulationCancelled, surface_samples)
from stone_relief_studio_fixed import coverage
from stone_relief_studio_fixed import pack_project, unpack_project, ProjectCancelled, FORMAT
from stone_relief_studio_fixed import open_checked_image


def point(x,y,z,pinned=False):
    return dict(id=f'{x}:{y}',x=x,y=y,z=z,pinned=pinned)


def test_full_surface_reproduces_tilt_without_erasing_detail():
    yy,xx=np.indices((41,41)); detail=((xx+yy)%2)*.035
    z=5+detail
    controls=[point(x,y,float(z[y+20,x+20]+1+.02*x+.03*y)) for x,y in [(-10,-10),(10,-10),(0,10)]]
    result=deform_control_surface(z,controls,40,40,fixed=False,mode='full')
    expected=z+1+.02*(xx-20)+.03*(yy-20)
    assert result==pytest.approx(expected,abs=2e-5)


@pytest.mark.parametrize('mode',['local','full'])
def test_control_constraints_and_pins_hold_edges_and_outside_mask(mode):
    z=np.full((41,41),5,dtype=np.float32); mask=np.zeros(z.shape,bool); mask[4:37,4:37]=True
    controls=[point(0,0,7),point(-5,0,5,True)]
    result=deform_control_surface(z,controls,40,40,mask=mask,fixed=True,mode=mode,radius=12)
    assert result[20,20]==pytest.approx(7,abs=1e-5)
    assert result[20,15]==pytest.approx(5,abs=1e-5)
    assert np.array_equal(result[~mask],z[~mask])
    assert np.array_equal(result[4,:],z[4,:])
    assert np.array_equal(result[:,4],z[:,4])


def test_local_control_has_exact_compact_support():
    z=np.full((41,41),5,dtype=np.float32)
    result=deform_control_surface(z,[point(0,0,7)],40,40,fixed=False,radius=5)
    yy,xx=np.indices(z.shape)
    outside=(xx-20)**2+(yy-20)**2>=25
    assert np.array_equal(result[outside],z[outside])
    assert result[20,20]==7


@pytest.mark.parametrize('controls',[ [point(0,0,6),dict(point(0,0,7),id='other')], [point(30,0,6)], [point(-20,0,7)], [] ])
def test_invalid_control_constraints_report_actionable_error(controls):
    with pytest.raises(ValueError): deform_control_surface(np.full((41,41),5),controls,40,40)


def test_mesh_soft_selection_keeps_pinned_vertices_exact():
    vertices=np.array([[0,0,0],[1,0,0],[2,0,1],[20,0,-3]],float)
    before=vertices.copy()
    result=deform_mesh_vertices(vertices,[0],[2,3,4],radius=5,pinned=[2])
    assert np.array_equal(vertices,before)
    assert result[0]==pytest.approx([2,3,4])
    assert np.all(result[1]>vertices[1])
    assert np.array_equal(result[[2,3]],vertices[[2,3]])


def test_mesh_xyz_preserves_overhanging_coordinates_without_rasterizing():
    vertices=np.array([[0,0,0],[0,0,7],[0,1,2],[0,1,9]],float)
    result=deform_mesh_vertices(vertices,[0,1],[1,2,3],radius=0)
    assert result[:2]==pytest.approx(vertices[:2]+[1,2,3])
    assert np.array_equal(result[2:],vertices[2:])
    assert result[1,2]-result[0,2]==7


def test_control_computation_can_be_cancelled():
    z=np.full((41,41),5)
    def cancel(*args): raise ManipulationCancelled()
    with pytest.raises(ManipulationCancelled): deform_control_surface(z,[point(0,0,6)],40,40,progress=cancel)
    assert np.all(z==5)


def test_overlap_does_not_erase_an_independent_region_inside_another_hole():
    outer=[(0,0),(30,0),(30,30),(0,30)]
    hole=[(5,5),(25,5),(25,25),(5,25)]
    crossing=[(20,10),(35,10),(35,20),(20,20)]
    for contours in itertools.permutations([outer,hole,crossing]):
        mask=coverage(contours,40,40)
        assert mask[15,23]==1
        assert mask[15,10]==0
        assert mask[2,2]==1


def test_duplicate_boundaries_do_not_change_nested_island_parity():
    outer=[(0,0),(30,0),(30,30),(0,30)]
    hole=[(5,5),(25,5),(25,25),(5,25)]
    island=[(10,10),(20,10),(20,20),(10,20)]
    assert np.array_equal(coverage([outer,hole,island],40,40),coverage([outer,hole,hole[::-1],island,outer],40,40))


def test_project_cancel_keeps_previous_file_and_cleans_temporary(tmp_path):
    path=tmp_path/'old.srelief'; path.write_bytes(b'original')
    def cancel(*args): raise ProjectCancelled()
    with pytest.raises(ProjectCancelled): pack_project(path,{},arrays={'z':np.ones((100,100))},progress=cancel)
    assert path.read_bytes()==b'original'
    assert list(tmp_path.iterdir())==[path]


def test_forged_huge_npy_header_is_rejected_before_allocation(tmp_path,monkeypatch):
    header=io.BytesIO(); np.lib.format.write_array_header_2_0(header,dict(descr='<f8',fortran_order=False,shape=(1_000_000_000,2)))
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w') as archive: archive.writestr('z.npy',header.getvalue())
    path=tmp_path/'forged.srelief'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('project.json',json.dumps(dict(format=FORMAT,version=4,arrays=['z'])))
        archive.writestr('arrays.npz',payload.getvalue())
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Untrusted shape reached allocation'))
    with pytest.raises(ValueError,match='памяти'): unpack_project(path)


def test_zip_expansion_budget_is_checked_before_read(tmp_path,monkeypatch):
    import stone_relief_studio_fixed as project_storage
    path=tmp_path/'bomb.srelief'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('project.json',b' '*100_000)
    monkeypatch.setattr(project_storage,'MAX_PROJECT_BYTES',2000)
    with pytest.raises(ValueError,match='распаковки'): unpack_project(path)


def test_oversized_png_dimensions_are_rejected_before_pixel_decode():
    stream=io.BytesIO(); Image.new('L',(1,1)).save(stream,format='PNG')
    data=bytearray(stream.getvalue()); data[16:24]=struct.pack('>II',10000,10000)
    data[29:33]=struct.pack('>I',zlib.crc32(data[12:29]))
    with pytest.raises(ValueError,match='разрешение'): open_checked_image(data)


def test_nonfinite_photo_preview_is_rejected_before_replacing_photo():
    from stone_relief_studio_fixed import load_image_any
    stream=io.BytesIO(); Image.fromarray(np.array([[0,np.nan],[1,2]],np.float32)).save(stream,format='TIFF')
    with pytest.raises(ValueError,match='NaN'): load_image_any(stream.getvalue())
