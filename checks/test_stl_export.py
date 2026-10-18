import struct

import numpy as np
import pytest
import trimesh

from stone_relief_studio_fixed import (ExportCancelled, export_heightmap_stl,
                        export_indexed_stl, validate_stl_mesh)


def test_streamed_heightfield_is_closed_oriented_and_sized(tmp_path):
    z = 5+np.arange(63, dtype=np.float32).reshape(7,9)/100
    path = tmp_path/'relief.stl'
    report = export_heightmap_stl(path, z, 80, 55)
    mesh = trimesh.load(path)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.extents == pytest.approx((80,55,z.max()))
    assert mesh.volume == pytest.approx(report.volume, rel=1e-6)
    assert len(mesh.faces) == report.triangles
    assert path.stat().st_size == 84+50*report.triangles


def test_regular_flat_model_uses_exact_millimetres(tmp_path):
    report = export_heightmap_stl(tmp_path/'flat.stl', np.full((2,2),5), 100, 70)
    assert report.triangles == 14
    assert report.volume == 35000


@pytest.mark.parametrize('edge_limit', [0, 200000])
@pytest.mark.parametrize('problem',['open','winding','inward','degenerate','nan','index'])
def test_bad_mesh_does_not_overwrite_existing_file(tmp_path, problem, edge_limit, monkeypatch):
    import stone_relief_studio_fixed as stl_export
    monkeypatch.setattr(stl_export, '_STL_RAM_FACE_LIMIT', edge_limit)
    mesh = trimesh.creation.box(extents=(5,7,9))
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    if problem == 'open': faces = faces[:-1]
    elif problem == 'winding': faces[0] = faces[0,::-1]
    elif problem == 'inward': faces = faces[:,::-1]
    elif problem == 'degenerate': faces[0,1] = faces[0,0]
    elif problem == 'nan': vertices[0,0] = np.nan
    elif problem == 'index': faces[0,0] = len(vertices)
    path = tmp_path/'preserve.stl'; path.write_bytes(b'previous model')
    with pytest.raises(ValueError):
        export_indexed_stl(path, vertices, faces)
    assert path.read_bytes() == b'previous model'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['preserve.stl']


@pytest.mark.parametrize('value',[0,-1,np.nan,np.inf])
def test_invalid_heights_are_blocked_before_opening_destination(tmp_path, value):
    z = np.ones((5,5)); z[2,2] = value
    with pytest.raises(ValueError): export_heightmap_stl(tmp_path/'bad.stl',z,10,10)
    assert not list(tmp_path.iterdir())


def test_cancellation_keeps_existing_stl_and_removes_tempfile(tmp_path):
    path = tmp_path/'original.stl'; path.write_bytes(b'old content')
    def cancel(phase, current, total):
        if phase == 'write': raise ExportCancelled()
    with pytest.raises(ExportCancelled):
        export_heightmap_stl(path,np.ones((50,70)),20,40,cancel)
    assert path.read_bytes() == b'old content'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['original.stl']


def test_write_error_does_not_replace_existing_model(tmp_path, monkeypatch):
    import stone_relief_studio_fixed as stl_export
    path = tmp_path/'model.stl'; path.write_bytes(b'existing')
    def fail(*args): raise OSError('disk write failed')
    monkeypatch.setattr(stl_export,'_stl_records',fail)
    with pytest.raises(OSError): export_heightmap_stl(path,np.ones((3,3)),10,20)
    assert path.read_bytes() == b'existing'
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize('edge_limit', [0, 200000])
def test_valid_indexed_mesh_is_exported_after_preflight(tmp_path, edge_limit, monkeypatch):
    import stone_relief_studio_fixed as stl_export
    monkeypatch.setattr(stl_export, '_STL_RAM_FACE_LIMIT', edge_limit)
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=8)
    report = export_indexed_stl(tmp_path/'sphere.stl',mesh.vertices,mesh.faces,chunk_size=7)
    output = trimesh.load(tmp_path/'sphere.stl')
    assert output.is_watertight and output.is_winding_consistent
    assert output.volume == pytest.approx(report.volume, rel=1e-6)


def test_progress_includes_check_before_write(tmp_path):
    phases = []
    export_heightmap_stl(tmp_path/'phases.stl',np.ones((100,20)),50,90,
                         lambda phase,*args:phases.append(phase))
    assert phases[0] == 'check'
    assert phases[-1] == 'write'
    assert phases.index('write') > phases.index('check')


def test_minimum_base_thickness_prevents_too_thin_export(tmp_path):
    z=np.full((5,5),5.); z[2,2]=.1
    path=tmp_path/'thin.stl'; path.write_bytes(b'previous')
    with pytest.raises(ValueError,match='Толщина'):
        export_heightmap_stl(path,z,20,20,minimum_thickness=.5)
    assert path.read_bytes()==b'previous'


def test_float32_error_is_rejected_before_export(tmp_path):
    mesh=trimesh.creation.box(extents=(10,12,14)); mesh.apply_translation((100000003,0,0))
    path=tmp_path/'precision.stl'; path.write_bytes(b'previous')
    with pytest.raises(ValueError,match='float32'):
        export_indexed_stl(path,mesh.vertices,mesh.faces,precision_tolerance=.001)
    assert path.read_bytes()==b'previous'


def test_legacy_stl_writer_also_preserves_existing_file_on_invalid_input(tmp_path):
    from stone_relief_studio_fixed import write_stl
    mesh=trimesh.creation.box(); path=tmp_path/'old.stl'; path.write_bytes(b'previous')
    with pytest.raises(ValueError): write_stl(path,mesh.vertices,mesh.faces[:-1])
    assert path.read_bytes()==b'previous'
