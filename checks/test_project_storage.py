import io
import json
import zipfile

import numpy as np
import pytest

from stone_relief_studio_fixed import FORMAT, VERSION, pack_project, unpack_project


def test_project_preserves_height_precision_and_mesh_indices(tmp_path):
    path = tmp_path/'relief.srelief'
    z = np.full((4, 5), 5.123456789012345, dtype=np.float64)
    pack_project(path, {'settings': {'w': 2.54, 'h': 4.25}}, arrays={'z': z})
    meta, _, data = unpack_project(path)
    assert np.array_equal(data['z'], z)
    assert meta['version'] == VERSION
    vertices = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float64)
    faces = np.array([[0,2,1], [0,1,3], [1,2,3], [2,0,3]], dtype=np.int64)
    pack_project(path, {}, arrays={'mesh_vertices': vertices, 'mesh_faces': faces})
    _, _, data = unpack_project(path)
    assert data['mesh_faces'].dtype == np.int64
    assert np.array_equal(data['mesh_faces'], faces)
    assert np.array_equal(data['mesh_vertices'], vertices)


@pytest.mark.parametrize('version', [2, 3, 4])
def test_old_project_versions_remain_readable(tmp_path, version):
    path = tmp_path/'old.srelief'
    buffer = io.BytesIO(); np.savez_compressed(buffer, z=np.full((3,4), 5, np.float32))
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('project.json', json.dumps(dict(format=FORMAT, version=version, arrays=['z'])))
        archive.writestr('arrays.npz', buffer.getvalue())
    meta, _, arrays = unpack_project(path)
    assert meta['version'] == version
    assert np.all(arrays['z'] == 5)


def test_project_write_failure_keeps_previous_file(tmp_path, monkeypatch):
    import stone_relief_studio_fixed as project_storage
    path = tmp_path/'preserve.srelief'; path.write_bytes(b'previous document')
    def fail(*args):
        raise OSError('disk replacement failed')
    monkeypatch.setattr(project_storage.os, 'replace', fail)
    with pytest.raises(OSError):
        pack_project(path, {}, arrays={'z': np.ones((2,2))})
    assert path.read_bytes() == b'previous document'
    assert [p.name for p in tmp_path.iterdir()] == ['preserve.srelief']


@pytest.mark.parametrize('arrays', [
    {'z': np.array([[1, np.nan], [1, 1]])},
    {'z': np.ones(4)},
    {'z': np.full((2,2), 1e100)},
    {'mesh_faces': np.zeros((4,3), dtype=np.int64)},
    {'mesh_vertices': np.zeros((3,3)), 'mesh_faces': np.array([[0,1,99]])},
    {'mesh_vertices': np.zeros((3,3)), 'mesh_faces': np.array([[0.,1.,2.]])},
])
def test_invalid_geometry_cannot_replace_project(tmp_path, arrays):
    path = tmp_path/'existing.srelief'; path.write_bytes(b'previous document')
    with pytest.raises(ValueError): pack_project(path, {}, arrays=arrays)
    assert path.read_bytes() == b'previous document'
