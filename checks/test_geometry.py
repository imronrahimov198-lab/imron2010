import numpy as np
import pytest
import trimesh

from stone_relief_studio_fixed import (parse_svg_path, svg_to_paths, svg_document, polyline_path,
                         raster_coverage, two_rail_param, build_mesh_arrays, grid_from_step)
from stone_relief_studio_fixed import sculpt_weight, photo_heights


def test_svg_arc_is_a_curve():
    path = parse_svg_path('M0 0 A10 10 0 0 1 20 0')[0]
    points = np.array(path.flatten(32))
    assert np.max(np.abs(points[:, 1])) == pytest.approx(10, abs=.03)
    assert points[-1] == pytest.approx([20, 0])


def test_svg_smooth_quadratic_reflects_previous_control():
    path = parse_svg_path('M0 0 Q10 10 20 0 T40 0')[0]
    assert path.seg_points(1, 2)[1] == pytest.approx((30, -5))


def test_svg_transforms_and_physical_units():
    text = '<svg xmlns="http://www.w3.org/2000/svg" width="50mm" height="30mm" viewBox="0 0 50 30"><g transform="translate(20,10)"><path d="M0 0 L10 0"/></g></svg>'
    paths, bounds = svg_to_paths(text)
    assert bounds[2:] == pytest.approx((50, 30), abs=1e-4)
    assert (paths[0].nodes[0].x, paths[0].nodes[0].y) == pytest.approx((20, 10), abs=1e-4)
    assert paths[0].nodes[-1].x == pytest.approx(30, abs=1e-4)


def test_svg_roundtrip_keeps_millimetres():
    paths = [polyline_path([(10, 20), (40, 20), (40, 50)], True)]
    loaded, _ = svg_to_paths(svg_document(paths, 100, 100, 1))
    assert np.array(loaded[0].flatten()) == pytest.approx(np.array(paths[0].flatten()), abs=1e-4)


def test_nested_contours_make_holes_and_overlaps_make_union():
    outer = [(0, 0), (20, 0), (20, 20), (0, 20)]
    inner = [(5, 5), (15, 5), (15, 15), (5, 15)]
    mask = raster_coverage([outer, inner], 21, 21)
    assert mask[10, 10] == 0
    assert mask[2, 2] == 1
    overlap = [(15, 0), (25, 0), (25, 20), (15, 20)]
    assert raster_coverage([outer, overlap], 30, 25)[10, 17] == 1


def test_self_intersecting_region_reports_error():
    with pytest.raises(ValueError, match='пересекает'):
        raster_coverage([[(0, 0), (10, 10), (0, 10), (10, 0)]], 11, 11)


def test_rails_have_continuous_longitudinal_coordinates():
    mask = np.ones((1001, 21), bool)
    v, width, s = two_rail_param(mask, .1, .1, [(0, 0), (0, 1000)], [(20, 0), (20, 1000)])
    expected = np.arange(1001)*.001
    assert s.reshape(mask.shape)[:, 10] == pytest.approx(expected, abs=1e-9)
    assert width == pytest.approx(np.full(mask.sum(), 2))
    assert v.reshape(mask.shape)[:, 10] == pytest.approx(np.full(1001, .5))


def test_sculpt_has_local_support_and_exactly_pins_boundary():
    mask = np.zeros((41, 41), bool)
    mask[5:36, 5:36] = True
    weight = sculpt_weight(mask.shape, (20, 20), (1, 1), 25, mask, True)
    assert weight[20, 20] == 1
    assert not weight[~mask].any()
    assert not weight[5, :].any()
    assert not weight[:, 5].any()
    local = sculpt_weight(mask.shape, (20, 20), (1, 1), 5)
    assert local[20, 26] == 0
    # Adding a broad deformation does not filter the pre-existing fine pattern.
    detail = np.indices(mask.shape).sum(axis=0) % 2 * .05
    result = detail + 2*weight
    assert result - 2*weight == pytest.approx(detail, abs=1e-12)


def test_export_mesh_is_closed_with_consistent_normals():
    z = 5 + np.sin(np.linspace(0, np.pi, 21))[None, :]*np.ones((21, 1))
    vertices, faces = build_mesh_arrays(z, 100, 70)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0
    assert mesh.extents[:2] == pytest.approx((100, 70))


def test_16bit_heightmap_preserves_adjacent_levels():
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.array([[0, 32768], [32769, 65535]], dtype=np.uint16)).save(buffer, format='TIFF')
    height = np.flipud(photo_heights(buffer.getvalue(), 2, 2))
    assert height[1, 0]-height[0, 1] == pytest.approx(1/65535, abs=1e-7)
    assert height[0, 0]==0 and height[1, 1]==1


@pytest.mark.parametrize('width,height',[(3000,3000),(3000,10),(1e6,.01),(.01,1e6)])
def test_grid_memory_cap_is_strict_for_large_and_thin_models(width,height):
    nx,ny,used=grid_from_step(width,height,.001)
    assert nx>=2 and ny>=2
    assert nx*ny<=8000000
    assert used>=.001
