"""SVG geometry conversion. SVG and DXF coordinates are normalized to millimetres."""
import io

from svgelements import (SVG, Path, Shape, Move, Close, CubicBezier,
                         QuadraticBezier)


def convert_path(path, node_type, path_type, scale=1.0):
    path = Path(path)
    path.approximate_arcs_with_cubics(error=0.01)
    result, nodes = [], []

    def point(p):
        return (float(p.x) * scale, float(p.y) * scale)

    def flush(closed=False):
        nonlocal nodes
        if closed and len(nodes) > 1:
            first, last = nodes[0], nodes[-1]
            if abs(first.x-last.x) + abs(first.y-last.y) < 1e-8:
                first.hin = last.hin
                nodes.pop()
        if len(nodes) >= 2:
            result.append(path_type(nodes, closed))
        nodes = []

    for seg in path:
        if isinstance(seg, Move):
            flush()
            nodes = [node_type(*point(seg.end))]
        elif isinstance(seg, Close):
            flush(True)
        else:
            if not nodes:
                nodes = [node_type(*point(seg.start))]
            incoming = None
            if isinstance(seg, CubicBezier):
                nodes[-1].hout = point(seg.control1)
                incoming = point(seg.control2)
            elif isinstance(seg, QuadraticBezier):
                a, q, b = point(seg.start), point(seg.control), point(seg.end)
                nodes[-1].hout = tuple(a[i]+2*(q[i]-a[i])/3 for i in (0, 1))
                incoming = tuple(b[i]+2*(q[i]-b[i])/3 for i in (0, 1))
            nodes.append(node_type(*point(seg.end), hin=incoming))
    flush()
    return result


def parse_path(data, node_type, path_type):
    return convert_path(Path(data or ''), node_type, path_type)


def parse_svg(text, node_type, path_type):
    # svgelements applies inherited transforms, viewBox, units and <use>.
    svg = SVG.parse(io.StringIO(text), reify=True, ppi=96,
                    parse_display_none=False, on_error='raise')
    paths = []
    for element in svg.elements():
        if isinstance(element, Shape):
            paths.extend(convert_path(element, node_type, path_type, 25.4 / 96))
    width, height = float(svg.width) * 25.4/96, float(svg.height) * 25.4/96
    return paths, (0.0, 0.0, width, height)
