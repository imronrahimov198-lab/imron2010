#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stone Relief Studio — редактор векторов и рельефов, версия 3.0.

Все части программы объединены в этот файл. Другие файлы исходного кода не нужны.
Запуск: python stone_relief_studio_fixed.py
Установка библиотек:
python -m pip install PySide6 numpy Pillow scipy trimesh pyvista pyvistaqt qtpy ezdxf rtree svgelements shapely "vtk>=9.5,<9.6"

Собственные значки и оформление, линейки в мм, F12, две направляющие,
редактирование поверхности мышью в 3D, импорт SVG/DXF и экспорт STL.
Независимые слои и управляющие линии пока не реализованы.
"""

# ================== Импорт SVG ==================

import io
from svgelements import SVG, Path, Shape, Move, Close, CubicBezier, QuadraticBezier

def convert_path(path, node_type, path_type, scale=1.0):
    path = Path(path)
    path.approximate_arcs_with_cubics(error=0.01)
    result, nodes = ([], [])

    def point(p):
        return (float(p.x) * scale, float(p.y) * scale)

    def flush(closed=False):
        nonlocal nodes
        if closed and len(nodes) > 1:
            first, last = (nodes[0], nodes[-1])
            if abs(first.x - last.x) + abs(first.y - last.y) < 1e-08:
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
                a, q, b = (point(seg.start), point(seg.control), point(seg.end))
                nodes[-1].hout = tuple((a[i] + 2 * (q[i] - a[i]) / 3 for i in (0, 1)))
                incoming = tuple((b[i] + 2 * (q[i] - b[i]) / 3 for i in (0, 1)))
            nodes.append(node_type(*point(seg.end), hin=incoming))
    flush()
    return result

def parse_path(data, node_type, path_type):
    return convert_path(Path(data or ''), node_type, path_type)

def parse_svg(text, node_type, path_type):
    svg = SVG.parse(io.StringIO(text), reify=True, ppi=96, parse_display_none=False, on_error='raise')
    paths = []
    for element in svg.elements():
        if isinstance(element, Shape):
            paths.extend(convert_path(element, node_type, path_type, 25.4 / 96))
    width, height = (float(svg.width) * 25.4 / 96, float(svg.height) * 25.4 / 96)
    return (paths, (0.0, 0.0, width, height))

# ================== Расчёт геометрии рельефа ==================

import io
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from shapely.geometry import Polygon
from scipy.ndimage import distance_transform_edt, gaussian_filter

def photo_heights(raw, nx, ny, blur=0, invert=False):
    """Keep the source height precision; RGB previews never feed this calculation."""
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    if image.mode in ('I;16', 'I;16L', 'I;16B', 'I', 'F'):
        values = np.asarray(image, dtype=np.float32)
        if image.mode == 'F':
            if values.min() < 0 or values.max() > 1:
                values = (values - values.min()) / max(float(np.ptp(values)), 1e-12)
        else:
            values /= 65535.0
    else:
        if 'A' in image.getbands() or 'transparency' in image.info:
            rgba = image.convert('RGBA')
            bg = Image.new('RGBA', rgba.size, 'white')
            image = Image.alpha_composite(bg, rgba).convert('RGB')
        values = np.asarray(image.convert('L'), dtype=np.float32) / 255.0
    if not np.isfinite(values).all():
        raise ValueError('Карта высот содержит недопустимые значения.')
    if blur > 0:
        values = gaussian_filter(values, blur)
    resized = Image.fromarray(values).resize((nx, ny), Image.Resampling.LANCZOS)
    result = np.flipud(np.clip(np.asarray(resized, dtype=np.float32), 0, 1))
    return 1 - result if invert else result

def project_ruled(px, py, a, b):
    """Invert neighbouring bilinear patches instead of quantizing to a rail sample."""
    points = np.column_stack((px, py))
    d = b - a
    dd = np.maximum(np.sum(d * d, axis=1), 1e-18)
    outputs = []
    for start in range(0, len(points), 800):
        q = points[start:start + 800]
        r = q[:, None, :] - a[None, :, :]
        v = np.clip(np.sum(r * d, axis=2) / dd, 0, 1)
        err = np.sum((r - v[:, :, None] * d) ** 2, axis=2)
        k = np.argmin(err, axis=1)
        best = np.full(len(q), np.inf)
        best_v = np.zeros(len(q))
        best_s = np.zeros(len(q))
        best_w = np.zeros(len(q))
        for j in (np.maximum(k - 1, 0), np.minimum(k, len(a) - 2)):
            aa, da = (a[j], a[j + 1] - a[j])
            d0, delta = (d[j], d[j + 1] - d[j])
            t = np.full(len(q), 0.5)
            u = v[np.arange(len(q)), k].copy()
            for _ in range(8):
                dv = d0 + t[:, None] * delta
                ds = da + u[:, None] * delta
                residual = aa + t[:, None] * da + u[:, None] * dv - q
                det = ds[:, 0] * dv[:, 1] - ds[:, 1] * dv[:, 0]
                good = np.abs(det) > 1e-14
                safe = np.where(good, det, 1)
                dt = (residual[:, 0] * dv[:, 1] - residual[:, 1] * dv[:, 0]) / safe
                du = (ds[:, 0] * residual[:, 1] - ds[:, 1] * residual[:, 0]) / safe
                t = np.clip(t - np.where(good, dt, 0), 0, 1)
                u = np.clip(u - np.where(good, du, 0), 0, 1)
            width_vec = d0 + t[:, None] * delta
            err = np.sum((aa + t[:, None] * da + u[:, None] * width_vec - q) ** 2, axis=1)
            take = err < best
            best[take] = err[take]
            best_v[take] = u[take]
            best_s[take] = (j[take] + t[take]) / (len(a) - 1)
            best_w[take] = np.linalg.norm(width_vec[take], axis=1)
        outputs.append((best_v, best_w, best_s))
    if not outputs:
        return (np.empty(0), np.empty(0), np.empty(0))
    return tuple((np.concatenate([out[i] for out in outputs]) for i in range(3)))

def coverage(polys, nx, ny, ss=None):
    """Union overlapping regions; alternate fill for completely nested contours."""
    ss = ss or (4 if nx * ny <= 1200000 else 2)
    shapes = [Polygon(p) for p in polys if len(p) >= 3]
    if any((not p.is_valid for p in shapes)):
        raise ValueError('Контур пересекает сам себя. Исправьте узлы перед построением рельефа.')
    ordered = sorted(shapes, key=lambda p: p.area, reverse=True)
    image = Image.new('L', (nx * ss, ny * ss), 0)
    draw = ImageDraw.Draw(image)
    for index, p in enumerate(ordered):
        depth = sum((p.within(parent) and (not p.equals(parent)) for parent in ordered[:index]))
        xy = [((x + 0.5) * ss - 0.5, (y + 0.5) * ss - 0.5) for x, y in p.exterior.coords]
        draw.polygon(xy, fill=0 if depth % 2 else 255)
    return np.asarray(image).reshape(ny, ss, nx, ss).sum(axis=(1, 3), dtype=np.float32) / (255 * ss * ss)

def sculpt_weight(shape, center, cell, radius, mask=None, fixed=True):
    """Compact-support C2 displacement. Adds height without resampling fine detail."""
    yy, xx = np.indices(shape, dtype=np.float32)
    x, y = center
    cx, cy = cell
    r2 = ((xx - x) * cx) ** 2 + ((yy - y) * cy) ** 2
    weight = np.maximum(1 - r2 / max(radius * radius, 1e-12), 0) ** 3
    if mask is not None:
        weight *= mask
        if fixed:
            distance = distance_transform_edt(np.pad(mask, 1), sampling=(cy, cx))[1:-1, 1:-1]
            t = np.clip((distance - max(cx, cy)) / max(radius * 0.2, max(cx, cy)), 0, 1)
            weight *= t * t * (3 - 2 * t)
    return weight

# ================== Собственные значки ==================

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
PATHS = {'new': '<path d="M8 4h12l5 5v19H8z M20 4v6h5 M12 18h9 M16.5 13.5v9"/>', 'open': '<path d="M4 10h10l3 4h12l-4 13H5z M5 10V7h10l3 3"/>', 'save': '<path d="M6 5h18l3 3v19H6z M11 5v8h11V5 M11 27v-9h11v9"/>', 'undo': '<path d="M12 7l-7 7 7 6 M6 14h13q8 0 8 7v4"/>', 'redo': '<path d="M20 7l7 7-7 6 M26 14H13q-8 0-8 7v4"/>', 'select': '<path d="M8 4l19 15-10 1-4 9z"/>', 'nodes': '<path d="M6 24C6 6 26 27 26 8 M6 24l5-17 M26 8l-4 18"/><rect x="3" y="21" width="6" height="6"/><rect x="23" y="5" width="6" height="6"/>', 'pen': '<path d="M7 25l2-8L23 3l6 6-14 14z M9 17l6 6 M21 5l6 6"/>', 'rectangle': '<rect x="4" y="8" width="24" height="17" rx="1"/>', 'square': '<rect x="6" y="6" width="20" height="20" rx="1"/>', 'ellipse': '<ellipse cx="16" cy="16" rx="13" ry="8"/>', 'circle': '<circle cx="16" cy="16" r="11"/>', 'polygon': '<path d="M16 3l12 9-5 15H9L4 12z"/>', 'star': '<path d="M16 3l4 8 9 2-7 7 1 9-7-4-8 4 2-9-7-7 9-2z"/>', 'text': '<path d="M5 8V5h22v3 M16 5v22 M11 27h10"/>', 'point': '<circle cx="16" cy="16" r="5"/><path d="M16 2v6 M16 24v6 M2 16h6 M24 16h6"/>', 'shape': '<path d="M3 25h26 M5 23C7 4 25 4 27 23 M11 23C12 12 20 12 21 23"/>', 'rails': '<path d="M4 26Q5 10 13 4 M17 28Q30 18 26 5 M5 23Q17 11 27 19 M7 15Q18 5 27 12"/>', 'dome': '<path d="M4 23q12-28 24 0 M4 23q12 9 24 0 M16 6v21"/>', 'plate': '<path d="M3 14l17-8 10 9-17 9z M3 14v5l10 9 17-8v-5 M13 24v4"/>', 'sculpt': '<path d="M3 26q7-19 13-12t13 5 M17 6l4-3 6 8-5 3z M17 6l-5 10 10-2"/>', 'smooth': '<path d="M3 23l5-10 5 7 6-14 10 12 M3 27q11-13 26-4"/>', 'zero': '<path d="M3 24h26 M8 8l16 11 M24 8L8 19"/>', 'stl': '<path d="M16 3L3 11v14l13 5 13-9V9z M3 11l13 7 13-9 M16 18v12 M16 3v15"/>', 'photo': '<rect x="3" y="5" width="26" height="22" rx="2"/><circle cx="10" cy="12" r="3"/><path d="M4 25l8-8 5 4 6-9 6 13"/>', 'fit': '<path d="M11 4H4v7 M21 4h7v7 M4 21v7h7 M28 21v7h-7"/><rect x="9" y="10" width="14" height="12"/>', 'copy': '<rect x="5" y="4" width="16" height="19"/><path d="M12 23v5h15V10h-6"/>', 'delete': '<path d="M5 8h22 M12 8V4h8v4 M8 8l2 20h12l2-20 M13 12v11 M19 12v11"/>', 'transform': '<path d="M4 13V4h9 M4 4l9 9 M19 28h9v-9 M28 28l-9-9"/><rect x="11" y="11" width="10" height="10"/>'}

def icon(name, accent=False):
    color = '#b66a3b' if accent else '#344b5b'
    body = PATHS.get(name, PATHS['point'])
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32"><g fill="none" stroke="{color}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{body}</g></svg>'
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    QSvgRenderer(QByteArray(svg.encode())).render(painter)
    painter.end()
    return QIcon(pixmap)

# ================== Основа редактора ==================

import hashlib
import uuid
import itertools
import io
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
LANCZOS = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS')
Image.MAX_IMAGE_PIXELS = None

class Node:
    __slots__ = ('x', 'y', 'hin', 'hout')

    def __init__(self, x, y, hin=None, hout=None):
        self.x, self.y, self.hin, self.hout = (float(x), float(y), hin, hout)

    def copy(self):
        return Node(self.x, self.y, None if self.hin is None else tuple(self.hin), None if self.hout is None else tuple(self.hout))

def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

class PathData:
    """Вектор: узлы с (необязательными) ручками Безье; открытый или замкнутый."""

    def __init__(self, nodes=None, closed=False, group=0):
        self.nodes = [n.copy() for n in nodes or []]
        self.closed = bool(closed)
        self.group = int(group)
        self.uid = uuid.uuid4().hex

    def copy(self):
        return PathData(self.nodes, self.closed, self.group)

    def seg_count(self):
        n = len(self.nodes)
        return n if self.closed and n > 2 else max(n - 1, 0)

    def seg_ctrl(self, i):
        a = self.nodes[i]
        b = self.nodes[(i + 1) % len(self.nodes)]
        p0, p3 = ((a.x, a.y), (b.x, b.y))
        p1 = tuple(a.hout) if a.hout is not None else p0
        p2 = tuple(b.hin) if b.hin is not None else p3
        return (p0, p1, p2, p3, a.hout is None and b.hin is None)

    def seg_points(self, i, n=24):
        p0, p1, p2, p3, straight = self.seg_ctrl(i)
        if straight:
            return [p0, p3]
        pts = []
        for k in range(n + 1):
            t = k / n
            u = 1 - t
            pts.append((u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0], u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1]))
        return pts

    def flatten(self, n=24):
        pts = []
        for i in range(self.seg_count()):
            sp = self.seg_points(i, n)
            pts.extend(sp if not pts else sp[1:])
        if not pts and self.nodes:
            pts = [(self.nodes[0].x, self.nodes[0].y)]
        return pts

    def nearest_segment(self, pt, n=16):
        """(индекс сегмента, расстояние, t≈0..1, ближайшая точка) или None."""
        best = None
        for i in range(self.seg_count()):
            sp = self.seg_points(i, n)
            m = len(sp) - 1
            for k in range(m):
                a, b = (sp[k], sp[k + 1])
                abx, aby = (b[0] - a[0], b[1] - a[1])
                den = abx * abx + aby * aby
                u = 0.0 if den < 1e-12 else max(0.0, min(1.0, ((pt[0] - a[0]) * abx + (pt[1] - a[1]) * aby) / den))
                q = (a[0] + abx * u, a[1] + aby * u)
                d = math.hypot(pt[0] - q[0], pt[1] - q[1])
                if best is None or d < best[1]:
                    best = (i, d, (k + u) / m, q)
        return best

    def hit(self, x, y, tol):
        best, bd = (None, tol)
        for i, n in enumerate(self.nodes):
            d = math.hypot(n.x - x, n.y - y)
            if d <= bd:
                bd, best = (d, ('node', i))
        if best:
            return best
        r = self.nearest_segment((x, y))
        if r and r[1] <= tol:
            return ('seg', r[0], r[2])
        return None

    def insert_at(self, i, t):
        """Вставить узел на сегмент i в параметре t (форма кривой сохраняется)."""
        p0, p1, p2, p3, straight = self.seg_ctrl(i)
        a = self.nodes[i]
        b = self.nodes[(i + 1) % len(self.nodes)]
        if straight:
            q = _lerp(p0, p3, t)
            self.nodes.insert(i + 1, Node(q[0], q[1]))
            return
        q0, q1, q2 = (_lerp(p0, p1, t), _lerp(p1, p2, t), _lerp(p2, p3, t))
        r0, r1 = (_lerp(q0, q1, t), _lerp(q1, q2, t))
        s = _lerp(r0, r1, t)
        a.hout = q0
        b.hin = q2
        self.nodes.insert(i + 1, Node(s[0], s[1], r0, r1))

    def remove_node(self, i):
        if len(self.nodes) > (3 if self.closed else 2):
            self.nodes.pop(i)
            return True
        return False

    def is_smooth(self, i):
        n = self.nodes[i]
        return n.hin is not None and n.hout is not None

    def make_corner(self, i):
        self.nodes[i].hin = self.nodes[i].hout = None

    def make_smooth(self, i):
        n = self.nodes[i]
        m = len(self.nodes)
        prev = self.nodes[(i - 1) % m] if self.closed or i > 0 else None
        nxt = self.nodes[(i + 1) % m] if self.closed or i < m - 1 else None
        if prev is None and nxt is None:
            return
        if prev is None:
            n.hin = None
            n.hout = (n.x + (nxt.x - n.x) / 3, n.y + (nxt.y - n.y) / 3)
        elif nxt is None:
            n.hout = None
            n.hin = (n.x + (prev.x - n.x) / 3, n.y + (prev.y - n.y) / 3)
        else:
            tx, ty = ((nxt.x - prev.x) / 6, (nxt.y - prev.y) / 6)
            n.hin, n.hout = ((n.x - tx, n.y - ty), (n.x + tx, n.y + ty))

    def remove_segment(self, i):
        """Удалить участок между узлами i и i+1; вернуть список независимых векторов."""
        n = len(self.nodes)
        if n < 2:
            return []
        if self.closed and n > 2:
            chain = [x.copy() for x in self.nodes[i + 1:] + self.nodes[:i + 1]]
            chain[-1].hout = None
            chain[0].hin = None
            return [PathData(chain, False, self.group)]
        left = [x.copy() for x in self.nodes[:i + 1]]
        right = [x.copy() for x in self.nodes[i + 1:]]
        left[-1].hout = None
        right[0].hin = None
        return [PathData(p, False, self.group) for p in (left, right) if len(p) >= 2]

    def split_at(self, i):
        n = len(self.nodes)
        if self.closed and n > 2:
            chain = [x.copy() for x in self.nodes[i:] + self.nodes[:i]] + [self.nodes[i].copy()]
            chain[0].hin = None
            chain[-1].hout = None
            return [PathData(chain, False, self.group)]
        left = [x.copy() for x in self.nodes[:i + 1]]
        right = [x.copy() for x in self.nodes[i:]]
        left[-1].hout = None
        right[0].hin = None
        return [PathData(p, False, self.group) for p in (left, right) if len(p) >= 2]

    def join(self, other, tol=1e-06):
        """Соединить два открытых вектора ближайшими концами."""
        a, b = (self.copy(), other.copy())
        best = None
        for ra in (False, True):
            for rb in (False, True):
                ea = a.nodes[0] if ra else a.nodes[-1]
                eb = b.nodes[-1] if rb else b.nodes[0]
                d = math.hypot(ea.x - eb.x, ea.y - eb.y)
                if best is None or d < best[0]:
                    best = (d, ra, rb)
        _, ra, rb = best
        if ra:
            a.reverse()
        if rb:
            b.reverse()
        if math.hypot(a.nodes[-1].x - b.nodes[0].x, a.nodes[-1].y - b.nodes[0].y) <= tol:
            a.nodes[-1].hout = b.nodes[0].hout
            b.nodes.pop(0)
        a.nodes.extend(b.nodes)
        a.closed = False
        return a

    def reverse(self):
        self.nodes.reverse()
        for n in self.nodes:
            n.hin, n.hout = (n.hout, n.hin)

    def transform(self, f):
        for n in self.nodes:
            n.x, n.y = f(n.x, n.y)
            if n.hin is not None:
                n.hin = f(*n.hin)
            if n.hout is not None:
                n.hout = f(*n.hout)

    def translate(self, dx, dy):
        self.transform(lambda x, y: (x + dx, y + dy))

    def to_dict(self):
        return {'n': [[n.x, n.y, None if n.hin is None else list(n.hin), None if n.hout is None else list(n.hout)] for n in self.nodes], 'c': self.closed, 'g': self.group, 'uid': self.uid}

    @staticmethod
    def from_dict(d):
        p = PathData([Node(x, y, tuple(a) if a else None, tuple(b) if b else None) for x, y, a, b in d['n']], d.get('c', False), d.get('g', 0))
        p.uid = d.get('uid') or p.uid
        return p

    def svg_d(self, s=1.0):
        if not self.nodes:
            return ''
        f = lambda p: f'{p[0] * s:.4f} {p[1] * s:.4f}'
        out = ['M ' + f((self.nodes[0].x, self.nodes[0].y))]
        for i in range(self.seg_count()):
            p0, p1, p2, p3, st = self.seg_ctrl(i)
            out.append('L ' + f(p3) if st else 'C ' + f(p1) + ' ' + f(p2) + ' ' + f(p3))
        if self.closed and len(self.nodes) > 2:
            out.append('Z')
        return ' '.join(out)

def polyline_path(points, closed=False):
    return PathData([Node(x, y) for x, y in points], closed)

def rdp(points, eps):
    pts = np.asarray(points, float)
    if len(pts) < 3:
        return [tuple(p) for p in pts]
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = pts[b] - pts[a]
        d = pts[a + 1:b] - pts[a]
        norm = math.hypot(*seg)
        dist = np.hypot(d[:, 0], d[:, 1]) if norm < 1e-12 else np.abs(seg[0] * d[:, 1] - seg[1] * d[:, 0]) / norm
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = a + 1 + k
            keep[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return [tuple(p) for p in pts[keep]]
_TOK = re.compile('[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?')

def parse_svg_path(d):
    return parse_path(d, Node, PathData)

def _ellipse_path(cx, cy, rx, ry):
    k = 0.5522847498
    return PathData([Node(cx + rx, cy, (cx + rx, cy - k * ry), (cx + rx, cy + k * ry)), Node(cx, cy + ry, (cx + k * rx, cy + ry), (cx - k * rx, cy + ry)), Node(cx - rx, cy, (cx - rx, cy + k * ry), (cx - rx, cy - k * ry)), Node(cx, cy - ry, (cx - k * rx, cy - ry), (cx + k * rx, cy - ry))], True)

def svg_to_paths(text):
    return parse_svg(text, Node, PathData)

def svg_document(paths, w_mm, h_mm, scale):
    """SVG в миллиметрах. scale = 1/K (пиксели сцены → мм)."""
    body = '\n'.join((f'  <path d="{p.svg_d(scale)}" fill="none" stroke="black" stroke-width="0.2"/>' for p in paths if p.nodes))
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{w_mm:g}mm" height="{h_mm:g}mm" viewBox="0 0 {w_mm:g} {h_mm:g}">\n{body}\n</svg>\n'

def grid_shape(W, H, res):
    res = max(int(res), 8)
    if W >= H:
        return (max(2, int(round(res * H / W))), res)
    return (res, max(2, int(round(res * W / H))))

def grid_from_step(W, H, step, max_cells=8000000):
    """Размер сетки (nx, ny) для шага в мм; при слишком большой сетке шаг увеличивается."""
    step = max(float(step), 0.001)
    nx, ny = (int(round(W / step)) + 1, int(round(H / step)) + 1)
    used = step
    if nx * ny > max_cells:
        used = math.sqrt(W * H / max_cells)
        nx, ny = (int(round(W / used)) + 1, int(round(H / used)) + 1)
    return (max(nx, 2), max(ny, 2), used)

def grid_shape_xy(W, H, res):
    ny, nx = grid_shape(W, H, res) if W >= H else (None, None)
    if W >= H:
        return (nx, ny)
    return (max(2, int(round(res * W / H))), max(int(res), 8))

def boundary_loop(ny, nx):
    idx = np.arange(nx * ny).reshape(ny, nx)
    return np.concatenate([idx[0, :-1], idx[:-1, -1], idx[-1, :0:-1], idx[:0:-1, 0]])

def build_mesh_arrays(z, W, H):
    """Замкнутая модель: верх по z (мм), плоское дно (веер от центра), боковые стенки."""
    ny, nx = z.shape
    xs = np.linspace(-W / 2, W / 2, nx, dtype=np.float32)
    ys = np.linspace(-H / 2, H / 2, ny, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    top = np.column_stack((xx.ravel(), yy.ravel(), np.asarray(z, dtype=np.float32).ravel()))
    loop = boundary_loop(ny, nx)
    m = len(loop)
    bot = top[loop].copy()
    bot[:, 2] = 0.0
    verts = np.vstack((top, bot, np.zeros((1, 3), np.float32)))
    off = nx * ny
    cidx = off + m
    idx = np.arange(nx * ny, dtype=np.int32).reshape(ny, nx)
    a, b, c, d = (idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel())
    top_f = np.concatenate([np.column_stack((a, b, d)), np.column_stack((a, d, c))])
    k = np.arange(m, dtype=np.int32)
    kn = np.roll(k, -1)
    lp = loop.astype(np.int32)
    side = np.concatenate([np.column_stack((lp, off + k, off + kn)), np.column_stack((lp, off + kn, np.roll(lp, -1)))])
    fan = np.column_stack((np.full(m, cidx, np.int32), off + kn, off + k))
    return (verts, np.concatenate([top_f, side, fan]).astype(np.int32))

def reduce_for_display(z, mask=None, max_side=1400, max_verts=600000):
    """Упрощённая сетка только для показа в 3D (STL и расчёты остаются в полном разрешении)."""
    ny, nx = z.shape
    s_ = max(1, int(math.ceil(max(nx, ny) / max_side)), int(math.ceil(math.sqrt(nx * ny / max_verts))))
    if s_ == 1:
        return (z, mask)
    iy = np.unique(np.r_[np.arange(0, ny, s_), ny - 1])
    ix = np.unique(np.r_[np.arange(0, nx, s_), nx - 1])
    return (z[np.ix_(iy, ix)], None if mask is None else mask[np.ix_(iy, ix)])

def write_stl(path, verts, faces):
    """Двоичный STL (единицы — мм), запись порциями."""
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces)
    with open(path, 'wb') as f:
        f.write(b'Stone Relief Studio Pro'.ljust(80, b' '))
        f.write(struct.pack('<I', len(faces)))
        for s0 in range(0, len(faces), 400000):
            tri = verts[faces[s0:s0 + 400000]]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            ln = np.linalg.norm(n, axis=1)
            ln[ln == 0] = 1
            n = n / ln[:, None]
            rec = np.zeros(len(tri), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
            rec['n'], rec['v'] = (n, tri)
            f.write(rec.tobytes())

def is_watertight(faces):
    f = np.asarray(faces, dtype=np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    key = e.min(axis=1) * (int(f.max()) + 1) + e.max(axis=1)
    _, cnt = np.unique(key, return_counts=True)
    return bool((cnt == 2).all())

def heightmap_from_mesh(mesh, W, H, nx, ny):
    xs = np.linspace(-W / 2 + 0.001, W / 2 - 0.001, nx)
    ys = np.linspace(-H / 2 + 0.001, H / 2 - 0.001, ny)
    xx, yy = np.meshgrid(xs, ys)
    top = float(mesh.bounds[1][2]) + 1.0
    origins = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, top)))
    dirs = np.tile([0.0, 0.0, -1.0], (xx.size, 1))
    locs, ridx, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=True)
    z = np.full(xx.size, np.nan)
    if len(ridx):
        tmp = np.full(xx.size, -np.inf)
        np.maximum.at(tmp, ridx, locs[:, 2])
        z = np.where(np.isfinite(tmp), tmp, np.nan)
    z = z.reshape(ny, nx)
    if np.isnan(z).all():
        raise ValueError('Не удалось получить карту высот')
    fill = float(np.nanmin(z))
    return np.where(np.isnan(z), fill, z)

def raster_mask(polys, nx, ny):
    img = Image.new('L', (nx, ny), 0)
    d = ImageDraw.Draw(img)
    for p in polys:
        if len(p) >= 3:
            d.polygon([(float(x), float(y)) for x, y in p], fill=255)
    return np.asarray(img) > 0

def edge_distance(mask, maxd):
    """0 на границе маски и растёт к центру до maxd."""
    maxd = int(max(1, min(maxd, 300)))
    ys, xs = np.where(mask)
    out = np.zeros(mask.shape, np.float32)
    if len(xs) == 0:
        return out
    y0, y1, x0, x1 = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    cur = np.pad(mask[y0:y1, x0:x1], 1)
    dist = np.zeros(cur.shape, np.float32)
    for k in range(1, maxd + 1):
        inner = cur.copy()
        inner[1:, :] &= cur[:-1, :]
        inner[:-1, :] &= cur[1:, :]
        inner[:, 1:] &= cur[:, :-1]
        inner[:, :-1] &= cur[:, 1:]
        dist[cur & ~inner] = k - 1
        cur = inner
    dist[cur] = maxd
    out[y0:y1, x0:x1] = dist[1:-1, 1:-1]
    return out

def section_curve(points, samples=24):
    P = np.array(sorted(points, key=lambda p: p[0]), float)
    if len(P) < 2:
        P = np.array([[0, 0], [1, 1]], float)
    ext = np.vstack([P[0], P, P[-1]])
    xs, ys = ([], [])
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = (ext[i], ext[i + 1], ext[i + 2], ext[i + 3])
        for k in range(samples):
            u = k / samples
            q = 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
            xs.append(q[0])
            ys.append(q[1])
    xs.append(P[-1][0])
    ys.append(P[-1][1])
    return (np.maximum.accumulate(np.array(xs)), np.array(ys))

def section_eval(points, t):
    xs, ys = section_curve(points)
    return np.interp(np.clip(t, 0, 1), xs, ys)

def _min_dist(px, py, G):
    out = np.empty(len(px))
    for s in range(0, len(px), 3000):
        dx = px[s:s + 3000, None] - G[None, :, 0]
        dy = py[s:s + 3000, None] - G[None, :, 1]
        out[s:s + 3000] = np.sqrt(dx * dx + dy * dy).min(axis=1)
    return out

def densify(poly, step):
    poly = np.asarray(poly, float)
    out = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(1, int(math.hypot(*b - a) / max(step, 1e-06)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)

def _box1d(a, r, axis):
    if r <= 0:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode='edge'), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)

def gaussian_blur(a, sigma):
    """Гауссово размытие (3 прохода box-фильтра), только numpy."""
    a = np.asarray(a, dtype=np.float64)
    if sigma < 0.3:
        return a
    r = max(1, int(round((-1 + math.sqrt(1 + 4 * sigma * sigma)) / 2)))
    for _ in range(3):
        a = _box1d(_box1d(a, r, 0), r, 1)
    return a

def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

def masked_smooth(a, mask, sigma):
    m = mask.astype(np.float64)
    num, den = (gaussian_blur(a * m, sigma), gaussian_blur(m, sigma))
    return np.where(mask, num / np.maximum(den, 1e-06), 0.0)

def raster_coverage(polys, nx, ny, ss=None):
    return coverage(polys, nx, ny, ss)

def distance_inside(mask):
    """Расстояние (в ячейках) от каждой внутренней ячейки до границы области (0 на границе)."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.zeros(mask.shape)
    try:
        from scipy.ndimage import distance_transform_edt
        return np.maximum(distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] - 0.5, 0.0)
    except Exception:
        ny, nx = mask.shape
        f = max(1, int(math.ceil(max(ny, nx) / 300.0)))
        ms = mask[::f, ::f]
        p = np.pad(ms, 1)
        er = p.copy()
        er[1:, :] &= p[:-1, :]
        er[:-1, :] &= p[1:, :]
        er[:, 1:] &= p[:, :-1]
        er[:, :-1] &= p[:, 1:]
        by, bx = np.where(p & ~er)
        iy, ix = np.where(ms)
        G = np.column_stack((bx - 1.0, by - 1.0))
        d = _min_dist(ix.astype(float), iy.astype(float), G)
        dsm = np.zeros(ms.shape)
        dsm[iy, ix] = d + 0.5
        yy, xx = np.indices((ny, nx), dtype=np.float64)
        sx = np.clip(xx / f, 0, ms.shape[1] - 1)
        sy = np.clip(yy / f, 0, ms.shape[0] - 1)
        if ms.shape[0] < 2 or ms.shape[1] < 2:
            return np.where(mask, 1.0, 0.0)
        out, _ = bilinear(dsm, sx, sy)
        return np.where(mask, out * f, 0.0)

def feather_weight(cov, feather_px):
    """0 на границе области, плавно (smoothstep) растёт к центру на feather_px."""
    f = max(float(feather_px), 0.5)
    if not (cov > 0.5).any():
        return np.zeros(cov.shape)
    return smoothstep(distance_inside(cov > 0.5) / f) * (cov > 0)

def shape_profile(kind, angle, d, w, cap=None):
    """Высота формы на расстоянии d от границы; w — максимальное расстояние (центр).
    kind: round | angular | plateau. Отрицательный угол — вогнутая форма (выемка)."""
    th = math.radians(min(max(abs(angle), 1.0), 89.9))
    sign = -1.0 if angle < 0 else 1.0
    d = np.clip(np.asarray(d, float), 0, w)
    if kind == 'round':
        R = w / math.sin(th)
        h = -R * math.cos(th) + np.sqrt(np.maximum(R * R - (d - w) ** 2, 0.0))
    elif kind == 'angular':
        h = d * math.tan(th)
    else:
        top = cap if cap is not None else 0.5 * w * math.tan(th)
        h = np.minimum(d * math.tan(th), top)
    return sign * np.maximum(h, 0.0)

def build_shape(cov, cell_x, cell_y, kind='round', angle=45.0, mode='free', value=1.0, start_h=0.0, scale=1.0):
    """Форма внутри области (как «Редактор формы» ArtCAM): мм над нулём рельефа.
    mode: free (без ограничений) | limit (ограничить по высоте) | scale (масштаб по высоте) | const (постоянная)."""
    inside = cov > 0.5
    if not inside.any():
        raise ValueError('Область слишком мала для текущей детализации.')
    d = distance_inside(inside) * 0.5 * (cell_x + cell_y)
    w = max(float(d.max()), 1e-09)
    h = shape_profile(kind, angle, d, w, cap=value if mode in ('limit', 'scale') else None)
    if mode == 'limit':
        h = np.clip(h, -abs(value), abs(value))
    elif mode == 'scale':
        mx = float(np.abs(h).max())
        if mx > 0:
            h = h * (abs(value) / mx)
    elif mode == 'const':
        h = np.full(d.shape, float(value))
    return np.where(cov > 0.001, start_h + h * scale, 0.0)

def resample_polyline(pts, n):
    pts = np.asarray(pts, float)
    seg = np.hypot(*np.diff(pts, axis=0).T) if len(pts) > 1 else np.zeros(0)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    if L[-1] < 1e-09:
        return np.repeat(pts[:1], n, axis=0)
    s_ = np.linspace(0, L[-1], n)
    return np.column_stack([np.interp(s_, L, pts[:, 0]), np.interp(s_, L, pts[:, 1])])

def _polylen(P):
    return float(np.hypot(*np.diff(P, axis=0).T).sum()) if len(P) > 1 else 0.0

def prepare_rails(A, B, cell_x, cell_y, n=None, flip1=False, flip2=False):
    """Направляющие в мм: автоматически одной ориентации, затем «Развернуть» по флагам; ресэмпл на n точек."""
    A = np.asarray(A, float) * [cell_x, cell_y]
    B = np.asarray(B, float) * [cell_x, cell_y]
    if math.hypot(*A[0] - B[0]) + math.hypot(*A[-1] - B[-1]) > math.hypot(*A[0] - B[-1]) + math.hypot(*A[-1] - B[0]):
        B = B[::-1]
    if flip1:
        A = A[::-1]
    if flip2:
        B = B[::-1]
    if n is None:
        n = int(np.clip(max(_polylen(A), _polylen(B)) / 1.0, 200, 600))
    return (resample_polyline(A, n), resample_polyline(B, n))

def rails_polygon(A, B, cell_x, cell_y):
    """Контур области между направляющими в координатах сетки (без самопересечений)."""
    A_, B_ = prepare_rails(A, B, cell_x, cell_y, n=120)
    pts = np.vstack((A_, B_[::-1])) / [cell_x, cell_y]
    return [tuple(p) for p in pts]

def two_rail_param(mask, cell_x, cell_y, A, B, n=None, max_direct=80000, flip1=False, flip2=False):
    """(v, ширина, s): v — 0 у направляющей A … 1 у B; ширина между направляющими (мм);
    s — положение вдоль направляющих 0…1; порядок np.where. Большие области — крупная решётка + интерполяция."""
    A, B = prepare_rails(A, B, cell_x, cell_y, n, flip1, flip2)
    n = len(A)
    D = B - A
    DD = np.maximum((D * D).sum(1), 1e-12)

    def evaluate(px, py):
        return project_ruled(px, py, A, B)
    ys, xs = np.where(mask)
    if len(xs) <= max_direct:
        return evaluate(xs * cell_x, ys * cell_y)
    f = int(math.ceil(math.sqrt(len(xs) / float(max_direct))))
    x0, x1, y0, y1 = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
    gx = x0 + f * np.arange(max(1, int(math.ceil((x1 - x0) / f))) + 1)
    gy = y0 + f * np.arange(max(1, int(math.ceil((y1 - y0) / f))) + 1)
    GX, GY = np.meshgrid(gx, gy)
    vv, ww, ss = evaluate(GX.ravel() * cell_x, GY.ravel() * cell_y)
    sx, sy = ((xs - x0) / float(f), (ys - y0) / float(f))
    out = []
    for arr in (vv, ww, ss):
        r, _ = bilinear(arr.reshape(GY.shape), sx, sy)
        out.append(r)
    return (np.clip(out[0], 0, 1), out[1], np.clip(out[2], 0, 1))

def zboundary_heights(A, B, Z, cell_x, cell_y, flip1=False, flip2=False):
    """«Граница по Z»: высота сечения в каждой точке траектории = расстояние от середины между
    направляющими до вектора границы (мм). Возвращает (s, высоты)."""
    A_, B_ = prepare_rails(A, B, cell_x, cell_y, None, flip1, flip2)
    M = (A_ + B_) / 2
    Zd = densify(np.asarray(Z, float) * [cell_x, cell_y], max(min(cell_x, cell_y), 0.001))
    return (np.linspace(0, 1, len(M)), _min_dist(M[:, 0], M[:, 1], Zd))

def sweep_values(v, wid, sp, profs, pos, p):
    """Высоты вытягивания: сечения profs (функции v→мм) на позициях pos вдоль направляющих.
    p: spans, scale_w, transition, start_h, scale_end, end_h, hA, hB, zb=(s, H)."""
    m = len(v)
    k = len(profs)
    spans = p.get('spans') or [None] * k
    mean_w = float(np.mean(wid)) if m else 1.0
    grid = np.linspace(0, 1, 201)
    vals, mx = ([], [])
    for f, span in zip(profs, spans):
        pv = np.asarray(f(v), float)
        gm = float(np.max(f(grid)))
        if p.get('scale_w', True):
            fac = wid / max(span or mean_w, 1e-09)
            pv, mxi = (pv * fac, gm * fac)
        else:
            mxi = np.full(m, gm)
        vals.append(pv)
        mx.append(mxi)
    vals, mx = (np.stack(vals), np.stack(mx))
    ar = np.arange(m)
    if k == 1:
        prof, pmx = (vals[0], mx[0])
    else:
        pos = np.asarray(pos, float)
        if p.get('transition', True):
            idx = np.clip(np.searchsorted(pos, sp, side='right') - 1, 0, k - 2)
            t = np.clip((sp - pos[idx]) / np.maximum(pos[idx + 1] - pos[idx], 1e-09), 0, 1)
            t = t * t * (3 - 2 * t)
            prof = vals[idx, ar] * (1 - t) + vals[idx + 1, ar] * t
            pmx = mx[idx, ar] * (1 - t) + mx[idx + 1, ar] * t
        else:
            idx = np.argmin(np.abs(sp[None, :] - pos[:, None]), axis=0)
            prof, pmx = (vals[idx, ar], mx[idx, ar])
    zb = p.get('zb')
    if zb is not None:
        prof = prof * (np.interp(sp, zb[0], zb[1]) / np.maximum(pmx, 1e-06))
    start = p.get('start_h', 0.0)
    base = start + (p.get('end_h', start) - start) * sp if p.get('scale_end') else start
    return (1 - v) * p.get('hA', 0.0) + v * p.get('hB', 0.0) + base + prof

def profile_from_points(pts):
    """pts [(x, h_мм)] → функция v∈[0,1] → мм; профиль растягивается между направляющими."""
    P = sorted(pts)
    xs = np.array([p[0] for p in P], float)
    hs = np.array([p[1] for p in P], float)
    span = xs.max() - xs.min() if len(P) else 0.0
    if len(P) < 2 or span < 1e-09:
        raise ValueError('Профиль слишком короткий.')
    vs = (xs - xs.min()) / span
    f = lambda v: np.interp(np.clip(v, 0, 1), vs, hs)
    f.span = float(span)
    return f

def deform_offsets(mask, cell_x, cell_y, p, guides=None, ctrl=None, rails=None, v_param=None):
    """Форма (мм над нулём) в области mask. p['mode']: bend | wave | apex | guides | sweep | points."""
    ny, nx = mask.shape
    ys_, xs_ = np.where(mask)
    if len(xs_) == 0:
        raise ValueError('Область пуста')
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    S = lambda t: section_eval(p['section'], t)
    mode = p['mode']
    off = np.zeros((ny, nx))
    if mode in ('bend', 'wave'):
        th = math.radians(p['angle'])
        proj = xx * cell_x * math.cos(th) + yy * cell_y * math.sin(th)
        pv = proj[mask]
        lo, hi = (float(pv.min()), float(pv.max()))
        t = np.clip((proj - lo) / max(hi - lo, 1e-09), 0, 1)
        if mode == 'bend':
            off = p['a'] + (p['b'] - p['a']) * S(t)
        else:
            u = t * max(p['waves'], 0.5)
            frac = u - np.floor(u)
            off = p['amount'] * S(1 - np.abs(2 * frac - 1))
    elif mode == 'apex':
        x0, x1, y0, y1 = (xs_.min(), xs_.max(), ys_.min(), ys_.max())
        cx, cy = (x0 + (x1 - x0) * p['apex_x'], y0 + (y1 - y0) * p['apex_y'])
        R = max(p['radius'] / 100.0 * max((x1 - x0) * cell_x, (y1 - y0) * cell_y), 1e-06)
        d = np.hypot((xx - cx) * cell_x, (yy - cy) * cell_y)
        off = p['amount'] * S(np.clip(1 - d / R, 0, 1))
    elif mode == 'guides':
        if guides is None or len(guides) < 2:
            raise ValueError('Выделите два открытых вектора (направляющие).')
        v, _, _ = v_param if v_param is not None else two_rail_param(mask, cell_x, cell_y, guides[0], guides[1])
        off[ys_, xs_] = p['hA'] + (p['hB'] - p['hA']) * S(v)
    elif mode == 'sweep':
        if rails is None or len(rails) < 2:
            raise ValueError('Для выдавливания выделите две открытые направляющие.')
        v, wid, sp = v_param if v_param is not None else two_rail_param(mask, cell_x, cell_y, rails[0], rails[1])
        q = dict(p, scale_w=p.get('scale_h', False), spans=[p.get('profile_span')])
        off[ys_, xs_] = sweep_values(v, wid, sp, [p['profile']], [0.0], q)
    elif mode == 'points':
        if not ctrl:
            raise ValueError('Добавьте управляющие точки (инструмент «◉»).')
        R = max(p['influence'], 0.1)
        num = np.zeros(len(xs_))
        den = np.full(len(xs_), 1.0 / (R * R))
        for gx, gy, dh in ctrl:
            w = 1.0 / (((xs_ - gx) * cell_x) ** 2 + ((ys_ - gy) * cell_y) ** 2 + 0.05 ** 2)
            num += w * dh
            den += w
        off[ys_, xs_] = num / den
    else:
        raise ValueError('Неизвестный режим')
    if mode in ('guides', 'sweep', 'points'):
        off = masked_smooth(off, mask, 0.9)
    return off
COMBINE_KEYS = ['add', 'high', 'low', 'sub', 'replace']
COMBINE_NAMES = ['Добавить сверху', 'Слить по наибольшему', 'Слить по наименьшему', 'Вычесть', 'Заменить']

def combine_relief(z, shape, weight, base, mode):
    """Слияние формы (мм над нулём) с рельефом. mode: add | sub | high | low | replace | zero."""
    r = np.asarray(z, dtype=np.float64) - base
    if mode == 'add':
        rn = r + shape
    elif mode == 'sub':
        rn = r - shape
    elif mode == 'high':
        rn = np.maximum(r, shape)
    elif mode == 'low':
        rn = np.minimum(r, shape)
    elif mode == 'replace':
        rn = shape * np.ones_like(r)
    elif mode == 'zero':
        rn = np.zeros_like(r)
    else:
        raise ValueError('Неизвестный режим слияния')
    return np.maximum(base + r + weight * (rn - r), 0.05)

def load_image_any(raw):
    """Открыть практически любое растровое фото → RGB (CMYK, палитра, прозрачность, 16 бит)."""
    img = Image.open(io.BytesIO(raw))
    img.load()
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode in ('RGBA', 'LA', 'PA') or (img.mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        bg = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba).convert('RGB')
    elif img.mode in ('I;16', 'I;16L', 'I;16B', 'I', 'F'):
        a = np.asarray(img, dtype=np.float64)
        a = (a - a.min()) / max(a.max() - a.min(), 1e-09) * 255
        img = Image.fromarray(a.astype(np.uint8)).convert('RGB')
    else:
        img = img.convert('RGB')
    if min(img.size) < 2:
        raise ValueError('Изображение слишком маленькое.')
    return img

def bilinear(arr, sx, sy):
    ny, nx = arr.shape
    inside = (sx >= 0) & (sx <= nx - 1) & (sy >= 0) & (sy <= ny - 1)
    x0 = np.clip(np.floor(sx).astype(int), 0, nx - 2)
    y0 = np.clip(np.floor(sy).astype(int), 0, ny - 2)
    fx, fy = (sx - x0, sy - y0)
    v = arr[y0, x0] * (1 - fx) * (1 - fy) + arr[y0, x0 + 1] * fx * (1 - fy) + arr[y0 + 1, x0] * (1 - fx) * fy + arr[y0 + 1, x0 + 1] * fx * fy
    return (v, inside)

def mask_centroid(mask):
    ys, xs = np.where(mask)
    return (float(xs.mean()), float(ys.mean()))

def ring_fill(z, mask, rings=3):
    cur = mask.copy()
    for _ in range(rings):
        n = cur.copy()
        n[1:, :] |= cur[:-1, :]
        n[:-1, :] |= cur[1:, :]
        n[:, 1:] |= cur[:, :-1]
        n[:, :-1] |= cur[:, 1:]
        cur = n
    ring = cur & ~mask
    return float(np.median(z[ring])) if ring.any() else float(np.median(z))

def transform_region(z, mask, tx, ty, angle, scale, flipx, flipy, copy, cell_x, cell_y):
    """Перенос/поворот/масштаб/отражение области рельефа. tx,ty — в ячейках сетки (y вверх)."""
    ny, nx = z.shape
    cx, cy = mask_centroid(mask)
    yy, xx = np.indices(z.shape, dtype=np.float64)
    ux = (xx - cx - tx) * cell_x / scale
    uy = (yy - cy - ty) * cell_y / scale
    a = math.radians(angle)
    ca, sa = (math.cos(a), math.sin(a))
    rx, ry = (ca * ux + sa * uy, -sa * ux + ca * uy)
    if flipx:
        rx = -rx
    if flipy:
        ry = -ry
    sx, sy = (cx + rx / cell_x, cy + ry / cell_y)
    zv, ins = bilinear(np.asarray(z, float), sx, sy)
    mv, ins2 = bilinear(mask.astype(float), sx, sy)
    dest = (mv > 0.5) & ins & ins2
    out = np.array(z, dtype=np.float64, copy=True)
    if not copy:
        out[mask] = ring_fill(z, mask)
    out[dest] = zv[dest]
    return (out, dest)

def height_contours(h, level, eps=0.4):
    """Контуры уровня: список (точки в координатах сетки (gx,gy), замкнут?)."""
    h = np.asarray(h, float)
    m = h >= level
    code = m[:-1, :-1].astype(int) | m[:-1, 1:].astype(int) << 1 | m[1:, 1:].astype(int) << 2 | m[1:, :-1].astype(int) << 3
    adj = {}

    def link(a, b):
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for j, i in np.argwhere((code != 0) & (code != 15)):
        j, i = (int(j), int(i))
        edges = [((0, j, i), m[j, i] != m[j, i + 1]), ((1, j, i + 1), m[j, i + 1] != m[j + 1, i + 1]), ((0, j + 1, i), m[j + 1, i] != m[j + 1, i + 1]), ((1, j, i), m[j, i] != m[j + 1, i])]
        cr = [e for e, f in edges if f]
        if len(cr) == 2:
            link(cr[0], cr[1])
        elif len(cr) == 4:
            e0, e1, e2, e3 = (e for e, _ in edges)
            center_in = (h[j, i] + h[j, i + 1] + h[j + 1, i] + h[j + 1, i + 1]) / 4 >= level
            if center_in == bool(m[j, i]):
                link(e0, e1)
                link(e2, e3)
            else:
                link(e0, e3)
                link(e1, e2)

    def pt(e):
        k, j, i = e
        if k == 0:
            a, b = (h[j, i], h[j, i + 1])
            return (i + (level - a) / (b - a), float(j))
        a, b = (h[j, i], h[j + 1, i])
        return (float(i), j + (level - a) / (b - a))
    visited, chains = (set(), [])

    def walk(start):
        chain, cur = ([start], start)
        visited.add(start)
        while True:
            nxt = [e for e in adj[cur] if e not in visited]
            if not nxt:
                return chain
            cur = nxt[0]
            visited.add(cur)
            chain.append(cur)
    for e in list(adj):
        if len(adj[e]) == 1 and e not in visited:
            chains.append((walk(e), False))
    for e in list(adj):
        if e not in visited:
            ch = walk(e)
            chains.append((ch, len(ch) > 2 and ch[0] in adj[ch[-1]]))
    out = []
    for ch, closed in chains:
        pts = [pt(e) for e in ch]
        if len(pts) < 2:
            continue
        pts = rdp(pts + ([pts[0]] if closed else []), eps)
        if closed:
            pts = pts[:-1]
        if len(pts) >= (3 if closed else 2):
            out.append((pts, closed))
    return out

def hillshade(h, azimuth=45.0, elevation=40.0, relief=25.0):
    gy, gx = np.gradient(np.asarray(h, float))
    nx_, ny_, nz_ = (-gx * relief, gy * relief, np.ones_like(gx))
    norm = np.sqrt(nx_ ** 2 + ny_ ** 2 + nz_ ** 2)
    az, el = (math.radians(azimuth), math.radians(elevation))
    lx, ly, lz = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    shade = np.clip((nx_ * lx + ny_ * ly + nz_ * lz) / norm, 0, 1)
    return (np.clip(0.12 + 0.88 * shade, 0, 1) * 255).astype(np.uint8)

def demo_relief(n=160):
    y, x = np.mgrid[0:n, 0:n] / (n - 1.0)
    u, v = (x - 0.5, y - 0.5)
    h = np.clip(1 - ((u / 0.38) ** 2 + (v / 0.22) ** 2), 0, 1) ** 0.6
    return np.clip(h + 0.04 * np.sin(40 * (u * 0.6 + v)) * (h > 0), 0, 1.2)
FORMAT, VERSION, EXT = ('StoneReliefStudioPro', 3, '.srelief')

def pack_project(path, meta, photo=None, photo_name=None, arrays=None):
    meta = dict(meta, format=FORMAT, version=VERSION, photo_file=None, arrays=sorted((arrays or {}).keys()))
    if photo:
        meta['photo_file'] = 'photo' + (os.path.splitext(photo_name or '')[1].lower() or '.png')
    tmp = path + '.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('project.json', json.dumps(meta, ensure_ascii=False, indent=1))
        if photo:
            zf.writestr(meta['photo_file'], photo)
        if arrays:
            buf = io.BytesIO()
            np.savez_compressed(buf, **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()})
            zf.writestr('arrays.npz', buf.getvalue())
    os.replace(tmp, path)

def unpack_project(path):
    with zipfile.ZipFile(path) as zf:
        if 'project.json' not in zf.namelist():
            raise ValueError('Это не файл проекта Stone Relief.')
        meta = json.loads(zf.read('project.json').decode('utf-8'))
        if meta.get('format') != FORMAT:
            raise ValueError('Неизвестный формат файла.')
        if int(meta.get('version', 0)) > VERSION:
            raise ValueError('Проект создан в более новой версии.')
        photo = zf.read(meta['photo_file']) if meta.get('photo_file') else None
        arrays = {}
        if meta.get('arrays'):
            with np.load(io.BytesIO(zf.read('arrays.npz'))) as d:
                arrays = {k: d[k].astype(np.float64) for k in d.files}
    return (meta, photo, arrays)
from PySide6.QtCore import Qt, QPointF, QRectF, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QBrush, QColor, QIcon, QImage, QKeySequence, QMouseEvent, QPainter, QPainterPath, QPainterPathStroker, QPen, QPixmap, QPolygonF, QShortcut
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout, QGraphicsItem, QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QHBoxLayout, QLabel, QListWidget, QMainWindow, QMenu, QMessageBox, QPushButton, QScrollArea, QSlider, QSpinBox, QInputDialog, QRadioButton, QButtonGroup, QTabWidget, QTextBrowser, QToolBar, QToolButton, QVBoxLayout, QWidget
APP = 'Stone Relief Studio Pro'

def glyph_icon(ch, size=30, color='#eeeeee'):
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.TextAntialiasing)
    p.setPen(QColor(color))
    f = p.font()
    f.setPointSize(int(size * 0.5))
    p.setFont(f)
    p.drawText(pm.rect(), Qt.AlignCenter, ch)
    p.end()
    return QIcon(pm)

def build_qpath(data):
    path = QPainterPath()
    if data.nodes:
        path.moveTo(data.nodes[0].x, data.nodes[0].y)
        for i in range(data.seg_count()):
            p0, p1, p2, p3, st = data.seg_ctrl(i)
            if st:
                path.lineTo(p3[0], p3[1])
            else:
                path.cubicTo(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
    return path

def gray_qimage(a):
    a = np.ascontiguousarray(a, dtype=np.uint8)
    h, w = a.shape
    return QImage(a.data, w, h, w, QImage.Format_Grayscale8).copy()

def rgb_qimage(a):
    a = np.ascontiguousarray(a, dtype=np.uint8)
    h, w, _ = a.shape
    return QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()
_SEQ = itertools.count()

class VectorItem(QGraphicsPathItem):
    """Тонкий чёрный вектор (линии и кривые Безье), узлы показываются при выделении."""

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.seq = next(_SEQ)
        self.overlay = None
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
        self.setZValue(10)
        self.refresh()

    def refresh(self):
        self.prepareGeometryChange()
        self.setPath(build_qpath(self.data))
        sel = self.isSelected()
        pen = QPen(QColor('#0a58ff') if sel else QColor('#000000'), 1.2 if sel else 1.0)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(0, 0, 0, 18)) if sel and self.data.closed else QBrush(Qt.NoBrush))
        self.update()

    def scene_points(self):
        p = self.pos()
        return [(x + p.x(), y + p.y()) for x, y in self.data.flatten()]

    def _zoom(self):
        sc = self.scene()
        views = sc.views() if sc else []
        return views[0].transform().m11() if views else 1.0

    def boundingRect(self):
        return self.path().boundingRect().adjusted(-200, -200, 200, 200)

    def shape(self):
        st = QPainterPathStroker()
        st.setWidth(max(9.0 / max(self._zoom(), 1e-06), 1.0))
        return st.createStroke(self.path())

    def paint(self, painter, option, widget=None):
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        painter.drawPath(self.path())

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.refresh()
            self.sync_overlay(bool(value))
        return super().itemChange(change, value)

    def sync_overlay(self, selected):
        if selected and self.overlay is None and (self.scene() is not None):
            self.overlay = NodeOverlay(self)
        elif not selected and self.overlay is not None:
            if self.scene() is not None:
                self.scene().removeItem(self.overlay)
            self.overlay = None

class NodeOverlay(QGraphicsItem):
    """Узлы и ручки Безье выделенного вектора: перетаскивание, двойной щелчок — гладкий/острый."""

    def __init__(self, vec):
        super().__init__(vec)
        self.vec = vec
        self.drag = None
        self.keyboard_handle = None
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setZValue(30)

    def handles(self):
        out = []
        for i, n in enumerate(self.vec.data.nodes):
            out.append(('n', i, (n.x, n.y)))
            if n.hin is not None:
                out.append(('i', i, n.hin))
            if n.hout is not None:
                out.append(('o', i, n.hout))
        return out

    def _zoom(self):
        return self.vec._zoom()

    def boundingRect(self):
        return self.vec.boundingRect()

    def shape(self):
        path = QPainterPath()
        r = 8.0 / max(self._zoom(), 1e-06)
        for _, _, (x, y) in self.handles():
            path.addEllipse(QPointF(x, y), r, r)
        return path

    def paint(self, painter, option, widget=None):
        z = max(painter.worldTransform().m11(), 1e-06)
        r = 4.0 / z
        n = self.vec.data.nodes
        pen = QPen(QColor('#777777'), 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        for i, nd in enumerate(n):
            for h in (nd.hin, nd.hout):
                if h is not None:
                    painter.drawLine(QPointF(nd.x, nd.y), QPointF(h[0], h[1]))
        for kind, i, (x, y) in self.handles():
            pen = QPen(QColor('#222222'), 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            if kind == 'n':
                if self.keyboard_handle == ('n', i):
                    painter.setPen(QPen(QColor('#ff7a00'), 2))
                    painter.drawEllipse(QPointF(x, y), r * 1.9, r * 1.9)
                painter.setBrush(QBrush(QColor('#ffd25b') if self.vec.data.is_smooth(i) else QColor('#ffffff')))
                painter.drawRect(QRectF(x - r, y - r, 2 * r, 2 * r))
            else:
                painter.setBrush(QBrush(QColor('#4aa3ff')))
                painter.drawEllipse(QPointF(x, y), r * 0.8, r * 0.8)

    def _pick(self, pos):
        tol = 9.0 / max(self._zoom(), 1e-06)
        best, bd = (None, tol)
        for kind, i, (x, y) in self.handles():
            d = math.hypot(x - pos.x(), y - pos.y())
            if d <= bd:
                bd, best = (d, (kind, i))
        return best

    def mousePressEvent(self, e):
        h = self._pick(e.pos())
        if h is None:
            e.ignore()
            return
        self.drag = h
        self.keyboard_handle = h
        e.accept()

    def mouseMoveEvent(self, e):
        if not self.drag:
            return
        kind, i = self.drag
        views = self.scene().views()
        sp = self.mapToScene(e.pos())
        if views and hasattr(views[0], 'snap_point') and (kind == 'n'):
            sp = views[0].snap_point(sp, exclude=self.vec)
        p = self.mapFromScene(sp)
        n = self.vec.data.nodes[i]
        if kind == 'n':
            dx, dy = (p.x() - n.x, p.y() - n.y)
            n.x, n.y = (p.x(), p.y())
            if n.hin is not None:
                n.hin = (n.hin[0] + dx, n.hin[1] + dy)
            if n.hout is not None:
                n.hout = (n.hout[0] + dx, n.hout[1] + dy)
        else:
            moved, other = ('hin', 'hout') if kind == 'i' else ('hout', 'hin')
            setattr(n, moved, (p.x(), p.y()))
            o = getattr(n, other)
            if o is not None and (not e.modifiers() & Qt.AltModifier):
                vx, vy = (p.x() - n.x, p.y() - n.y)
                ln = math.hypot(vx, vy)
                ol = math.hypot(o[0] - n.x, o[1] - n.y)
                if ln > 1e-09:
                    setattr(n, other, (n.x - vx / ln * ol, n.y - vy / ln * ol))
        self.vec.refresh()
        self.update()
        e.accept()

    def mouseReleaseEvent(self, e):
        self.drag = None
        e.accept()

    def mouseDoubleClickEvent(self, e):
        h = self._pick(e.pos())
        if h and h[0] == 'n':
            d = self.vec.data
            d.make_corner(h[1]) if d.is_smooth(h[1]) else d.make_smooth(h[1])
            self.vec.refresh()
            self.update()
        e.accept()

class CtrlPoint(QGraphicsEllipseItem):
    """Управляющая точка деформации: dh — смещение высоты в мм."""

    def __init__(self, x, y, dh):
        super().__init__(-7, -7, 14, 14)
        self.setPos(x, y)
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIgnoresTransformations)
        self.setZValue(25)
        self.label = QGraphicsSimpleTextItem('', self)
        self.label.setPos(8, -18)
        self.label.setBrush(QBrush(QColor('#111111')))
        self.set_dh(dh)

    def set_dh(self, dh):
        self.dh = float(dh)
        col = QColor('#2fbf4f' if dh >= 0 else '#e04040')
        self.setBrush(QBrush(col))
        pen = QPen(QColor('#111111'), 1.5 if self.isSelected() else 1)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.label.setText(f'{dh:+g}')

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.set_dh(self.dh)
        return super().itemChange(change, value)

class SectionEditor(QWidget):
    """Рисование сечения: кривая «положение 0…1 → доля смещения 0…1»."""
    changed = Signal()
    PRESETS = {'Прямой наклон': [(0, 0), (1, 1)], 'Плавный переход (S)': [(0, 0), (0.3, 0.1), (0.7, 0.9), (1, 1)], 'Выпуклая дуга': [(0, 0), (0.35, 0.75), (1, 1)], 'Вогнутая дуга': [(0, 0), (0.65, 0.25), (1, 1)], 'Купол': [(0, 0), (0.4, 0.88), (1, 1)], 'Арка (холм) — для выдавливания': [(0, 0), (0.5, 1), (1, 0)], 'Валик (округлый)': [(0, 0), (0.1, 0.6), (0.5, 1), (0.9, 0.6), (1, 0)], 'Ступень': [(0, 0), (0.4, 0), (0.6, 1), (1, 1)]}
    YMIN, YMAX = (-0.2, 1.2)

    def __init__(self):
        super().__init__()
        self.setFixedSize(270, 140)
        self.pts = [list(p) for p in self.PRESETS['Прямой наклон']]
        self.drag = None
        self.setToolTip('ЛКМ по пустому месту — добавить точку; тянуть — менять; ПКМ по точке — удалить.')

    def points(self):
        return [tuple(p) for p in self.pts]

    def set_preset(self, name):
        self.pts = [list(p) for p in self.PRESETS[name]]
        self.update()
        self.changed.emit()

    def _to_px(self, x, y):
        w, h = (self.width(), self.height())
        return (10 + x * (w - 20), h - 10 - (y - self.YMIN) / (self.YMAX - self.YMIN) * (h - 20))

    def _from_px(self, px, py):
        w, h = (self.width(), self.height())
        return ((px - 10) / (w - 20), self.YMIN + (h - 10 - py) / (h - 20) * (self.YMAX - self.YMIN))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor('#ffffff'))
        p.setPen(QPen(QColor('#cccccc'), 1))
        for v in (0, 1):
            a, b = (self._to_px(0, v), self._to_px(1, v))
            p.drawLine(QPointF(*a), QPointF(*b))
        xs, ys = section_curve(self.points())
        poly = QPolygonF([QPointF(*self._to_px(x, y)) for x, y in zip(xs, ys)])
        p.setPen(QPen(QColor('#111111'), 1.6))
        p.drawPolyline(poly)
        p.setBrush(QBrush(QColor('#ffd25b')))
        for x, y in self.pts:
            px, py = self._to_px(x, y)
            p.drawRect(QRectF(px - 4, py - 4, 8, 8))

    def _hit(self, pos):
        for i, (x, y) in enumerate(self.pts):
            px, py = self._to_px(x, y)
            if math.hypot(px - pos.x(), py - pos.y()) < 9:
                return i
        return None

    def mousePressEvent(self, e):
        i = self._hit(e.position())
        if e.button() == Qt.RightButton:
            if i is not None and 0 < i < len(self.pts) - 1:
                self.pts.pop(i)
                self.update()
                self.changed.emit()
            return
        if i is None:
            x, y = self._from_px(e.position().x(), e.position().y())
            if 0.02 < x < 0.98:
                self.pts.append([x, max(self.YMIN, min(self.YMAX, y))])
                self.pts.sort(key=lambda q: q[0])
                i = next((k for k, q in enumerate(self.pts) if q[0] == x))
        self.drag = i
        self.update()
        self.changed.emit()

    def mouseMoveEvent(self, e):
        if self.drag is None:
            return
        x, y = self._from_px(e.position().x(), e.position().y())
        y = max(self.YMIN, min(self.YMAX, y))
        if 0 < self.drag < len(self.pts) - 1:
            lo, hi = (self.pts[self.drag - 1][0] + 0.02, self.pts[self.drag + 1][0] - 0.02)
            self.pts[self.drag][0] = max(lo, min(hi, x))
        self.pts[self.drag][1] = y
        self.update()
        self.changed.emit()

    def mouseReleaseEvent(self, e):
        self.drag = None

class ShapeProfileView(QWidget):
    """Поперечный профиль формы (чёрное поле «Редактора формы»)."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(190, 190)
        self.kind, self.angle = (0, 45)

    def set_params(self, kind, angle):
        self.kind, self.angle = (kind, angle)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor('#0b0b0b'))
        w, h = (self.width(), self.height())
        xs = np.linspace(-1, 1, 121)
        y = shape_profile(['round', 'angular', 'plateau'][self.kind], self.angle, 1 - np.abs(xs), 1.0)
        y = y / max(float(np.abs(y).max()), 1e-06)
        base_y, amp = (h * 0.62, h * 0.45)
        p.setPen(QPen(QColor('#dddddd'), 1))
        p.drawLine(8, int(base_y), w - 8, int(base_y))
        p.setPen(QPen(QColor('#ffd25b'), 2))
        p.drawPolyline(QPolygonF([QPointF(8 + (x + 1) / 2 * (w - 16), base_y - v * amp) for x, v in zip(xs, y)]))
        p.end()

class VectorView(QGraphicsView):

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor('#6f6f6f'))
        self.setFocusPolicy(Qt.StrongFocus)
        self.mode = 'select'
        self.K = 1.0
        self.cw = self.ch = 0.0
        self.canvas_item = self.photo_item = None
        self.snap_nodes = self.snap_lines = self.snap_grid = False
        self.show_grid = False
        self.grid_mm = 10.0
        self.draw_nodes, self.preview, self._drag_node, self._press = ([], None, None, None)
        self._pan = None
        self._pending = None
        self._suppress_ctx = False
        self.keyboard_nudge_mm = 0.1
        self.set_mode('select')

    def set_canvas(self, w, h, K):
        self.cw, self.ch, self.K = (w, h, K)
        if self.canvas_item is None:
            self.canvas_item = self.scene_obj.addRect(0, 0, w, h, QPen(Qt.NoPen), QBrush(QColor('#ffffff')))
            self.canvas_item.setZValue(-200)
            self.photo_item = self.scene_obj.addPixmap(QPixmap())
            self.photo_item.setZValue(-100)
        else:
            self.canvas_item.setRect(0, 0, w, h)
        self.scene_obj.setSceneRect(QRectF(-w, -h, 3 * w, 3 * h))
        self.fit_all()

    def set_photo(self, pixmap):
        self.photo_item.setPixmap(pixmap if pixmap is not None else QPixmap())

    def fit_all(self):
        if self.cw > 0:
            self.fitInView(QRectF(0, 0, self.cw, self.ch).adjusted(-20, -20, 20, 20), Qt.KeepAspectRatio)

    def vectors(self):
        return sorted((i for i in self.scene_obj.items() if isinstance(i, VectorItem)), key=lambda v: v.seq)

    def ctrl_points(self):
        return [i for i in self.scene_obj.items() if isinstance(i, CtrlPoint)]

    def set_mode(self, mode):
        self.cancel_draw()
        self.mode = mode
        self.setDragMode(QGraphicsView.RubberBandDrag if mode == 'select' else QGraphicsView.NoDrag)
        self.setCursor(Qt.ArrowCursor if mode == 'select' else Qt.CrossCursor)
        if mode != 'select':
            self.setFocus()

    def cancel_draw(self):
        self.draw_nodes = []
        self._drag_node = None
        if self.preview is not None:
            self.scene_obj.removeItem(self.preview)
            self.preview = None

    def snap_point(self, p, exclude=None):
        tol = 10.0 / max(self.transform().m11(), 1e-06)
        best, bd = (None, tol)
        if self.snap_nodes or self.snap_lines:
            for v in self.vectors():
                if v is exclude:
                    continue
                if self.snap_nodes:
                    for n in v.data.nodes:
                        d = math.hypot(n.x - p.x(), n.y - p.y())
                        if d < bd:
                            bd, best = (d, QPointF(n.x, n.y))
            if best is None and self.snap_lines:
                for v in self.vectors():
                    if v is exclude:
                        continue
                    r = v.data.nearest_segment((p.x(), p.y()))
                    if r and r[1] < bd:
                        bd, best = (r[1], QPointF(*r[3]))
        if best is not None:
            return best
        if self.snap_grid:
            s = self.grid_mm * self.K
            if s > 0:
                return QPointF(round(p.x() / s) * s, round(p.y() / s) * s)
        return p

    def drawForeground(self, painter, rect):
        if not self.show_grid or self.cw <= 0:
            return
        s = self.grid_mm * self.K
        if s * self.transform().m11() < 6:
            return
        pen = QPen(QColor(0, 0, 0, 45), 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        x = 0.0
        while x <= self.cw + 1e-06:
            painter.drawLine(QPointF(x, 0), QPointF(x, self.ch))
            x += s
        y = 0.0
        while y <= self.ch + 1e-06:
            painter.drawLine(QPointF(0, y), QPointF(self.cw, y))
            y += s

    def _update_preview(self, cursor=None):
        nodes = [n.copy() for n in self.draw_nodes]
        if cursor is not None and nodes:
            nodes.append(Node(cursor.x(), cursor.y()))
        if self.preview is None:
            self.preview = QGraphicsPathItem()
            pen = QPen(QColor('#000000'), 1, Qt.DashLine)
            pen.setCosmetic(True)
            self.preview.setPen(pen)
            self.preview.setZValue(40)
            self.scene_obj.addItem(self.preview)
        self.preview.setPath(build_qpath(PathData(nodes, False)))

    def finish_draw(self, closed):
        nodes = self.draw_nodes
        if len(nodes) < (3 if closed else 2):
            self.cancel_draw()
            return
        snap = self.owner.snapshot()
        data = PathData(nodes, closed)
        self.cancel_draw()
        item = VectorItem(data)
        self.scene_obj.addItem(item)
        self.owner.push_history('Рисование вектора', snap)
        self.owner.set_tool('select')
        self.scene_obj.clearSelection()
        item.setSelected(True)
        self.owner.after_change()

    def mousePressEvent(self, e):
        self._suppress_ctx = False
        if e.button() == Qt.MiddleButton:
            self._pan = e.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        pos = self.mapToScene(e.position().toPoint())
        if self.mode == 'draw':
            if e.button() == Qt.RightButton:
                self._suppress_ctx = True
                self.finish_draw(False)
                return
            if e.button() == Qt.LeftButton:
                sp = self.snap_point(pos)
                if len(self.draw_nodes) >= 3:
                    first = self.mapFromScene(QPointF(self.draw_nodes[0].x, self.draw_nodes[0].y))
                    if (e.position() - QPointF(first)).manhattanLength() <= 12:
                        self.finish_draw(True)
                        return
                if not self.draw_nodes or math.hypot(sp.x() - self.draw_nodes[-1].x, sp.y() - self.draw_nodes[-1].y) > 1e-06:
                    node = Node(sp.x(), sp.y())
                    self.draw_nodes.append(node)
                    self._drag_node = node
                    self._press = e.position()
                self._update_preview(sp)
            return
        if self.mode == 'ctrl' and e.button() == Qt.LeftButton:
            self.owner.add_ctrl_point(self.snap_point(pos))
            return
        if e.button() == Qt.LeftButton:
            self._pending = (self.owner.snapshot(), self.owner.state_sig_light())
            if e.modifiers() & Qt.ShiftModifier:
                try:
                    e = QMouseEvent(e.type(), e.position(), e.globalPosition(), e.button(), e.buttons(), e.modifiers() | Qt.ControlModifier)
                except Exception:
                    pass
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if self.mode == 'draw' and self.draw_nodes:
            pos = self.mapToScene(e.position().toPoint())
            node = self._drag_node
            if node is not None and e.buttons() & Qt.LeftButton and (self._press is not None) and ((e.position() - self._press).manhattanLength() > 5):
                node.hout = (pos.x(), pos.y())
                node.hin = (2 * node.x - pos.x(), 2 * node.y - pos.y())
                self._update_preview(None)
            else:
                self._update_preview(self.snap_point(pos))
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self._pan = None
            self.setCursor(Qt.ArrowCursor if self.mode == 'select' else Qt.CrossCursor)
            return
        if self.mode == 'draw':
            self._drag_node = None
            return
        super().mouseReleaseEvent(e)
        for v in self.vectors():
            if v.pos() != QPointF(0, 0):
                v.data.translate(v.pos().x(), v.pos().y())
                v.setPos(0, 0)
                v.refresh()
        pend, self._pending = (self._pending, None)
        if pend and e.button() == Qt.LeftButton and (self.owner.state_sig_light() != pend[1]):
            self.owner.push_history('Правка вектора / перемещение', pend[0])
            self.owner.after_change()

    def mouseDoubleClickEvent(self, e):
        if self.mode == 'draw':
            self.finish_draw(False)
            return
        super().mouseDoubleClickEvent(e)

    def wheelEvent(self, e):
        f = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        z = self.transform().m11() * f
        if 0.005 < z < 400:
            self.scale(f, f)
        e.accept()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            if getattr(self.owner, '_sw_pending_pick', None):
                self.owner._sw_pending_pick = None
                self.owner.open_sweep_dialog()
                return
            self.owner.escape()
            return
        if e.key() in (Qt.Key_Return, Qt.Key_Enter) and getattr(self.owner, '_sw_pending_pick', None):
            self.owner.sw_complete_pick()
            return
        if e.key() in (Qt.Key_Return, Qt.Key_Enter) and self.mode == 'draw':
            self.finish_draw(False)
            return
        vecs = [v for v in self.scene_obj.selectedItems() if isinstance(v, VectorItem)]
        if self.mode == 'select' and len(vecs) == 1 and vecs[0].data.nodes:
            v = vecs[0]
            ov = v.overlay
            if ov is not None:
                if e.key() in (Qt.Key_Tab, Qt.Key_Backtab):
                    old = ov.keyboard_handle[1] if ov.keyboard_handle else -1 if e.key() == Qt.Key_Tab else 0
                    delta = -1 if e.key() == Qt.Key_Backtab or e.modifiers() & Qt.ShiftModifier else 1
                    ov.keyboard_handle = ('n', (old + delta) % len(v.data.nodes))
                    ov.update()
                    self.owner.statusBar().showMessage(f'Узел {ov.keyboard_handle[1] + 1}/{len(v.data.nodes)}. Стрелки — переместить, Shift — 1 мм, Ctrl — 0.01 мм.')
                    e.accept()
                    return
                dirs = {Qt.Key_Left: (-1, 0), Qt.Key_Right: (1, 0), Qt.Key_Up: (0, -1), Qt.Key_Down: (0, 1)}
                if e.key() in dirs:
                    if ov.keyboard_handle is None:
                        ov.keyboard_handle = ('n', 0)
                    kind, idx = ov.keyboard_handle
                    if kind != 'n':
                        kind = 'n'
                        ov.keyboard_handle = ('n', idx)
                    step = 0.01 if e.modifiers() & Qt.ControlModifier else 1.0 if e.modifiers() & Qt.ShiftModifier else self.keyboard_nudge_mm
                    dx, dy = dirs[e.key()]
                    dx *= step * self.K
                    dy *= step * self.K
                    snap = self.owner.snapshot()
                    node = v.data.nodes[idx]
                    node.x += dx
                    node.y += dy
                    if node.hin is not None:
                        node.hin = (node.hin[0] + dx, node.hin[1] + dy)
                    if node.hout is not None:
                        node.hout = (node.hout[0] + dx, node.hout[1] + dy)
                    v.refresh()
                    ov.update()
                    self.owner.push_history('Перемещение узла', snap)
                    self.owner.after_change()
                    e.accept()
                    return
        if e.key() == Qt.Key_Delete:
            if len(vecs) == 1 and vecs[0].overlay and vecs[0].overlay.keyboard_handle:
                v = vecs[0]
                index = v.overlay.keyboard_handle[1]
                snap = self.owner.snapshot()
                if v.data.remove_node(index):
                    v.overlay.keyboard_handle = ('n', min(index, len(v.data.nodes) - 1))
                    v.refresh()
                    v.overlay.update()
                    self.owner.push_history('Удаление узла', snap)
                    self.owner.after_change()
                return
            self.owner.delete_selected()
            return
        super().keyPressEvent(e)

    def focusNextPrevChild(self, next):
        if self.mode == 'select' and any((isinstance(v, VectorItem) for v in self.scene_obj.selectedItems())):
            return False
        return super().focusNextPrevChild(next)

    def contextMenuEvent(self, e):
        if self._suppress_ctx or self.mode != 'select':
            self._suppress_ctx = False
            e.accept()
            return
        self.owner.vector_context_menu(self.mapToScene(e.pos()), e.globalPos())
MATERIALS = [('Белый камень', '#ececec'), ('Жёлтый камень', '#d8b25a'), ('Серый камень', '#8f8f8f'), ('Чёрный камень', '#2a2a2a')]
TIPS = {'select': 'Выделение (ЛКМ, рамка, Shift — несколько). Esc — снять выделение', 'draw': 'Рисовать вектор: ЛКМ — точка (тянуть — кривая), ПКМ — закончить, клик по первой — замкнуть, Esc — отмена', 'ctrl': 'Управляющая точка деформации: ЛКМ — поставить'}

def busy(fn):
    nargs = fn.__code__.co_argcount - 1

    def wrapper(self, *a, **k):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            return fn(self, *a[:nargs], **k)
        finally:
            QApplication.restoreOverrideCursor()
    wrapper.__name__ = fn.__name__
    return wrapper

class Studio(QMainWindow):

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, v):
        self._z = None if v is None else np.asarray(v, dtype=np.float32)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP)
        self.resize(1560, 940)
        self.z = None
        self.stl_mesh = None
        self.photo_pil = self.photo_bytes = self.photo_name = self.photo_arr = None
        self.actor = self.sun_actor = None
        self.sun_on = True
        self.undo_stack, self.redo_stack = ([], [])
        self.preview_active = False
        self.preview_z = self.preview_mask = None
        self.project_path = None
        self._saved_sig = None
        self._gid = 0
        self._in_group_sync = False
        self._first_show = True
        self._base_prev = 5.0
        self.K = 1.0
        self.canvas_w = self.canvas_h = 0.0
        self.tmpdir = tempfile.mkdtemp(prefix='srs_')
        self.book_dir = None
        self.book_data = None
        self.rail_items = [None, None]
        self.profile_item = None
        self.relief_dialog = None
        self.sweep_dialog = None
        self.sw_rails = [None, None]
        self.sw_zb = None
        self.sw_sections = []
        self.sh_preview_flag = False
        self._v_key = None
        self._v_cache = None
        self.setAcceptDrops(True)
        self.setStyleSheet('QMainWindow,QWidget{background:#2b2b2b;color:#eeeeee}QPushButton,QToolButton{background:#444;border:1px solid #5a5a5a;border-radius:3px;padding:4px}QToolButton:checked{background:#2f6fd0}QToolButton:hover,QPushButton:hover{background:#555}QDoubleSpinBox,QSpinBox,QComboBox{background:#3d3d3d;color:#fff;padding:2px}QToolTip{background:#ffffe0;color:#111;border:1px solid #999}QTabBar::tab{padding:6px 10px;background:#3a3a3a}QTabBar::tab:selected{background:#555}')
        self._build_ui()
        self.rebuild_canvas()
        self.set_tool('select')
        self._saved_sig = self.state_sig()
        self.update_light()
        self.statusBar().showMessage('Готово. F1 — книга инструкций. Всё работает без интернета.')

    def _spin(self, v, lo, hi, dec=1, suffix=' мм', step=1.0):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setValue(v)
        s.setSingleStep(step)
        if suffix:
            s.setSuffix(suffix)
        return s

    def _slider(self, lo, hi, v):
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(v)
        return s

    def _act(self, menu, text, slot, shortcut=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    def _build_ui(self):
        from pyvistaqt import QtInteractor
        top = QToolBar('Размеры проекта')
        top.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, top)
        self.sp_w = self._spin(500, 10, 3000, 1, ' мм', 10)
        self.sp_h = self._spin(100, 10, 3000, 1, ' мм', 10)
        self.sp_base = self._spin(5, 0.5, 200, 2)
        self.sp_depth = self._spin(10, 0.5, 200, 2)
        self.sp_step = self._spin(0.2, 0.05, 2.0, 2, ' мм', 0.05)
        self.sp_grid = self._spin(10, 0.5, 500, 1, ' мм', 1)
        for label, w, tip in [('Ширина', self.sp_w, 'Ширина проекта в мм'), ('Высота', self.sp_h, 'Высота проекта в мм'), ('Основание', self.sp_base, 'Общая толщина основания в мм (можно менять в любой момент)'), ('Макс. рельеф', self.sp_depth, 'Максимальная высота рельефа в мм (для нового рельефа из фото)'), ('Шаг сетки', self.sp_step, 'Шаг рельефа в мм (0,2 — для станка). Меньше шаг — мельче детали, но тяжелее и медленнее.'), ('Сетка', self.sp_grid, 'Шаг сетки привязки в мм')]:
            top.addWidget(QLabel(f' {label} '))
            w.setToolTip(tip)
            top.addWidget(w)
        self.sp_w.valueChanged.connect(self.rebuild_canvas)
        self.sp_h.valueChanged.connect(self.rebuild_canvas)
        self.sp_base.valueChanged.connect(self.on_base_changed)
        self.sp_grid.valueChanged.connect(self.on_grid_changed)
        top.addSeparator()
        self.chk_invert = QCheckBox('Углублённый')
        self.chk_invert.setToolTip('Тёмное выше, светлое ниже (инверсия фото)')
        self.sp_blur = self._spin(1, 0, 15, 1, ' px', 0.5)
        self.sp_blur.setToolTip('Сглаживание фото перед созданием рельефа')
        top.addWidget(self.chk_invert)
        top.addWidget(self.sp_blur)
        for text, tip, slot in [('Открыть фото', 'Открыть фотографию (JPG, PNG, BMP, TIFF, WEBP…) как подложку — Ctrl+I', self.open_photo), ('Рельеф из фото', 'Создать 3D-рельеф из фотографии (если фото не открыто — предложит выбрать)', self.create_relief), ('Плоская плита', 'Создать плоскую плиту для работы без фото', self.create_plate)]:
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            top.addWidget(b)
        left = QToolBar('Инструменты')
        left.setOrientation(Qt.Vertical)
        left.setMovable(False)
        left.setIconSize(left.iconSize() * 1.2)
        self.addToolBar(Qt.LeftToolBarArea, left)
        self.tool_group = QActionGroup(self)
        self.tool_actions = {}
        for key, ch in [('select', '⬚'), ('draw', '✎'), ('ctrl', '◉')]:
            a = QAction(glyph_icon(ch), TIPS[key].split(':')[0], self)
            a.setCheckable(True)
            a.setToolTip(TIPS[key])
            a.setStatusTip(TIPS[key])
            a.triggered.connect(lambda _=False, k=key: self.set_tool(k))
            self.tool_group.addAction(a)
            left.addAction(a)
            self.tool_actions[key] = a
        left.addSeparator()
        for ch, tip, slot in [('⧉', 'Копировать выделенные векторы', self.copy_selected), ('⇋', 'Отразить по горизонтали', lambda: self.mirror_selected('h')), ('⇅', 'Отразить по вертикали', lambda: self.mirror_selected('v')), ('⟳', 'Переместить / повернуть / масштабировать вектор…', self.transform_dialog), ('▣', 'Сгруппировать выделенные', self.group_selected), ('▢', 'Разгруппировать', self.ungroup_selected), ('⚭', 'Соединить два открытых вектора', self.join_selected), ('🗑', 'Удалить выделенное (Delete)', self.delete_selected), ('▤', 'Редактор формы (F12): форма по вектору, обнуление внутри/снаружи, слияние', self.open_relief_editor)]:
            a = QAction(glyph_icon(ch), tip, self)
            a.setToolTip(tip)
            a.setStatusTip(tip)
            a.triggered.connect(slot)
            left.addAction(a)
        left.addSeparator()
        for ch, tip, slot in [('↶', 'Отменить (Ctrl+Z)', self.undo), ('↷', 'Повторить (Ctrl+Y)', self.redo), ('⛶', 'Показать весь холст', lambda: self.view.fit_all())]:
            a = QAction(glyph_icon(ch), tip, self)
            a.setToolTip(tip)
            a.triggered.connect(slot)
            left.addAction(a)
        left.addSeparator()
        self.snap_actions = {}
        for key, ch, tip in [('nodes', '◈', 'Привязка к узлам'), ('lines', '╱', 'Привязка к линиям'), ('grid', '▦', 'Привязка к сетке'), ('showgrid', '#', 'Показать сетку')]:
            a = QAction(glyph_icon(ch), tip, self)
            a.setCheckable(True)
            a.setToolTip(tip)
            a.toggled.connect(lambda on, k=key: self.on_snap(k, on))
            left.addAction(a)
            self.snap_actions[key] = a
        mb = self.menuBar()
        m = mb.addMenu('Файл')
        self._act(m, 'Новый проект', self.new_project, 'Ctrl+N')
        self._act(m, 'Открыть проект…', self.open_project, 'Ctrl+O')
        self._act(m, 'Сохранить проект', self.save_project, 'Ctrl+S')
        self._act(m, 'Сохранить проект как…', self.save_project_as, 'Ctrl+Shift+S')
        m.addSeparator()
        self._act(m, 'Открыть фото (JPG/PNG…)', self.open_photo, 'Ctrl+I')
        self._act(m, 'Импорт DXF…', self.import_dxf)
        self._act(m, 'Импорт SVG…', self.import_svg)
        self._act(m, 'Экспорт SVG…', self.export_svg)
        self._act(m, 'Открыть STL…', self.open_stl)
        self._act(m, 'Экспорт STL…', self.export_stl)
        m.addSeparator()
        self._act(m, 'Выход', self.close, 'Ctrl+Q')
        m = mb.addMenu('Правка')
        self._act(m, 'Отменить', self.undo, 'Ctrl+Z')
        self._act(m, 'Повторить', self.redo, 'Ctrl+Y')
        m = mb.addMenu('Рельефы')
        self._act(m, 'Редактор формы…', self.open_relief_editor, 'F12')
        sub = m.addMenu('Вытягиваемые профили')
        self._act(sub, 'Вытягивание по двум направляющим…', self.open_sweep_dialog)
        for t in ('Выдавливание…', 'Вращение…', 'Поворот…'):
            sub.addAction(t).setEnabled(False)
        self._act(m, 'Создать купол…', self.create_dome)
        m.addSeparator()
        self._act(m, 'Обнулить', lambda: self.relief_op('zero'), 'Shift+Delete')
        self._act(m, 'Сглаживание…', lambda: self.relief_op('smooth'))
        self._act(m, 'Масштаб…', lambda: self.relief_op('scale'))
        self._act(m, 'Инвертировать', lambda: self.relief_op('invert'))
        sub = m.addMenu('Отразить')
        self._act(sub, 'По горизонтали', lambda: self.relief_op('mirror_h'))
        self._act(sub, 'По вертикали', lambda: self.relief_op('mirror_v'))
        self._act(m, 'Смещение…', lambda: self.relief_op('offset'))
        m.addSeparator()
        self._act(m, 'Создать STL-модель…', self.export_stl)
        m = mb.addMenu('Справка')
        self._act(m, 'Книга инструкций', self.open_help, 'F1')
        self._act(m, 'О программе', lambda: QMessageBox.information(self, 'О программе', APP + '\n2D-векторы и 3D-рельефы для обработки камня на ЧПУ.\nРаботает без интернета.'))
        book = QToolButton()
        book.setText('📖')
        book.setToolTip('Книга инструкций (F1)')
        book.setAutoRaise(True)
        book.clicked.connect(self.open_help)
        mb.setCornerWidget(book, Qt.TopRightCorner)
        QShortcut(QKeySequence('Esc'), self, activated=self.escape)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        w2d = QWidget()
        l2d = QVBoxLayout(w2d)
        l2d.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.addWidget(QLabel('Осветление фотографии до белого фона:'))
        self.sl_photo = self._slider(0, 100, 0)
        self.sl_photo.setToolTip('0 — исходное фото, 100 — белый фон. Векторы и рельеф не меняются.')
        self.lb_photo = QLabel('0%')
        self.sl_photo.valueChanged.connect(self.update_photo_view)
        bar.addWidget(self.sl_photo, 1)
        bar.addWidget(self.lb_photo)
        l2d.addLayout(bar)
        self.view = VectorView(self)
        self.scene = self.view.scene_obj
        self.scene.selectionChanged.connect(self._sync_groups)
        self.scene.selectionChanged.connect(self.update_selection_ui)
        l2d.addWidget(self.view, 1)
        self.tabs.addTab(w2d, '2D — фото и векторы')
        w3d = QWidget()
        l3d = QVBoxLayout(w3d)
        l3d.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.cb_mode3d = QComboBox()
        self.cb_mode3d.addItems(['Только просмотр', 'Редактирование 3D'])
        self.cb_mode3d.setToolTip('В режиме «Только просмотр» кнопки изменения рельефа отключены')
        self.cb_mode3d.currentIndexChanged.connect(self.update_edit_enabled)
        row.addWidget(self.cb_mode3d)
        for name, key in [('Сверху', 'top'), ('Спереди', 'front'), ('Слева', 'left'), ('Справа', 'right'), ('Сзади', 'back'), ('Снизу', 'bottom'), ('Под углом', 'iso')]:
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, k=key: self.set_view(k))
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(QLabel('ЛКМ — вращать · колесо — масштаб · Shift+ЛКМ или средняя — сдвиг'))
        l3d.addLayout(row)
        self.plotter = QtInteractor(w3d)
        self.plotter.set_background('#303030')
        self.plotter.add_axes()
        l3d.addWidget(self.plotter, 1)
        self.tabs.addTab(w3d, '3D — рельеф и свет')
        self._build_dock()
        self.update_edit_enabled()

    def _row(self, layout, key, label, widget):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel(label))
        h.addWidget(widget, 1)
        layout.addWidget(w)
        self.rows[key] = w
        return widget

    def _build_dock(self):
        self.rows = {}
        dock = QDockWidget('Инструменты рельефа', self)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        tabs = QTabWidget()
        dock.setWidget(tabs)
        dock.setMinimumWidth(330)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        def page(title):
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            w = QWidget()
            lay = QVBoxLayout(w)
            sc.setWidget(w)
            tabs.addTab(sc, title)
            return lay
        lay = page('Свет 3D')
        self.cb_mat = QComboBox()
        self.cb_mat.addItems([m[0] for m in MATERIALS])
        self.cb_mat.setCurrentIndex(2)
        self.cb_mat.currentIndexChanged.connect(self.on_material_changed)
        self._row(lay, 'mat', 'Материал', self.cb_mat)
        self.l_bright = self._row(lay, 'lb', 'Яркость', self._slider(0, 100, 75))
        self.l_contrast = self._row(lay, 'lc', 'Контраст', self._slider(0, 100, 65))
        self.l_az = self._row(lay, 'la', 'Солнце: поворот', self._slider(-180, 180, 45))
        self.l_el = self._row(lay, 'le', 'Солнце: высота', self._slider(3, 90, 35))
        self.l_bright.valueChanged.connect(self.update_light)
        self.l_az.valueChanged.connect(self.update_light)
        self.l_el.valueChanged.connect(self.update_light)
        self.l_contrast.valueChanged.connect(self.refresh_material)
        self.chk_sun = QCheckBox('Солнце включено')
        self.chk_sun.setChecked(True)
        self.chk_fill = QCheckBox('Дополнительный (заполняющий) свет')
        self.chk_fill.setChecked(True)
        self.chk_icon = QCheckBox('Показывать значок солнца')
        self.chk_shadow = QCheckBox('Тени (медленнее)')
        for c in (self.chk_sun, self.chk_fill, self.chk_icon, self.chk_shadow):
            c.toggled.connect(self.update_light)
            lay.addWidget(c)
        lay.addStretch(1)
        lay = page('Деформация')
        lay.addWidget(QLabel('1. Нарисуйте и выделите замкнутый вектор — это область.\n2. Выберите вид, задайте высоты и сечение.\n3. «Предпросмотр» → «Применить».'))
        btn_sw = QPushButton('Вытягивание по двум направляющим…')
        btn_sw.setToolTip('Окно как в ArtCAM: сечения вытягиваются вдоль двух направляющих (Рельефы → Вытягиваемые профили)')
        btn_sw.clicked.connect(self.open_sweep_dialog)
        lay.addWidget(btn_sw)
        self.d_mode = QComboBox()
        self.d_mode.addItems(['Плавный изгиб (подъём сторон)', 'Волна', 'Вершина', 'Между направляющими (плавный переход)', 'Управляющие точки'])
        self.d_mode.currentIndexChanged.connect(self.update_deform_rows)
        self._row(lay, 'mode', 'Вид', self.d_mode)
        self.d_angle = self._row(lay, 'angle', 'Направление, °', self._spin(0, -360, 360, 0, '°', 15))
        self.d_a = self._row(lay, 'a', 'Сторона A, мм', self._spin(0, -100, 100, 2))
        self.d_b = self._row(lay, 'b', 'Сторона B, мм', self._spin(2, -100, 100, 2))
        self.d_amount = self._row(lay, 'amount', 'Высота, мм', self._spin(2, -100, 100, 2))
        self.d_waves = self._row(lay, 'waves', 'Волн', self._spin(3, 0.5, 50, 1, '', 0.5))
        self.d_apx = self._row(lay, 'apx', 'Вершина X, %', self._spin(50, 0, 100, 0, '%', 5))
        self.d_apy = self._row(lay, 'apy', 'Вершина Y, %', self._spin(50, 0, 100, 0, '%', 5))
        self.d_rad = self._row(lay, 'rad', 'Радиус, %', self._spin(60, 5, 200, 0, '%', 5))
        self.d_hA = self._row(lay, 'hA', 'Высота направляющей A, мм', self._spin(0, -100, 100, 2))
        self.d_hB = self._row(lay, 'hB', 'Высота направляющей B, мм', self._spin(2, -100, 100, 2))
        self.d_sweepH = self._row(lay, 'sweepH', 'Высота профиля, мм', self._spin(3, -100, 100, 2))
        self.d_profsrc = QComboBox()
        self.d_profsrc.addItems(['Сечение: кривая из редактора', 'Сечение: нарисованный вектор'])
        self._row(lay, 'profsrc', 'Сечение', self.d_profsrc)
        for key, kind, text, tip in [('rail1btn', 'rail1', 'Направляющая 1 ← выделенный вектор', 'Выделите открытый вектор и нажмите — это первая граница траектории'), ('rail2btn', 'rail2', 'Направляющая 2 ← выделенный вектор', 'Выделите открытый вектор и нажмите — это вторая граница траектории'), ('profbtn', 'profile', 'Сечение ← выделенный вектор', 'Нарисуйте сечение открытым вектором: от начала к концу — от направляющей 1 к 2, вверх — выше')]:
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _=False, k=kind: self.assign_role(k))
            self._row(lay, key, '', btn)
        self.lb_prof = QLabel('Направляющие и сечение не заданы')
        self._row(lay, 'lbprof', '', self.lb_prof)
        self.d_scaleh = QCheckBox('Масштабировать высоту сечения вместе с шириной между направляющими')
        self.d_scaleh.toggled.connect(self.schedule_preview)
        self.rows['scaleh'] = self.d_scaleh
        lay.addWidget(self.d_scaleh)
        self.d_infl = self._row(lay, 'infl', 'Радиус влияния, мм', self._spin(30, 1, 1000, 1, ' мм', 5))
        self.d_cdh = self._row(lay, 'cdh', 'Высота точки, мм', self._spin(2, -100, 100, 2))
        self.d_cdh.valueChanged.connect(self.set_selected_ctrl_height)
        self.rows['cdh'].setToolTip('Для новых и выделенных управляющих точек')
        lay.addWidget(QLabel('Сечение (форма перехода):'))
        self.section = SectionEditor()
        self.cb_preset = QComboBox()
        self.cb_preset.addItems(list(SectionEditor.PRESETS))
        self.cb_preset.currentTextChanged.connect(self.section.set_preset)
        lay.addWidget(self.cb_preset)
        lay.addWidget(self.section)
        self.d_fixed = QCheckBox('Закрепить границы (иначе — границы двигаются)')
        self.d_fixed.setChecked(True)
        self.d_feather = self._row(lay, 'feather', 'Плавность края, мм', self._spin(5, 0.5, 200, 1, ' мм', 1))
        lay.addWidget(self.d_fixed)
        self.d_combine = QComboBox()
        self.d_combine.addItems(COMBINE_NAMES)
        self.d_combine.setToolTip('Как форма объединяется с существующим рельефом (как в ArtCAM)')
        self._row(lay, 'combine', 'Слияние', self.d_combine)
        self.d_dim = self._row(lay, 'dim', 'Затемнить остальное', self._slider(0, 90, 45))
        for w in (self.d_angle, self.d_a, self.d_b, self.d_amount, self.d_waves, self.d_apx, self.d_apy, self.d_rad, self.d_hA, self.d_hB, self.d_infl, self.d_feather, self.d_sweepH):
            w.valueChanged.connect(self.schedule_preview)
        self.d_fixed.toggled.connect(self.schedule_preview)
        self.d_profsrc.currentIndexChanged.connect(self.schedule_preview)
        self.d_combine.currentIndexChanged.connect(self.schedule_preview)
        self.d_dim.valueChanged.connect(self.schedule_preview)
        self.section.changed.connect(self.schedule_preview)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._refresh_preview)
        self.edit_widgets = []
        for text, slot in [('Предпросмотр', self.preview_deform), ('Применить', self.apply_deform), ('Отмена', self.cancel_preview)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            lay.addWidget(b)
            self.edit_widgets.append(b)
        lay.addStretch(1)
        self.update_deform_rows()
        lay = page('Перемещение')
        lay.addWidget(QLabel('Работает с областью выделенного замкнутого вектора.'))
        self.t_dx = self._row(lay, 'tdx', 'Сдвиг X, мм', self._spin(0, -3000, 3000, 2))
        self.t_dy = self._row(lay, 'tdy', 'Сдвиг Y, мм (вверх +)', self._spin(0, -3000, 3000, 2))
        self.t_ang = self._row(lay, 'tang', 'Поворот, °', self._spin(0, -360, 360, 1, '°', 5))
        self.t_scale = self._row(lay, 'tsc', 'Масштаб, %', self._spin(100, 10, 500, 1, '%', 5))
        self.t_vec = QCheckBox('Перемещать вместе со связанными векторами')
        self.t_vec.setChecked(True)
        lay.addWidget(self.t_vec)
        for text, slot in [('Переместить', lambda: self.do_transform(False, False, False)), ('Копировать в новое место', lambda: self.do_transform(True, False, False)), ('Отразить горизонтально', lambda: self.do_transform(False, True, False)), ('Отразить вертикально', lambda: self.do_transform(False, False, True))]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            lay.addWidget(b)
            self.edit_widgets.append(b)
        lay.addStretch(1)
        lay = page('Контуры по высоте')
        lay.addWidget(QLabel('Строит редактируемые векторы по границе заданного уровня рельефа.'))
        self.c_level = self._row(lay, 'clev', 'Уровень, мм', self._spin(7, 0, 500, 2))
        self.lb_range = QLabel('Высота рельефа: —')
        lay.addWidget(self.lb_range)
        b = QPushButton('Построить контуры')
        b.clicked.connect(self.make_contours)
        lay.addWidget(b)
        lay.addStretch(1)
        lay = page('История')
        self.hist = QListWidget()
        lay.addWidget(self.hist)
        row = QHBoxLayout()
        for text, slot in [('Отменить', self.undo), ('Повторить', self.redo)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        lay.addLayout(row)

    def has_content(self):
        return self.z is not None or self.stl_mesh is not None or self.photo_pil is not None or bool(self.view.vectors())

    def update_lock(self):
        free = not self.has_content()
        for w in (self.sp_w, self.sp_h, self.sp_step):
            w.setEnabled(free)

    def grid_dims(self, W=None, H=None):
        W = W or self.sp_w.value()
        H = H or self.sp_h.value()
        if self.z is not None:
            return (self.z.shape[1], self.z.shape[0])
        nx, ny, used = grid_from_step(W, H, self.sp_step.value())
        if used > self.sp_step.value() * 1.001:
            self.statusBar().showMessage(f'Шаг сетки увеличен до {used:.2f} мм: сетка не должна превышать 8 млн ячеек.')
        return (nx, ny)

    def cell(self):
        ny, nx = self.z.shape
        return (self.sp_w.value() / (nx - 1), self.sp_h.value() / (ny - 1))

    def grid_to_scene(self, gx, gy):
        ny, nx = self.z.shape
        return (gx / (nx - 1) * self.canvas_w, self.canvas_h - gy / (ny - 1) * self.canvas_h)

    def scene_to_grid(self, x, y):
        ny, nx = self.z.shape
        return (x / self.canvas_w * (nx - 1), (self.canvas_h - y) / self.canvas_h * (ny - 1))

    def rebuild_canvas(self, *_, force=False):
        if self.has_content() and (not force):
            return
        W, H = (self.sp_w.value(), self.sp_h.value())
        self.K = 1500.0 / max(W, H)
        self.canvas_w, self.canvas_h = (W * self.K, H * self.K)
        self.view.set_canvas(self.canvas_w, self.canvas_h, self.K)
        self.update_photo_view()

    def on_grid_changed(self, v):
        self.view.grid_mm = v
        self.view.viewport().update()

    def on_snap(self, key, on):
        setattr(self.view, {'nodes': 'snap_nodes', 'lines': 'snap_lines', 'grid': 'snap_grid', 'showgrid': 'show_grid'}[key], on)
        self.view.viewport().update()

    def set_tool(self, tool):
        self.view.set_mode(tool)
        self.tool_actions[tool].setChecked(True)
        self.statusBar().showMessage(TIPS[tool])

    def escape(self):
        if self.view.mode == 'draw' and self.view.draw_nodes:
            self.view.cancel_draw()
        if self.view.mode != 'select':
            self.set_tool('select')
        else:
            self.scene.clearSelection()
        self.view.cancel_draw()

    def update_edit_enabled(self, *_):
        on = self.cb_mode3d.currentIndex() == 1
        for w in getattr(self, 'edit_widgets', []):
            w.setEnabled(on)

    def after_change(self):
        self.update_lock()
        self.update_selection_ui()

    def update_selection_ui(self):
        if self.z is not None:
            ny, nx = self.z.shape
            h = self.z - self.sp_base.value()
            self.lb_range.setText(f'Высота рельефа: {h.min():.2f} … {h.max():.2f} мм')

    def snapshot(self):
        return {'vec': [v.data.to_dict() for v in self.view.vectors()], 'ctrl': [(c.pos().x(), c.pos().y(), c.dh) for c in self.view.ctrl_points()], 'z': self.z, 'base': self.sp_base.value()}

    def state_sig_light(self):
        vec = sorted((json.dumps(v.data.to_dict()) for v in self.view.vectors()))
        ctl = sorted(((round(c.pos().x(), 3), round(c.pos().y(), 3), c.dh) for c in self.view.ctrl_points()))
        return hashlib.md5(json.dumps([vec, ctl]).encode()).hexdigest()

    def state_sig(self):
        h = hashlib.md5(self.state_sig_light().encode())
        h.update(self.photo_bytes or b'')
        if self.z is not None:
            h.update(np.ascontiguousarray(self.z).tobytes())
        h.update(str((self.sp_w.value(), self.sp_h.value(), self.sp_base.value(), self.sp_depth.value(), self.sl_photo.value())).encode())
        return h.hexdigest()

    def push_history(self, label, snap=None):
        self.undo_stack.append((label, snap if snap is not None else self.snapshot()))
        if len(self.undo_stack) > 25:
            self.undo_stack.pop(0)
        self.redo_stack = []
        self.refresh_history()

    def refresh_history(self):
        self.hist.clear()
        for i, (label, _) in enumerate(reversed(self.undo_stack), 1):
            self.hist.addItem(f'{i}. {label}')

    def restore(self, s):
        self.cancel_preview(silent=True)
        for it in self.view.vectors() + self.view.ctrl_points():
            self.scene.removeItem(it)
        for d in s['vec']:
            self.scene.addItem(VectorItem(PathData.from_dict(d)))
        for x, y, dh in s['ctrl']:
            self.scene.addItem(CtrlPoint(x, y, dh))
        self.sp_base.blockSignals(True)
        self.sp_base.setValue(s['base'])
        self.sp_base.blockSignals(False)
        self._base_prev = s['base']
        if s['z'] is not self.z:
            self.z = s['z']
            self.show_mesh(reset=self.z is not None)
        self.after_change()

    def undo(self):
        if not self.undo_stack:
            return
        label, s = self.undo_stack.pop()
        self.redo_stack.append((label, self.snapshot()))
        self.restore(s)
        self.refresh_history()
        self.statusBar().showMessage('Отменено: ' + label)

    def redo(self):
        if not self.redo_stack:
            return
        label, s = self.redo_stack.pop()
        self.undo_stack.append((label, self.snapshot()))
        self.restore(s)
        self.refresh_history()
        self.statusBar().showMessage('Повторено: ' + label)

    def open_photo(self, path=None):
        if not isinstance(path, str):
            path, _ = QFileDialog.getOpenFileName(self, 'Открыть фото', '', 'Изображения (*.jpg *.jpeg *.jpe *.png *.bmp *.gif *.tif *.tiff *.webp);;Все файлы (*.*)')
        if not path:
            return False
        try:
            with open(path, 'rb') as f:
                raw = f.read()
            img = load_image_any(raw)
        except Exception as e:
            QMessageBox.critical(self, 'Не удалось открыть фото', f'{path}\n\n{type(e).__name__}: {e}')
            return False
        if not self.has_content():
            W, H = (self.sp_w.value(), self.sp_h.value())
            ar = img.width / img.height
            if abs(W / H - ar) / ar > 0.02:
                r = QMessageBox.question(self, 'Пропорции фото', f'Подогнать высоту проекта под пропорции фото?\n(ширина останется {W:g} мм, высота станет {W / ar:.1f} мм)')
                if r == QMessageBox.Yes:
                    self.sp_h.setValue(float(min(3000, max(10, round(W / ar, 1)))))
                    self.rebuild_canvas(force=True)
        self.push_history('Открыть фото')
        old = (self.photo_pil, self.photo_bytes, self.photo_name, self.photo_arr)
        self.photo_pil, self.photo_bytes, self.photo_name, self.photo_arr = (img, raw, path, None)
        try:
            self.update_photo_view()
        except Exception as e:
            self.photo_pil, self.photo_bytes, self.photo_name, self.photo_arr = old
            QMessageBox.critical(self, 'Ошибка показа фото', f'{type(e).__name__}: {e}')
            return False
        self.update_lock()
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage(f'Фото открыто: {os.path.basename(path)} ({img.width}×{img.height}). Ползунок над окном 2D осветляет подложку.')
        return True

    def update_photo_view(self, *_):
        self.lb_photo.setText(f'{self.sl_photo.value()}%')
        if self.photo_pil is None or self.canvas_w <= 0:
            self.view.set_photo(None)
            return
        if self.photo_arr is None or self.photo_arr.shape[1] != int(self.canvas_w):
            self.photo_arr = np.asarray(self.photo_pil.resize((max(2, int(self.canvas_w)), max(2, int(self.canvas_h))), LANCZOS), dtype=np.float32)
        f = self.sl_photo.value() / 100.0
        arr = (self.photo_arr * (1 - f) + 255.0 * f).astype(np.uint8)
        self.view.set_photo(QPixmap.fromImage(rgb_qimage(arr)))

    def _check_free_preview(self):
        if self.preview_active:
            self.cancel_preview()

    def create_relief(self):
        if self.photo_pil is None and (not self.open_photo()):
            return
        self._check_free_preview()
        W, H = (self.sp_w.value(), self.sp_h.value())
        nx, ny = self.grid_dims(W, H)
        h = photo_heights(self.photo_bytes, nx, ny, self.sp_blur.value(), self.chk_invert.isChecked())
        self.push_history('Рельеф из фото')
        self.stl_mesh = None
        self.z = self.sp_base.value() + self.sp_depth.value() * h
        self.show_mesh(reset=True)
        self.update_lock()
        self.after_change()
        self.tabs.setCurrentIndex(1)
        note = ''
        if self.photo_pil.width < nx * 0.6:
            note = f' Фото ({self.photo_pil.width} px по ширине) грубее сетки — для шага {W / (nx - 1):.2f} мм нужно ≈{nx} px.'
        self.statusBar().showMessage(f'Рельеф {nx}×{ny} ячеек, шаг {W / (nx - 1):.2f} мм.' + note)

    def create_plate(self):
        self._check_free_preview()
        nx, ny = self.grid_dims()
        self.push_history('Плоская плита')
        self.stl_mesh = None
        self.z = np.full((ny, nx), self.sp_base.value() + 0.0)
        self.show_mesh(reset=True)
        self.update_lock()
        self.after_change()
        self.tabs.setCurrentIndex(1)

    def on_base_changed(self, v):
        if self.z is not None:
            self.cancel_preview(silent=True)
            snap = self.snapshot()
            snap['base'] = self._base_prev
            self.push_history('Толщина основания', snap)
            self.z = self.z + (v - self._base_prev)
            self.show_mesh()
        self._base_prev = v

    def material_color(self):
        return MATERIALS[self.cb_mat.currentIndex()][1]

    def show_mesh(self, z=None, mask=None, dim=0.0, reset=False):
        if self.actor is not None:
            try:
                self.plotter.remove_actor(self.actor)
            except Exception:
                pass
            self.actor = None
        import pyvista as pv
        W, H = (self.sp_w.value(), self.sp_h.value())
        if z is None:
            z = self.z
        colors = None
        if z is not None:
            zd, md = reduce_for_display(np.asarray(z), mask)
            verts, faces = build_mesh_arrays(zd, W, H)
            if md is not None and dim > 0:
                base = np.array([int(self.material_color()[i:i + 2], 16) for i in (1, 3, 5)], float)
                ny_, nx_ = zd.shape
                colors = np.tile(base, (len(verts), 1))
                loop = boundary_loop(ny_, nx_)
                top_m = md.ravel()
                colors[:nx_ * ny_][~top_m] = base * (1 - dim)
                colors[nx_ * ny_:nx_ * ny_ + len(loop)][~top_m[loop]] = base * (1 - dim)
                colors = colors.astype(np.uint8)
        elif self.stl_mesh is not None:
            verts, faces = (np.asarray(self.stl_mesh.vertices), np.asarray(self.stl_mesh.faces))
        else:
            self.plotter.render()
            return
        poly = pv.PolyData(verts, np.hstack((np.full((len(faces), 1), 3), faces)).ravel())
        kw = dict(smooth_shading=True, show_scalar_bar=False, reset_camera=False)
        try:
            if colors is not None:
                self.actor = self.plotter.add_mesh(poly, scalars=colors, rgb=True, split_sharp_edges=True, **kw)
            else:
                self.actor = self.plotter.add_mesh(poly, color=self.material_color(), split_sharp_edges=True, **kw)
        except TypeError:
            if colors is not None:
                self.actor = self.plotter.add_mesh(poly, scalars=colors, rgb=True, **kw)
            else:
                self.actor = self.plotter.add_mesh(poly, color=self.material_color(), **kw)
        self.refresh_material()
        if reset or self._first_show:
            self._first_show = False
            self.update_light()
            self.set_view('iso')
        else:
            self.plotter.render()

    def refresh_material(self, *_):
        if self.actor is None:
            return
        import pyvista as pv
        prop = self.actor.GetProperty()
        c = self.l_contrast.value() / 100.0
        prop.SetAmbient(0.05 + 0.2 * (1 - c))
        prop.SetDiffuse(0.45 + 0.45 * c)
        prop.SetSpecular(0.03)
        if not self.preview_active:
            prop.SetColor(*pv.Color(self.material_color()).float_rgb)
        self.plotter.render()

    def on_material_changed(self, *_):
        if self.z is not None or self.stl_mesh is not None:
            self.show_mesh(self.preview_z if self.preview_active else None, self.preview_mask if self.preview_active else None, self.d_dim.value() / 100.0 if self.preview_active else 0.0)

    def update_light(self, *_):
        if not hasattr(self, 'plotter') or not hasattr(self, 'chk_sun'):
            return
        import pyvista as pv
        p = self.plotter
        try:
            p.remove_all_lights()
            p.renderer.SetAutomaticLightCreation(False)
        except Exception:
            pass
        if self.sun_actor is not None:
            try:
                p.remove_actor(self.sun_actor)
            except Exception:
                pass
            self.sun_actor = None
        if self.chk_sun.isChecked():
            az, el = (math.radians(self.l_az.value()), math.radians(self.l_el.value()))
            R = max(self.sp_w.value(), self.sp_h.value(), 100.0) * 1.5
            pos = (R * math.cos(el) * math.cos(az), R * math.cos(el) * math.sin(az), R * math.sin(el))
            sun = pv.Light(position=pos, focal_point=(0, 0, 0), color='white', intensity=self.l_bright.value() / 100.0, light_type='scene light')
            p.add_light(sun)
            if self.chk_icon.isChecked():
                self.sun_actor = p.add_mesh(pv.Sphere(radius=max(R * 0.02, 2.0), center=pos), color='#ffce49', lighting=False, pickable=False, reset_camera=False)
        if self.chk_fill.isChecked():
            p.add_light(pv.Light(light_type='headlight', intensity=0.22))
        try:
            p.enable_shadows() if self.chk_shadow.isChecked() else p.disable_shadows()
        except Exception:
            pass
        p.render()

    def set_view(self, key):
        p = self.plotter
        if key == 'iso':
            p.view_isometric()
        elif key == 'top':
            p.view_xy(negative=False)
        elif key == 'bottom':
            p.view_xy(negative=True)
        elif key == 'front':
            p.view_xz(negative=True)
        elif key == 'back':
            p.view_xz(negative=False)
        elif key == 'left':
            p.view_yz(negative=True)
        elif key == 'right':
            p.view_yz(negative=False)
        p.reset_camera()
        self.tabs.setCurrentIndex(1)

    def sel_vectors(self):
        return sorted((i for i in self.scene.selectedItems() if isinstance(i, VectorItem)), key=lambda i: i.seq)

    def _add_vec(self, data, select=False):
        it = VectorItem(data)
        self.scene.addItem(it)
        if select:
            it.setSelected(True)
        return it

    def _need_sel(self, n=1):
        v = self.sel_vectors()
        if len(v) < n:
            QMessageBox.information(self, 'Выделение', 'Выделите вектор в 2D.' if n == 1 else f'Выделите векторов: {n}.')
            return None
        return v

    def delete_selected(self):
        sel = [i for i in self.scene.selectedItems() if isinstance(i, (VectorItem, CtrlPoint))]
        if not sel:
            return
        snap = self.snapshot()
        for i in sel:
            self.scene.removeItem(i)
        self.push_history('Удаление', snap)
        self.after_change()

    def copy_selected(self):
        sel = self._need_sel()
        if not sel:
            return
        snap = self.snapshot()
        self.scene.clearSelection()
        gmap = {}
        for v in sel:
            d = v.data.copy()
            if d.group:
                gmap.setdefault(d.group, self._new_gid())
                d.group = gmap[d.group]
            d.translate(5 * self.K, 5 * self.K)
            self._add_vec(d, True)
        self.push_history('Копирование', snap)
        self.after_change()

    def _sel_bbox_center(self, vecs):
        pts = [p for v in vecs for p in v.data.flatten()]
        xs, ys = ([p[0] for p in pts], [p[1] for p in pts])
        return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)

    def mirror_selected(self, axis):
        sel = self._need_sel()
        if not sel:
            return
        snap = self.snapshot()
        cx, cy = self._sel_bbox_center(sel)
        for v in sel:
            v.data.transform((lambda x, y: (2 * cx - x, y)) if axis == 'h' else lambda x, y: (x, 2 * cy - y))
            v.refresh()
        self.push_history('Отражение', snap)

    def transform_selected(self, angle, scale, dx_mm, dy_mm):
        sel = self.sel_vectors()
        if not sel:
            return
        snap = self.snapshot()
        cx, cy = self._sel_bbox_center(sel)
        a = math.radians(angle)
        ca, sa = (math.cos(a), math.sin(a))
        f = lambda x, y: (cx + scale * (ca * (x - cx) - sa * (y - cy)) + dx_mm * self.K, cy + scale * (sa * (x - cx) + ca * (y - cy)) - dy_mm * self.K)
        for v in sel:
            v.data.transform(f)
            v.refresh()
        self.push_history('Преобразование вектора', snap)

    def transform_dialog(self):
        if not self._need_sel():
            return
        d = QDialog(self)
        d.setWindowTitle('Преобразование вектора')
        form = QFormLayout(d)
        ang = self._spin(0, -360, 360, 1, '°', 5)
        sc = self._spin(100, 1, 1000, 1, '%', 5)
        dx = self._spin(0, -3000, 3000, 2)
        dy = self._spin(0, -3000, 3000, 2)
        for label, w in [('Поворот', ang), ('Масштаб', sc), ('Сдвиг X (мм)', dx), ('Сдвиг Y (мм, вверх +)', dy)]:
            form.addRow(label, w)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        form.addRow(bb)
        if d.exec() == QDialog.Accepted:
            self.transform_selected(ang.value(), sc.value() / 100.0, dx.value(), dy.value())

    def _new_gid(self):
        self._gid += 1
        return self._gid + 1000 * int(time.time() % 1000)

    def group_selected(self):
        sel = self._need_sel(2)
        if not sel:
            return
        snap = self.snapshot()
        g = self._new_gid()
        for v in sel:
            v.data.group = g
        self.push_history('Группировка', snap)

    def ungroup_selected(self):
        sel = self._need_sel()
        if not sel:
            return
        snap = self.snapshot()
        for v in sel:
            v.data.group = 0
        self.push_history('Разгруппировка', snap)

    def _sync_groups(self):
        if self._in_group_sync:
            return
        self._in_group_sync = True
        try:
            gids = {v.data.group for v in self.sel_vectors() if v.data.group}
            if gids:
                for v in self.view.vectors():
                    if v.data.group in gids and (not v.isSelected()):
                        v.setSelected(True)
        finally:
            self._in_group_sync = False

    def join_selected(self):
        sel = [v for v in self.sel_vectors() if not v.data.closed]
        if len(sel) != 2:
            QMessageBox.information(self, 'Соединение', 'Выделите ровно два открытых вектора.')
            return
        snap = self.snapshot()
        merged = sel[0].data.join(sel[1].data, tol=2.0)
        for v in sel:
            self.scene.removeItem(v)
        self._add_vec(merged, True)
        self.push_history('Соединение векторов', snap)

    def _replace(self, v, parts, label, snap):
        self.scene.removeItem(v)
        self.scene.clearSelection()
        for p in parts:
            self._add_vec(p, True)
        self.push_history(label, snap)
        self.after_change()

    def vector_context_menu(self, sp, gpos):
        tol = 9.0 / max(self.view.transform().m11(), 1e-06)
        hit = None
        for v in self.view.vectors():
            h = v.data.hit(sp.x(), sp.y(), tol)
            if h:
                hit, vec = (h, v)
                break
        if hit is None:
            return
        if not vec.isSelected():
            self.scene.clearSelection()
            vec.setSelected(True)
        d = vec.data
        menu = QMenu(self)
        acts = {}
        if hit[0] == 'seg':
            acts['del_seg'] = menu.addAction('Удалить участок')
            acts['add'] = menu.addAction('Добавить узел здесь')
        else:
            acts['smooth'] = menu.addAction('Сделать узел острым' if d.is_smooth(hit[1]) else 'Сделать узел сглаженным')
            acts['del_node'] = menu.addAction('Удалить узел')
            acts['split'] = menu.addAction('Разделить вектор в этом узле')
        menu.addSeparator()
        acts['close'] = menu.addAction('Разомкнуть вектор' if d.closed else 'Замкнуть вектор')
        acts['copy'] = menu.addAction('Копировать')
        acts['mh'] = menu.addAction('Отразить по горизонтали')
        acts['mv'] = menu.addAction('Отразить по вертикали')
        acts['tr'] = menu.addAction('Преобразовать…')
        acts['group'] = menu.addAction('Группировать')
        acts['ungroup'] = menu.addAction('Разгруппировать')
        acts['delete'] = menu.addAction('Удалить вектор')
        a = menu.exec(gpos)
        if a is None:
            return
        snap = self.snapshot()
        if a is acts.get('del_seg'):
            self._replace(vec, d.remove_segment(hit[1]), 'Удаление участка', snap)
        elif a is acts.get('add'):
            d.insert_at(hit[1], hit[2])
            vec.refresh()
            self.push_history('Добавление узла', snap)
        elif a is acts.get('smooth'):
            d.make_corner(hit[1]) if d.is_smooth(hit[1]) else d.make_smooth(hit[1])
            vec.refresh()
            vec.overlay and vec.overlay.update()
            self.push_history('Тип узла', snap)
        elif a is acts.get('del_node'):
            if d.remove_node(hit[1]):
                vec.refresh()
                vec.overlay and vec.overlay.update()
                self.push_history('Удаление узла', snap)
        elif a is acts.get('split'):
            self._replace(vec, d.split_at(hit[1]), 'Разделение вектора', snap)
        elif a is acts['close']:
            if d.closed:
                d.closed = False
            elif len(d.nodes) >= 3:
                d.closed = True
            vec.refresh()
            self.push_history('Замкнуть / разомкнуть', snap)
        elif a is acts['copy']:
            self.copy_selected()
        elif a is acts['mh']:
            self.mirror_selected('h')
        elif a is acts['mv']:
            self.mirror_selected('v')
        elif a is acts['tr']:
            self.transform_dialog()
        elif a is acts['group']:
            self.group_selected()
        elif a is acts['ungroup']:
            self.ungroup_selected()
        elif a is acts['delete']:
            self.delete_selected()

    def add_ctrl_point(self, sp):
        snap = self.snapshot()
        self.scene.addItem(CtrlPoint(sp.x(), sp.y(), self.d_cdh.value()))
        self.push_history('Управляющая точка', snap)
        self.d_mode.setCurrentIndex(4)

    def set_selected_ctrl_height(self, v):
        sel = [i for i in self.scene.selectedItems() if isinstance(i, CtrlPoint)]
        for c in sel:
            c.set_dh(v)
        if sel:
            self.schedule_preview()

    def update_deform_rows(self, *_):
        modes = {0: ('angle', 'a', 'b'), 1: ('angle', 'amount', 'waves'), 2: ('apx', 'apy', 'rad', 'amount'), 3: ('hA', 'hB', 'rail1btn', 'rail2btn', 'lbprof'), 4: ('infl', 'cdh')}
        every = {'angle', 'a', 'b', 'amount', 'waves', 'apx', 'apy', 'rad', 'hA', 'hB', 'infl', 'cdh', 'sweepH', 'profsrc', 'profbtn', 'lbprof', 'rail1btn', 'rail2btn', 'scaleh'}
        show = set(modes[self.d_mode.currentIndex()])
        for k in every:
            self.rows[k].setVisible(k in show)

    def deform_params(self):
        return dict(mode=['bend', 'wave', 'apex', 'guides', 'points'][self.d_mode.currentIndex()], angle=self.d_angle.value(), a=self.d_a.value(), b=self.d_b.value(), amount=self.d_amount.value(), waves=self.d_waves.value(), apex_x=self.d_apx.value() / 100.0, apex_y=self.d_apy.value() / 100.0, radius=self.d_rad.value(), hA=self.d_hA.value(), hB=self.d_hB.value(), influence=self.d_infl.value(), section=self.section.points(), fixed=self.d_fixed.isChecked(), feather_mm=self.d_feather.value())

    def _need_z(self):
        if self.z is None:
            raise ValueError('Сначала создайте рельеф («Рельеф из фото» или «Плоская плита»).')

    def _closed_polys(self):
        return [[self.scene_to_grid(x, y) for x, y in v.scene_points()] for v in self.sel_vectors() if v.data.closed]

    def region_cov(self, inside=True):
        """Гладкая (с долями покрытия) область выделенных замкнутых векторов."""
        self._need_z()
        polys = self._closed_polys()
        if not polys:
            raise ValueError('Выделите замкнутый вектор в 2D — он задаёт область рельефа.')
        ny, nx = self.z.shape
        cov = raster_coverage(polys, nx, ny)
        if cov.max() <= 0:
            raise ValueError('Область слишком мала для текущей детализации.')
        return cov if inside else 1.0 - cov

    def region_mask(self):
        mask = self.region_cov() > 0.5
        if not mask.any():
            raise ValueError('Область слишком мала для текущей детализации.')
        return mask

    @staticmethod
    def _alive(item):
        return item is not None and item.scene() is not None

    def assign_role(self, kind):
        opens = [v for v in self.sel_vectors() if not v.data.closed]
        if len(opens) != 1:
            QMessageBox.information(self, 'Назначение вектора', 'Выделите ровно один открытый вектор.')
            return
        v = opens[0]
        roles = {'rail1': self.rail_items[0], 'rail2': self.rail_items[1], 'profile': self.profile_item}
        if any((x is v for k, x in roles.items() if k != kind)):
            QMessageBox.information(self, 'Назначение вектора', 'Этот вектор уже назначен на другую роль.')
            return
        if kind == 'rail1':
            self.rail_items[0] = v
        elif kind == 'rail2':
            self.rail_items[1] = v
        else:
            self.profile_item = v
            self.d_profsrc.setCurrentIndex(1)
        self.update_roles_label()
        self.schedule_preview()

    def update_roles_label(self):
        ok = lambda it: '✓' if self._alive(it) else '—'
        self.lb_prof.setText(f'Направляющая 1: {ok(self.rail_items[0])}   Направляющая 2: {ok(self.rail_items[1])}   Сечение: {ok(self.profile_item)}')

    def rails_grid(self):
        """Направляющие: назначенные кнопками либо два выделенных открытых вектора."""
        if self._alive(self.rail_items[0]) and self._alive(self.rail_items[1]):
            items = list(self.rail_items)
        else:
            opens = [v for v in self.sel_vectors() if not v.data.closed]
            if len(opens) < 2:
                raise ValueError('Назначьте две направляющие кнопками (выделить вектор → «Направляющая 1/2») или выделите два открытых вектора.')
            opens.sort(key=lambda v: self._sel_bbox_center([v]))
            items = opens[:2]
        return [[self.scene_to_grid(x, y) for x, y in v.scene_points()] for v in items]

    def sweep_profile(self, p):
        """Функция сечения v∈[0,1] → высота (мм). Сечение берётся вдоль хорды от первой точки к последней."""
        if self.d_profsrc.currentIndex() == 1:
            if not self._alive(self.profile_item):
                raise ValueError('Нарисуйте сечение открытым вектором, выделите его и нажмите «Сечение ← выделенный вектор».')
            a = np.array(self.profile_item.scene_points(), float)
            d = a[-1] - a[0]
            L = float(np.hypot(*d))
            if L < 1e-06:
                raise ValueError('Сечение слишком короткое: начало и конец вектора совпадают.')
            u = d / L
            up = np.array([u[1], -u[0]])
            if up[1] > 1e-09 or (abs(up[1]) <= 1e-09 and up[0] < 0):
                up = -up
            rel = a - a[0]
            return profile_from_points(list(zip(rel @ u / self.K, rel @ up / self.K)))
        H, pts = (self.d_sweepH.value(), p['section'])
        return lambda v: H * section_eval(pts, v)

    def compute_shape(self, cov_override=None):
        """Форма (мм над нулём) и вес применения по панели «Деформация»."""
        self._need_z()
        cx, cy = self.cell()
        ny, nx = self.z.shape
        p = self.deform_params()
        mode = p['mode']
        rails = guides = ctrl = v_param = None
        if mode in ('guides', 'sweep'):
            rails = self.rails_grid()
        if mode == 'sweep':
            cov = raster_coverage([rails[0] + rails[1][::-1]], nx, ny)
            if cov_override is not None:
                cov = cov * cov_override
            elif self._closed_polys():
                cov = cov * raster_coverage(self._closed_polys(), nx, ny)
            p['profile'] = self.sweep_profile(p)
            p['scale_h'] = self.d_scaleh.isChecked()
            p['profile_span'] = getattr(p['profile'], 'span', None)
        else:
            cov = cov_override if cov_override is not None else self.region_cov()
        mask = cov > 0.001
        if not mask.any():
            raise ValueError('Область пуста. Для выдавливания направляющие должны ограничивать область.')
        if mode == 'guides':
            guides = rails
        if mode == 'points':
            ctrl = [(*self.scene_to_grid(c.pos().x(), c.pos().y()), c.dh) for c in self.view.ctrl_points()]
        if mode in ('guides', 'sweep'):
            key = (hash(mask.tobytes()), tuple(np.round(np.asarray(rails[0]), 2).ravel()), tuple(np.round(np.asarray(rails[1]), 2).ravel()))
            if key != self._v_key:
                self._v_cache = two_rail_param(mask, cx, cy, rails[0], rails[1])
                self._v_key = key
            v_param = self._v_cache
        off = deform_offsets(mask, cx, cy, p, guides, ctrl, rails, v_param)
        if p['fixed'] and mode != 'sweep':
            weight = cov * feather_weight(cov, p['feather_mm'] / min(cx, cy))
        else:
            weight = cov
        return (off, weight)

    def compute_deform(self):
        shape, weight = self.compute_shape()
        newz = combine_relief(self.z, shape, weight, self.sp_base.value(), COMBINE_KEYS[self.d_combine.currentIndex()])
        return (newz, weight > 0.001)

    @busy
    def preview_deform(self):
        try:
            newz, mask = self.compute_deform()
        except ValueError as e:
            QMessageBox.information(self, 'Деформация', str(e))
            return
        self.preview_z, self.preview_mask, self.preview_active = (newz, mask, True)
        self.show_mesh(newz, mask, self.d_dim.value() / 100.0)
        self.tabs.setCurrentIndex(1)
        self.statusBar().showMessage('ПРЕДПРОСМОТР — геометрия ещё не изменена. «Применить» или «Отмена».')

    def schedule_preview(self, *_):
        if self.preview_active:
            self.preview_timer.start(250)

    @busy
    def _refresh_preview(self):
        if not self.preview_active:
            return
        try:
            newz, mask = self.compute_deform()
        except ValueError:
            return
        self.preview_z, self.preview_mask = (newz, mask)
        self.show_mesh(newz, mask, self.d_dim.value() / 100.0)

    def apply_deform(self):
        if not self.preview_active:
            self.preview_deform()
            if not self.preview_active:
                return
        self.push_history('Деформация поверхности')
        self.z = self.preview_z
        self.preview_active, self.preview_z, self.preview_mask = (False, None, None)
        self.show_mesh()
        self.after_change()
        self.statusBar().showMessage('Деформация применена. Ctrl+Z — отменить.')

    def cancel_preview(self, silent=False):
        if not self.preview_active:
            return
        self.preview_active, self.preview_z, self.preview_mask = (False, None, None)
        if not silent:
            self.show_mesh()
            self.statusBar().showMessage('Предпросмотр отменён.')

    def drop_ext(self):
        return ('.jpg', '.jpeg', '.jpe', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp')

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            ext = os.path.splitext(p)[1].lower()
            if ext in self.drop_ext():
                self.open_photo(p)
            elif ext == EXT:
                self.open_project(p)
            elif ext == '.stl':
                self.open_stl(p)
            elif ext == '.dxf':
                self.import_dxf(p)
            elif ext == '.svg':
                self.import_svg(p)
    SH_MODES = [('Добавить', 'add'), ('Вычитание', 'sub'), ('Обнул. внутри', 'zero_in'), ('По наибольш.', 'high'), ('По наименьш.', 'low'), ('Обнул. снаружи', 'zero_out')]

    def open_relief_editor(self):
        if self.relief_dialog is None:
            self.relief_dialog = self._build_shape_dialog()
        self.relief_dialog.show()
        self.relief_dialog.raise_()
        self.relief_dialog.activateWindow()

    def _build_shape_dialog(self):
        d = QDialog(self)
        d.setWindowTitle('Редактор формы')
        d.setWindowFlags(d.windowFlags() | Qt.Tool)
        root = QVBoxLayout(d)
        self.sh_source = QComboBox()
        self.sh_source.addItems(['Форма по выделенным замкнутым векторам', 'Рельеф из фото (яркость) внутри векторов'])
        root.addWidget(self.sh_source)
        top = QHBoxLayout()
        root.addLayout(top)
        col = QVBoxLayout()
        self.sh_kind_btns = []
        for i, (ch, tip) in enumerate([('◠', 'Округлая форма'), ('◆', 'Угловая форма'), ('▰', 'Плато (плоская вершина)')]):
            b = QToolButton()
            b.setText(ch)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setFixedSize(54, 54)
            b.clicked.connect(lambda _=False, k=i: self.sh_set_kind(k))
            col.addWidget(b)
            self.sh_kind_btns.append(b)
        top.addLayout(col)
        self.sh_view = ShapeProfileView()
        top.addWidget(self.sh_view)
        self.sh_angle_sl = QSlider(Qt.Vertical)
        self.sh_angle_sl.setRange(-90, 90)
        self.sh_angle_sl.setValue(45)
        top.addWidget(self.sh_angle_sl)
        right = QVBoxLayout()
        top.addLayout(right)
        right.addWidget(QLabel('Угол'))
        self.sh_angle = QSpinBox()
        self.sh_angle.setRange(-90, 90)
        self.sh_angle.setValue(45)
        self.sh_angle.setSuffix('°')
        self.sh_angle.setToolTip('Угол наклона боков. Отрицательный — вогнутая форма (выемка).')
        right.addWidget(self.sh_angle)
        right.addWidget(QLabel('Нач. высота'))
        self.sh_start = self._spin(0, -500, 500, 2)
        right.addWidget(self.sh_start)
        self.sh_scale_lb = QLabel('Масштаб ×1.0')
        right.addWidget(self.sh_scale_lb)
        right.addStretch(1)
        self.sh_scale_sl = QSlider(Qt.Vertical)
        self.sh_scale_sl.setRange(1, 50)
        self.sh_scale_sl.setValue(10)
        self.sh_scale_sl.setToolTip('Масштаб высоты формы: 0.1 … 5.0')
        top.addWidget(self.sh_scale_sl)
        box = QWidget()
        form = QFormLayout(box)
        self.sh_r_free = QRadioButton('Без ограничений')
        self.sh_r_limit = QRadioButton('Ограничить по высоте')
        self.sh_r_scale = QRadioButton('Масштаб по высоте')
        self.sh_r_const = QRadioButton('Постоянная высота (для векторов)')
        self.sh_r_free.setChecked(True)
        self.sh_limit = self._spin(1, 0, 500, 2)
        self.sh_scaleh = self._spin(1, 0, 500, 2)
        self.sh_const = self._spin(0, -500, 500, 2)
        form.addRow(self.sh_r_free)
        form.addRow(self.sh_r_limit, self.sh_limit)
        form.addRow(self.sh_r_scale, self.sh_scaleh)
        form.addRow(self.sh_r_const, self.sh_const)
        root.addWidget(box)
        self.sh_modegrp = QButtonGroup(d)
        self.sh_mode_btns = []
        for r in range(2):
            row = QHBoxLayout()
            for name, key in self.SH_MODES[r * 3:(r + 1) * 3]:
                b = QPushButton(name)
                b.setCheckable(True)
                b.setChecked(key == 'add')
                self.sh_modegrp.addButton(b)
                self.sh_mode_btns.append(b)
                row.addWidget(b)
            root.addLayout(row)
        row = QHBoxLayout()
        for text, slot in [('Применить', lambda: self.shape_apply()), ('Предпросмотр', lambda: self.shape_apply(True)), ('Сброс', self.sh_reset), ('Закрыть', d.hide)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        root.addLayout(row)
        self.sh_timer = QTimer(self)
        self.sh_timer.setSingleShot(True)
        self.sh_timer.timeout.connect(lambda: self.shape_apply(True))
        self.sh_angle.valueChanged.connect(self.sh_angle_sl.setValue)
        self.sh_angle_sl.valueChanged.connect(self.sh_angle.setValue)
        self.sh_angle.valueChanged.connect(self.sh_update)
        self.sh_scale_sl.valueChanged.connect(self.sh_update)
        for w in (self.sh_start, self.sh_limit, self.sh_scaleh, self.sh_const):
            w.valueChanged.connect(self.sh_update)
        for w in (self.sh_r_free, self.sh_r_limit, self.sh_r_scale, self.sh_r_const) + tuple(self.sh_mode_btns):
            w.toggled.connect(self.sh_update)
        self.sh_set_kind(0)
        return d

    def sh_set_kind(self, i):
        self.sh_kind = i
        for j, b in enumerate(self.sh_kind_btns):
            b.setChecked(j == i)
        self.sh_update()

    def sh_update(self, *_):
        self.sh_scale_lb.setText(f'Масштаб ×{self.sh_scale_sl.value() / 10.0:.1f}')
        self.sh_view.set_params(self.sh_kind, self.sh_angle.value())
        if self.sh_preview_flag and self.preview_active:
            self.sh_timer.start(250)

    def sh_reset(self):
        self.sh_set_kind(0)
        self.sh_angle.setValue(45)
        self.sh_start.setValue(0)
        self.sh_scale_sl.setValue(10)
        self.sh_r_free.setChecked(True)
        self.sh_limit.setValue(1)
        self.sh_scaleh.setValue(1)
        self.sh_const.setValue(0)
        self.sh_mode_btns[0].setChecked(True)
        self.sh_preview_flag = False
        self.cancel_preview()

    def sh_mode_key(self):
        for (_, key), b in zip(self.SH_MODES, self.sh_mode_btns):
            if b.isChecked():
                return key
        return 'add'

    def sh_height_mode(self):
        if self.sh_r_limit.isChecked():
            return ('limit', self.sh_limit.value())
        if self.sh_r_scale.isChecked():
            return ('scale', self.sh_scaleh.value())
        if self.sh_r_const.isChecked():
            return ('const', self.sh_const.value())
        return ('free', 0.0)

    def photo_shape(self):
        if self.photo_pil is None:
            raise ValueError('Нет фото: откройте фотографию (Ctrl+I).')
        ny, nx = self.z.shape
        h = photo_heights(self.photo_bytes, nx, ny, self.sp_blur.value(), self.chk_invert.isChecked())
        return self.sp_depth.value() * h

    @busy
    def shape_apply(self, preview=False):
        try:
            self._need_z()
            mode = self.sh_mode_key()
            cov = self.region_cov(inside=True)
            base = self.sp_base.value()
            cx, cy = self.cell()
            names = dict(((k, n) for n, k in self.SH_MODES))
            label = 'Редактор формы: ' + names[mode].lower()
            if mode == 'zero_in':
                newz = combine_relief(self.z, np.zeros(self.z.shape), cov, base, 'zero')
                weight = cov
            elif mode == 'zero_out':
                weight = 1.0 - cov
                newz = combine_relief(self.z, np.zeros(self.z.shape), weight, base, 'zero')
            else:
                if self.sh_source.currentIndex() == 1:
                    shape = self.photo_shape()
                else:
                    hm, val = self.sh_height_mode()
                    shape = build_shape(cov, cx, cy, ['round', 'angular', 'plateau'][self.sh_kind], self.sh_angle.value(), hm, val, self.sh_start.value(), self.sh_scale_sl.value() / 10.0)
                weight = cov
                newz = combine_relief(self.z, shape, weight, base, mode)
        except ValueError as e:
            QMessageBox.information(self, 'Редактор формы', str(e))
            return
        if preview:
            self.preview_z, self.preview_mask, self.preview_active = (newz, weight > 0.001, True)
            self.sh_preview_flag = True
            self.show_mesh(newz, self.preview_mask, self.d_dim.value() / 100.0)
            self.tabs.setCurrentIndex(1)
            self.statusBar().showMessage('ПРЕДПРОСМОТР формы — нажмите «Применить» или «Сброс».')
            return
        self.sh_preview_flag = False
        self.cancel_preview(silent=True)
        self.push_history(label)
        self.z = newz
        self.show_mesh()
        self.after_change()
        self.statusBar().showMessage(label + ' — выполнено. Ctrl+Z — отменить.')

    def create_dome(self):
        self.open_relief_editor()
        self.sh_set_kind(0)
        self.sh_angle.setValue(90)
        self.sh_mode_btns[0].setChecked(True)
        self.statusBar().showMessage('Купол: выделите замкнутый вектор и нажмите «Применить» в Редакторе формы.')

    def profile_from_item(self, item):
        """Сечение из нарисованного вектора: вдоль хорды от первой точки к последней, вверх — выше."""
        a = np.array(item.scene_points(), float)
        d = a[-1] - a[0]
        L = float(np.hypot(*d))
        if L < 1e-06:
            raise ValueError('Сечение слишком короткое: начало и конец вектора совпадают.')
        u = d / L
        up = np.array([u[1], -u[0]])
        if up[1] > 1e-09 or (abs(up[1]) <= 1e-09 and up[0] < 0):
            up = -up
        rel = a - a[0]
        return profile_from_points(list(zip(rel @ u / self.K, rel @ up / self.K)))

    def open_sweep_dialog(self):
        if self.sweep_dialog is None:
            self.sweep_dialog = self._build_sweep_dialog()
        self.sw_refresh()
        self.sweep_dialog.show()
        self.sweep_dialog.raise_()
        self.sweep_dialog.activateWindow()

    def _build_sweep_dialog(self):
        d = QDialog(self)
        d.setWindowTitle('Вытягивание по двум направляющим')
        d.setWindowFlags(d.windowFlags() | Qt.Tool)
        d.resize(510, 590)
        d.setMinimumSize(350, 260)
        outer = QVBoxLayout(d)
        outer.setContentsMargins(5, 5, 5, 5)
        scroll = QScrollArea(d)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        lay = QVBoxLayout(content)
        intro = QLabel('Рельеф создаётся вытягиванием одного или нескольких сечений вдоль двух направляющих и объединяется с исходным рельефом. Нужно минимум три незамкнутых вектора: две направляющие и сечение. Если выбрано несколько векторов, при нажатии «Верхняя направляющая» первый вектор — верхняя, второй — нижняя, остальные — сечения. Если выбрана «Граница по Z», высота сечения определяется этим вектором (расстояние от середины между направляющими).')
        intro.setWordWrap(True)
        lay.addWidget(intro)
        lay.addWidget(QLabel('<b>Выбор векторов</b>'))
        form = QFormLayout()
        for text, which in [('Верхняя направляющая', 'top'), ('Нижняя направляющая', 'bottom')]:
            b = QPushButton('Выбор')
            b.clicked.connect(lambda _=False, w=which: self.sw_begin_pick(w))
            form.addRow(text, b)
        flips = QHBoxLayout()
        self.sw_flip1, self.sw_flip2 = (QCheckBox('Первую'), QCheckBox('Вторую'))
        flips.addWidget(self.sw_flip1)
        flips.addWidget(self.sw_flip2)
        form.addRow('Развернуть направляющие', flips)
        bz = QPushButton('Выбор')
        bz.clicked.connect(lambda: self.sw_begin_pick('z'))
        form.addRow('Граница по Z', bz)
        self.sw_use_z = QCheckBox('Кривая изменения высоты по Z')
        form.addRow(self.sw_use_z)
        lay.addLayout(form)
        lay.addWidget(QLabel('<b>Состояние</b>'))
        self.sw_state = QLabel()
        self.sw_state.setTextFormat(Qt.RichText)
        lay.addWidget(self.sw_state)
        self.sw_list = QListWidget()
        self.sw_list.setMaximumHeight(125)
        self.sw_positions = {}
        self.sw_list.itemDoubleClicked.connect(lambda _item: self.sw_edit_position())
        lay.addWidget(self.sw_list)
        editpos = QPushButton('Положение выбранного сечения, % (как в ArtCAM)')
        editpos.setToolTip('Можно также дважды нажать на сечение в списке')
        editpos.clicked.connect(self.sw_edit_position)
        lay.addWidget(editpos)
        row = QHBoxLayout()
        for text, slot in [('Добавить сечение', lambda: self.sw_begin_pick('section')), ('Вставить сечение', lambda: self.sw_begin_pick('insert')), ('Удалить сечение', self.sw_remove)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        lay.addLayout(row)
        self.sw_trans = QCheckBox('Переход между сечениями')
        self.sw_trans.setChecked(True)
        self.sw_scalew = QCheckBox('Масштабировать высоту по ширине')
        self.sw_scalew.setChecked(True)
        self.sw_scaleend = QCheckBox('Масштабировать конечную высоту')
        self.sw_end = self._spin(0, -500, 500, 2)
        self.sw_start = self._spin(0, -500, 500, 2)
        lay.addWidget(self.sw_trans)
        lay.addWidget(self.sw_scalew)
        r = QHBoxLayout()
        r.addWidget(self.sw_scaleend)
        r.addWidget(self.sw_end)
        lay.addLayout(r)
        r = QHBoxLayout()
        r.addWidget(QLabel('Начальная высота'))
        r.addWidget(self.sw_start)
        lay.addLayout(r)
        lay.addWidget(QLabel('<b>Объединить</b>'))
        self.sw_mode_btns = []
        self.sw_mode_group = QButtonGroup(d)
        for rowitems in ([('Добавить', 'add'), ('Вычесть', 'sub')], [('По наибольшему', 'high'), ('По наименьшему', 'low')]):
            r = QHBoxLayout()
            for name, key in rowitems:
                rb = QRadioButton(name)
                rb.setProperty('key', key)
                rb.setChecked(key == 'add')
                self.sw_mode_group.addButton(rb)
                self.sw_mode_btns.append(rb)
                r.addWidget(rb)
            lay.addLayout(r)
        row = QHBoxLayout()
        for text, slot in [('Сброс', self.sw_reset), ('Предпросмотр', lambda: self.sw_run(True)), ('Вычислить', lambda: self.sw_run(False)), ('Закрыть', d.hide)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        lay.addLayout(row)
        return d

    def sw_begin_pick(self, which):
        """Спрятать диалог, чтобы можно было действительно выбрать вектор мышью в 2D."""
        self._sw_pending_pick = which
        self.scene.clearSelection()
        self.sweep_dialog.hide()
        self.tabs.setCurrentIndex(0)
        self.view.set_mode('select')
        self.view.setFocus()
        self.statusBar().showMessage('Выделите ОТКРЫТЫЙ вектор мышью, затем нажмите Enter. Esc — отмена выбора.')

    def sw_complete_pick(self):
        which = getattr(self, '_sw_pending_pick', None)
        if not which:
            return False
        self._sw_pending_pick = None
        if which == 'section':
            self.sw_add(False)
        elif which == 'insert':
            self.sw_add(True)
        else:
            self.sw_pick(which)
        self.open_sweep_dialog()
        return True

    def sw_open_selected(self):
        return [v for v in self.sel_vectors() if not v.data.closed]

    def sw_used(self):
        return [x for x in (self.sw_rails[0], self.sw_rails[1], self.sw_zb) if x is not None] + list(self.sw_sections)

    def sw_pick(self, which):
        sel = self.sw_open_selected()
        if which == 'top':
            if len(sel) >= 2:
                self.sw_rails = [sel[0], sel[1]]
                if len(sel) > 2:
                    self.sw_sections = sel[2:]
                    self.sw_positions = {}
            elif len(sel) == 1:
                self.sw_rails[0] = sel[0]
            else:
                QMessageBox.information(self, 'Выбор', 'Выделите открытый вектор (или несколько).')
        elif len(sel) == 1:
            if which == 'bottom':
                self.sw_rails[1] = sel[0]
            else:
                self.sw_zb = sel[0]
        else:
            QMessageBox.information(self, 'Выбор', 'Выделите ровно один открытый вектор.')
        self.sw_refresh()
        if self.sw_rails[0] is self.sw_rails[1] and self.sw_rails[0] is not None:
            self.sw_rails[1] = None
            QMessageBox.information(self, 'Направляющие', 'Выберите два разных вектора.')

    def sw_add(self, insert):
        busy_ = [x for x in (self.sw_rails[0], self.sw_rails[1], self.sw_zb) if x is not None]
        new = [v for v in self.sw_open_selected() if all((v is not x for x in busy_ + self.sw_sections))]
        if not new:
            QMessageBox.information(self, 'Сечение', 'Выделите открытые векторы, не занятые как направляющие.')
            return
        row = self.sw_list.currentRow()
        at = row if insert and row >= 0 else len(self.sw_sections)
        self.sw_sections[at:at] = new
        self.sw_refresh()

    def sw_remove(self):
        row = self.sw_list.currentRow()
        if 0 <= row < len(self.sw_sections):
            removed = self.sw_sections.pop(row)
            self.sw_positions.pop(removed.data.uid, None)
        self.sw_refresh()

    def sw_reset(self):
        self.sw_rails, self.sw_zb, self.sw_sections = ([None, None], None, [])
        self.sw_positions = {}
        self.sw_flip1.setChecked(False)
        self.sw_flip2.setChecked(False)
        self.sw_use_z.setChecked(False)
        self.sw_trans.setChecked(True)
        self.sw_scalew.setChecked(True)
        self.sw_scaleend.setChecked(False)
        self.sw_start.setValue(0)
        self.sw_end.setValue(0)
        self.sw_mode_btns[0].setChecked(True)
        self.cancel_preview()
        self.sw_refresh()

    def sw_edit_position(self):
        """Точное положение профиля вдоль двух направляющих (0..100%)."""
        row = self.sw_list.currentRow()
        if not 0 <= row < len(self.sw_sections):
            QMessageBox.information(self, 'Сечение', 'Сначала выберите сечение в списке.')
            return
        item = self.sw_sections[row]
        current = self.sw_positions.get(item.data.uid, 100.0 * row / max(len(self.sw_sections) - 1, 1))
        value, ok = QInputDialog.getDouble(self, 'Положение сечения', 'Положение вдоль направляющих, % (0 — начало, 100 — конец):', current, 0.0, 100.0, 2)
        if ok:
            self.sw_positions[item.data.uid] = value
            self.sw_refresh()
            self.sw_list.setCurrentRow(row)

    def sw_refresh(self):
        self.sw_sections = [v for v in self.sw_sections if self._alive(v)]
        for i in (0, 1):
            if not self._alive(self.sw_rails[i]):
                self.sw_rails[i] = None
        if not self._alive(self.sw_zb):
            self.sw_zb = None
        red = lambda t: f'<span style="color:#e03030">{t}</span>'
        ok = lambda t: f'<b>{t}</b>'
        n = '—'
        if self.sw_rails[0] is not None and self.sw_rails[1] is not None:
            lens = [_polylen(np.array(v.scene_points(), float)) / self.K for v in self.sw_rails]
            n = str(int(np.clip(max(lens), 200, 600)))
        zt = 'Не используется' if not self.sw_use_z.isChecked() else ok('Выбрано') if self.sw_zb else red('Не выбрано')
        self.sw_state.setText(f"Первая: {(ok('Выбрано') if self.sw_rails[0] else red('Не выбрано'))}<br>Вторая: {(ok('Выбрано') if self.sw_rails[1] else red('Не выбрано'))}<br>Граница по Z: {zt}<br>Сечение: {(ok(str(len(self.sw_sections))) if self.sw_sections else red('Не выбрано'))}<br>Отрезков: {n}")
        self.sw_list.clear()
        k = len(self.sw_sections)
        for i in range(k):
            self.sw_list.addItem(f'Сечение {i + 1} — на {self.sw_positions.get(self.sw_sections[i].data.uid, 100 * i / max(k - 1, 1)):.2f}% длины')

    def sw_compute(self):
        self._need_z()
        self.sw_refresh()
        if self.sw_rails[0] is None or self.sw_rails[1] is None:
            raise ValueError('Выберите верхнюю и нижнюю направляющие.')
        if not self.sw_sections:
            raise ValueError('Добавьте хотя бы одно сечение (открытый вектор).')
        cx, cy = self.cell()
        ny, nx = self.z.shape
        to_grid = lambda v: [self.scene_to_grid(x, y) for x, y in v.scene_points()]
        A, B = (to_grid(self.sw_rails[0]), to_grid(self.sw_rails[1]))
        f1, f2 = (self.sw_flip1.isChecked(), self.sw_flip2.isChecked())
        cov = raster_coverage([rails_polygon(A, B, cx, cy)], nx, ny)
        mask = cov > 0.001
        if not mask.any():
            raise ValueError('Направляющие не ограничивают область.')
        key = (hash(mask.tobytes()), tuple(np.round(np.asarray(A), 2).ravel()), tuple(np.round(np.asarray(B), 2).ravel()), f1, f2)
        if key != self._v_key:
            self._v_cache = two_rail_param(mask, cx, cy, A, B, flip1=f1, flip2=f2)
            self._v_key = key
        v, wid, sp = self._v_cache
        k = len(self.sw_sections)
        positions = np.array([self.sw_positions.get(it.data.uid, 100.0 * i / max(k - 1, 1)) / 100.0 for i, it in enumerate(self.sw_sections)])
        order = np.argsort(positions, kind='stable')
        positions = positions[order]
        if len(positions) > 1 and np.any(np.diff(positions) < 1e-06):
            raise ValueError('Два сечения имеют одинаковое положение. Укажите разные проценты.')
        profs = [self.profile_from_item(self.sw_sections[int(i)]) for i in order]
        p = dict(spans=[getattr(f, 'span', None) for f in profs], scale_w=self.sw_scalew.isChecked(), transition=self.sw_trans.isChecked(), start_h=self.sw_start.value(), scale_end=self.sw_scaleend.isChecked(), end_h=self.sw_end.value())
        if self.sw_use_z.isChecked():
            if self.sw_zb is None:
                raise ValueError('Выберите вектор «Граница по Z» или отключите эту опцию.')
            p['zb'] = zboundary_heights(A, B, to_grid(self.sw_zb), cx, cy, f1, f2)
        ys, xs = np.where(mask)
        shape = np.zeros(self.z.shape)
        shape[ys, xs] = sweep_values(v, wid, sp, profs, positions, p)
        mode = next((b.property('key') for b in self.sw_mode_btns if b.isChecked()))
        return (combine_relief(self.z, shape, cov, self.sp_base.value(), mode), cov > 0.001, mode)

    @busy
    def sw_run(self, preview):
        try:
            newz, pm, mode = self.sw_compute()
        except ValueError as e:
            QMessageBox.information(self, 'Вытягивание по двум направляющим', str(e))
            return
        if preview:
            self.preview_z, self.preview_mask, self.preview_active = (newz, pm, True)
            self.sh_preview_flag = False
            self.show_mesh(newz, pm, self.d_dim.value() / 100.0)
            self.tabs.setCurrentIndex(1)
            self.statusBar().showMessage('ПРЕДПРОСМОТР вытягивания — нажмите «Вычислить» или «Сброс».')
            return
        self.cancel_preview(silent=True)
        self.push_history('Вытягивание по двум направляющим')
        self.z = newz
        self.show_mesh()
        self.after_change()
        self.statusBar().showMessage('Вытягивание по двум направляющим выполнено. Ctrl+Z — отменить.')

    @busy
    def relief_op(self, op):
        try:
            self._need_z()
        except ValueError as e:
            QMessageBox.information(self, 'Рельеф', str(e))
            return
        self.cancel_preview(silent=True)
        base = self.sp_base.value()
        r = self.z.astype(np.float64) - base
        try:
            cov = self.region_cov()
        except ValueError:
            cov = np.ones(self.z.shape)
        cx, cy = self.cell()
        name = {'zero': 'Обнулить', 'smooth': 'Сглаживание', 'scale': 'Масштаб', 'invert': 'Инвертировать', 'mirror_h': 'Отражение по горизонтали', 'mirror_v': 'Отражение по вертикали', 'offset': 'Смещение'}[op]
        if op == 'zero':
            r2 = np.zeros_like(r)
        elif op == 'smooth':
            val, ok = QInputDialog.getDouble(self, 'Сглаживание', 'Радиус сглаживания, мм:', 1.0, 0.1, 100.0, 1)
            if not ok:
                return
            r2 = gaussian_blur(r, val / min(cx, cy) / 2.0)
        elif op == 'scale':
            val, ok = QInputDialog.getDouble(self, 'Масштаб', 'Масштаб высот, %:', 100.0, 1.0, 1000.0, 1)
            if not ok:
                return
            r2 = r * val / 100.0
        elif op == 'invert':
            sel = r[cov > 0.5]
            r2 = (float(sel.max()) if sel.size else float(r.max())) - r
        elif op == 'offset':
            val, ok = QInputDialog.getDouble(self, 'Смещение', 'Добавить высоту, мм:', 1.0, -500.0, 500.0, 2)
            if not ok:
                return
            r2 = r + val
        else:
            self.push_history(name)
            self.z = np.flip(self.z, axis=1 if op == 'mirror_h' else 0)
            self.show_mesh()
            self.after_change()
            return
        self.push_history(name)
        self.z = np.maximum(base + r + cov * (r2 - r), 0.05)
        self.show_mesh()
        self.after_change()
        self.statusBar().showMessage(name + ' выполнено' + (' (внутри выделенных векторов).' if cov.min() < 1 else '.'))

    @busy
    def do_transform(self, copy, flipx, flipy):
        try:
            mask = self.region_mask()
        except ValueError as e:
            QMessageBox.information(self, 'Перемещение', str(e))
            return
        self.cancel_preview()
        cx, cy = self.cell()
        dx, dy, ang = (self.t_dx.value(), self.t_dy.value(), self.t_ang.value())
        sc = self.t_scale.value() / 100.0
        snap = self.snapshot()
        newz, _ = transform_region(self.z, mask, dx / cx, dy / cy, ang, sc, flipx, flipy, copy, cx, cy)
        vecs = self.sel_vectors()
        if self.t_vec.isChecked() and vecs:
            gx, gy = mask_centroid(mask)
            csx, csy = self.grid_to_scene(gx, gy)
            a = math.radians(ang)
            c, s = (math.cos(a), math.sin(a))

            def f(x, y):
                vx, vy = (x - csx, y - csy)
                if flipx:
                    vx = -vx
                if flipy:
                    vy = -vy
                rx, ry = (c * vx + s * vy, -s * vx + c * vy)
                return (csx + sc * rx + dx * self.K, csy + sc * ry - dy * self.K)
            if copy:
                self.scene.clearSelection()
                for v in vecs:
                    d = v.data.copy()
                    d.group = 0
                    d.transform(f)
                    self._add_vec(d, True)
            else:
                for v in vecs:
                    v.data.transform(f)
                    v.refresh()
        self.push_history('Копирование рельефа' if copy else 'Перемещение рельефа', snap)
        self.z = newz
        self.show_mesh()
        self.after_change()

    @busy
    def make_contours(self):
        if self.z is None:
            QMessageBox.information(self, 'Контуры', 'Сначала создайте рельеф.')
            return
        level = self.c_level.value()
        res = height_contours(self.z - self.sp_base.value(), level)
        if not res:
            QMessageBox.information(self, 'Контуры', 'На этом уровне пересечений нет.')
            return
        snap = self.snapshot()
        self.scene.clearSelection()
        for pts, closed in res:
            self._add_vec(polyline_path([self.grid_to_scene(x, y) for x, y in pts], closed), True)
        self.push_history('Контуры по высоте', snap)
        self.tabs.setCurrentIndex(0)
        self.after_change()
        self.statusBar().showMessage(f'Построено векторов: {len(res)} (уровень {level:g} мм). Их можно править.')

    def import_dxf(self, path=None):
        if not isinstance(path, str):
            path, _ = QFileDialog.getOpenFileName(self, 'Импорт DXF — размер 1:1', '', 'DXF (*.dxf)')
        if not path:
            return
        try:
            import ezdxf
            from ezdxf import path as ezpath, units
            from ezdxf.disassemble import recursive_decompose
            doc = ezdxf.readfile(path)
            factor = units.conversion_factor(doc.units, units.MM) if doc.units else 1.0
            polys, skipped = ([], {})
            for e in recursive_decompose(doc.modelspace()):
                try:
                    p = ezpath.make_path(e)
                    for sp in p.sub_paths():
                        pts = [(v.x * factor, v.y * factor) for v in sp.flattening(0.01 / factor)]
                        if len(pts) >= 2:
                            polys.append((pts, bool(sp.is_closed)))
                except (TypeError, ValueError, NotImplementedError, AttributeError):
                    name = e.dxftype()
                    skipped[name] = skipped.get(name, 0) + 1
            if not polys:
                raise ValueError('Нет поддерживаемых контуров. Текст преобразуйте в кривые.')
        except Exception as ex:
            QMessageBox.critical(self, 'Ошибка DXF', str(ex))
            return
        allp = [p for pts, _ in polys for p in pts]
        mx = (min((x for x, y in allp)) + max((x for x, y in allp))) / 2
        my = (min((y for x, y in allp)) + max((y for x, y in allp))) / 2
        snap = self.snapshot()
        self.scene.clearSelection()
        for pts, closed in polys:
            if closed and len(pts) > 2 and (pts[0] == pts[-1]):
                pts = pts[:-1]
            points = [((x - mx) * self.K + self.canvas_w / 2, self.canvas_h / 2 - (y - my) * self.K) for x, y in pts]
            self._add_vec(polyline_path(points, closed and len(points) >= 3), True)
        self.push_history('Импорт DXF 1:1', snap)
        self.after_change()
        self.tabs.setCurrentIndex(0)
        note = ' Без единиц в файле: приняты миллиметры.' if not doc.units else ''
        if skipped:
            note += ' Пропущено: ' + ', '.join((f'{k}: {v}' for k, v in sorted(skipped.items())))
        self.statusBar().showMessage('DXF: размер 1:1, точность аппроксимации 0,01 мм.' + note, 20000)

    def import_svg(self, path=None):
        if not isinstance(path, str):
            path, _ = QFileDialog.getOpenFileName(self, 'Импорт SVG — размер 1:1', '', 'SVG (*.svg)')
        if not path:
            return
        try:
            with open(path, encoding='utf-8-sig') as f:
                paths, _ = svg_to_paths(f.read())
            if not paths:
                raise ValueError('В файле нет контуров. Преобразуйте текст в кривые перед импортом.')
        except Exception as ex:
            QMessageBox.critical(self, 'Ошибка SVG', str(ex))
            return
        snap = self.snapshot()
        self.scene.clearSelection()
        for d in paths:
            d.transform(lambda x, y: (x * self.K, y * self.K))
            self._add_vec(d, True)
        self.push_history('Импорт SVG 1:1', snap)
        self.after_change()
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage('SVG импортирован в миллиметрах, без подгонки размера. Векторы выделены.')

    def export_svg(self):
        vecs = self.view.vectors()
        if not vecs:
            QMessageBox.information(self, 'SVG', 'Нет векторов для экспорта.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Экспорт SVG', '', 'SVG (*.svg)')
        if not path:
            return
        if not path.lower().endswith('.svg'):
            path += '.svg'
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(svg_document([v.data for v in vecs], self.sp_w.value(), self.sp_h.value(), 1.0 / self.K))
        except Exception as ex:
            QMessageBox.critical(self, 'Ошибка', str(ex))
            return
        self.statusBar().showMessage('SVG сохранён (миллиметры): ' + path)

    def open_stl(self, path=None):
        if not isinstance(path, str):
            path, _ = QFileDialog.getOpenFileName(self, 'Открыть STL', '', 'STL (*.stl)')
        if not path:
            return
        try:
            import trimesh
            mesh = trimesh.load(path, force='mesh')
            if mesh.is_empty or not hasattr(mesh, 'faces'):
                raise ValueError('В файле нет треугольной сетки.')
        except Exception as ex:
            QMessageBox.critical(self, 'Ошибка STL', str(ex))
            return
        self.cancel_preview(silent=True)
        b = mesh.bounds
        mesh.apply_translation([-(b[0][0] + b[1][0]) / 2, -(b[0][1] + b[1][1]) / 2, -b[0][2]])
        ext = mesh.extents
        self.push_history('Открыть STL')
        self.stl_mesh = mesh
        self.z = None
        for w, v in ((self.sp_w, ext[0]), (self.sp_h, ext[1])):
            w.blockSignals(True)
            w.setValue(float(np.clip(v, 10, 3000)))
            w.blockSignals(False)
        edit_note = ''
        try:
            step_stl = self.sp_step.value()
            nx, ny, _ = grid_from_step(float(ext[0]), float(ext[1]), step_stl)
            z = heightmap_from_mesh(mesh, float(ext[0]), float(ext[1]), nx, ny)
            self.z = z
            self.stl_mesh = None
            for w, v in ((self.sp_base, max(float(z.min()), 0.5)), (self.sp_depth, max(float(z.max() - z.min()), 0.5))):
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)
            self._base_prev = self.sp_base.value()
            edit_note = ' Рельеф доступен для редактирования.'
        except Exception as ex:
            edit_note = f' Преобразование в рельеф не выполнено: {type(ex).__name__}: {ex}. Исходная сетка сохранена для просмотра и экспорта.'
        if not self.view.vectors() and self.photo_pil is None:
            self.rebuild_canvas(force=True)
        self.show_mesh(reset=True)
        self.update_lock()
        self.tabs.setCurrentIndex(1)
        self.statusBar().showMessage('STL загружен. Размеры, мм: ' + ', '.join((f'{v:.2f}' for v in ext)) + '.' + edit_note)

    def export_stl(self):
        self.cancel_preview()
        if self.z is None and self.stl_mesh is None:
            QMessageBox.warning(self, 'Нет модели', 'Создайте рельеф или откройте STL.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Экспорт STL', '', 'STL (*.stl)')
        if not path:
            return
        if not path.lower().endswith('.stl'):
            path += '.stl'
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self.z is not None:
                verts, faces = build_mesh_arrays(self.z, self.sp_w.value(), self.sp_h.value())
            else:
                verts, faces = (np.asarray(self.stl_mesh.vertices), np.asarray(self.stl_mesh.faces))
            import trimesh
            check = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            if not np.isfinite(verts).all():
                raise ValueError('Модель содержит недопустимые координаты.')
            tight = check.is_watertight
            winding = check.is_winding_consistent
            degenerate = int(np.count_nonzero(check.area_faces < 1e-12))
            write_stl(path, verts, faces)
        except Exception as ex:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, 'Ошибка экспорта', str(ex))
            return
        QApplication.restoreOverrideCursor()
        QMessageBox.information(self, 'Готово', f"STL сохранён (мм):\n{path}\nЗамкнутая модель: {('да' if tight else 'нет')}\nОриентация граней согласована: {('да' if winding else 'нет')}\nВырожденных треугольников: {degenerate}")

    def _settings(self):
        return dict(w=self.sp_w.value(), h=self.sp_h.value(), base=self.sp_base.value(), depth=self.sp_depth.value(), step=self.sp_step.value(), grid=self.sp_grid.value(), blur=self.sp_blur.value(), invert=self.chk_invert.isChecked(), photo_fade=self.sl_photo.value(), material=self.cb_mat.currentIndex(), bright=self.l_bright.value(), contrast=self.l_contrast.value(), az=self.l_az.value(), el=self.l_el.value(), sun=self.chk_sun.isChecked(), fill=self.chk_fill.isChecked(), level=self.c_level.value())

    def _write(self, path):
        try:
            self.cancel_preview()
            meta = {'settings': self._settings(), 'saved': time.strftime('%Y-%m-%d %H:%M:%S'), 'vec': [v.data.to_dict() for v in self.view.vectors()], 'ctrl': [(c.pos().x(), c.pos().y(), c.dh) for c in self.view.ctrl_points()]}
            arrays = {'z': self.z} if self.z is not None else {}
            pack_project(path, meta, self.photo_bytes, self.photo_name, arrays)
        except Exception as ex:
            QMessageBox.critical(self, 'Ошибка сохранения', str(ex))
            return False
        self.project_path = path
        self._saved_sig = self.state_sig()
        self.setWindowTitle(f'{APP} — {os.path.basename(path)}')
        self.statusBar().showMessage('Проект сохранён: ' + path)
        return True

    def save_project(self):
        return self._write(self.project_path) if self.project_path else self.save_project_as()

    def save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Сохранить проект', '', 'Проект Stone Relief (*.srelief)')
        if not path:
            return False
        if not path.lower().endswith(EXT):
            path += EXT
        return self._write(path)

    def confirm_discard(self):
        if not self.has_content() or self.state_sig() == self._saved_sig:
            return True
        r = QMessageBox.question(self, 'Несохранённые изменения', 'Сохранить проект?', QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Cancel:
            return False
        return bool(self.save_project()) if r == QMessageBox.Save else True

    def _clear_all(self):
        self.cancel_preview(silent=True)
        for it in self.view.vectors() + self.view.ctrl_points():
            self.scene.removeItem(it)
        self.z = None
        self.stl_mesh = None
        self.photo_pil = self.photo_bytes = self.photo_name = self.photo_arr = None
        self.undo_stack, self.redo_stack = ([], [])
        self.refresh_history()
        if self.actor is not None:
            try:
                self.plotter.remove_actor(self.actor)
            except Exception:
                pass
            self.actor = None
        self.view.set_photo(None)

    def new_project(self):
        if not self.confirm_discard():
            return
        self._clear_all()
        self.project_path = None
        self.setWindowTitle(APP)
        self.update_lock()
        self.rebuild_canvas()
        self._saved_sig = self.state_sig()

    def open_project(self, path=None):
        if not self.confirm_discard():
            return
        if not isinstance(path, str):
            path, _ = QFileDialog.getOpenFileName(self, 'Открыть проект', '', 'Проект Stone Relief (*.srelief)')
        if not path:
            return
        try:
            meta, photo, arrays = unpack_project(path)
        except Exception as ex:
            QMessageBox.critical(self, 'Ошибка открытия', str(ex))
            return
        self._clear_all()
        st = meta.get('settings', {})
        for w, k in [(self.sp_w, 'w'), (self.sp_h, 'h'), (self.sp_base, 'base'), (self.sp_depth, 'depth'), (self.sp_step, 'step'), (self.sp_grid, 'grid'), (self.sp_blur, 'blur'), (self.sl_photo, 'photo_fade'), (self.cb_mat, 'material'), (self.l_bright, 'bright'), (self.l_contrast, 'contrast'), (self.l_az, 'az'), (self.l_el, 'el'), (self.c_level, 'level')]:
            if k in st:
                w.blockSignals(True)
                (w.setCurrentIndex if isinstance(w, QComboBox) else w.setValue)(st[k])
                w.blockSignals(False)
        for w, k in [(self.chk_invert, 'invert'), (self.chk_sun, 'sun'), (self.chk_fill, 'fill')]:
            if k in st:
                w.blockSignals(True)
                w.setChecked(bool(st[k]))
                w.blockSignals(False)
        self._base_prev = self.sp_base.value()
        self.on_grid_changed(self.sp_grid.value())
        self.rebuild_canvas()
        if photo:
            self.photo_bytes, self.photo_name = (photo, meta.get('photo_file') or 'photo.png')
            self.photo_pil = ImageOps.exif_transpose(Image.open(io.BytesIO(photo))).convert('RGB')
        self.update_photo_view()
        for d in meta.get('vec', []):
            self.scene.addItem(VectorItem(PathData.from_dict(d)))
        for x, y, dh in meta.get('ctrl', []):
            self.scene.addItem(CtrlPoint(x, y, dh))
        if 'z' in arrays:
            self.z = arrays['z']
            self.show_mesh(reset=True)
        self.project_path = path
        self.setWindowTitle(f'{APP} — {os.path.basename(path)}')
        self.update_lock()
        self.update_light()
        self.after_change()
        self._saved_sig = self.state_sig()
        self.statusBar().showMessage('Проект открыт: ' + path)

    def closeEvent(self, e):
        if self.confirm_discard():
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            if self.book_dir:
                shutil.rmtree(self.book_dir, ignore_errors=True)
            e.accept()
        else:
            e.ignore()

    def open_help(self):
        if self.book_data is None:
            self.book_dir = tempfile.mkdtemp(prefix='srs_book_')
            self.book_data = build_book_images(self.book_dir)
        d = QDialog(self)
        d.setWindowTitle('Книга инструкций — справочник инструментов')
        d.resize(1080, 720)
        root = QHBoxLayout(d)
        lst = QListWidget()
        lst.setFixedWidth(270)
        for entry in BOOK:
            lst.addItem(entry[1])
        browser = QTextBrowser()
        browser.setSearchPaths([self.book_dir])
        browser.setStyleSheet('background:#ffffff;color:#111111;')
        right = QVBoxLayout()
        right.addWidget(browser)
        close = QPushButton('Закрыть')
        close.clicked.connect(d.accept)
        right.addWidget(close)
        root.addWidget(lst)
        root.addLayout(right, 1)

        def show(i):
            key, title, purpose, steps, mouse, example = BOOK[i]
            cells = ''.join((f'<td align="center" style="padding:4px"><img src="{n}"><br><i>{c}</i></td>' for n, c in self.book_data.get(key, [])))
            html = f'<h2>{title}</h2><p><b>Назначение.</b> {purpose}</p><table><tr>{cells}</tr></table><h3>Пошагово</h3><ol>' + ''.join((f'<li>{s}</li>' for s in steps)) + '</ol><h3>Кнопки мыши и клавиши</h3><ul>' + ''.join((f'<li>{m}</li>' for m in mouse)) + f'</ul><h3>Пример на рельефе</h3><p>{example}</p>'
            browser.setHtml(html)
        lst.currentRowChanged.connect(show)
        lst.setCurrentRow(0)
        d.exec()

def _paint_data(img, data, nodes=True, red=None, fill=False, highlight=None):
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor('#111111'), 1.4)
    p.setPen(pen)
    if fill:
        p.setBrush(QBrush(QColor(20, 20, 20, 40)))
    p.drawPath(build_qpath(data))
    if red is not None:
        sub = PathData(data.nodes, data.closed)
        pts = sub.seg_points(red, 20)
        p.setPen(QPen(QColor('#e02020'), 3))
        p.drawPolyline(QPolygonF([QPointF(*q) for q in pts]))
    if nodes:
        p.setPen(QPen(QColor('#242424'), 1))
        for i, n in enumerate(data.nodes):
            p.setBrush(QBrush(QColor('#ff8a00' if i == highlight else '#ffd25b')))
            p.drawRect(int(n.x) - 4, int(n.y) - 4, 8, 8)
    p.end()

def _petal(scale=320):
    nodes = [Node((0.5 + 0.38 * math.cos(2 * math.pi * i / 8)) * scale, (0.5 + 0.22 * math.sin(2 * math.pi * i / 8)) * scale) for i in range(8)]
    d = PathData(nodes, True)
    for i in range(8):
        d.make_smooth(i)
    return d

def _big(a, k=2):
    return gray_qimage(a).scaled(a.shape[1] * k, a.shape[0] * k, Qt.KeepAspectRatio, Qt.SmoothTransformation).convertToFormat(QImage.Format_RGB32)

def _card(lines):
    img = QImage(320, 200, QImage.Format_RGB32)
    img.fill(QColor('#e9eef7'))
    p = QPainter(img)
    p.setPen(QColor('#222222'))
    y = 45
    for ln in lines:
        p.drawText(20, y, ln)
        y += 34
    p.end()
    return img

def build_book_images(folder):
    out = {}

    def save(key, items):
        res = []
        for i, (img, cap) in enumerate(items):
            name = f'{key}_{i}.png'
            img.save(os.path.join(folder, name))
            res.append((name, cap))
        out[key] = res
    h = demo_relief(160)
    mask = h > 0.05
    base = _big(hillshade(h, 45, 40))
    petal = _petal()
    img_a = base.copy()
    _paint_data(img_a, petal)
    save('draw', [(base.copy(), 'До: только рельеф'), (img_a, 'После: контур лепестка (кривые Безье)')])
    moved = petal.copy()
    moved.nodes[2].x += 15
    moved.nodes[2].y -= 40
    moved.nodes[2].hin = (moved.nodes[2].hin[0] + 15, moved.nodes[2].hin[1] - 40)
    moved.nodes[2].hout = (moved.nodes[2].hout[0] + 15, moved.nodes[2].hout[1] - 40)
    a, b = (base.copy(), base.copy())
    _paint_data(a, petal)
    _paint_data(b, moved, highlight=2)
    save('nodes', [(a, 'До'), (b, 'После: узел и его ручки перетащены')])
    a, b = (base.copy(), base.copy())
    _paint_data(a, petal, red=3)
    cut = petal.remove_segment(3)[0]
    _paint_data(b, cut)
    save('segment', [(a, 'До: красный участок будет удалён'), (b, 'После: остался открытый вектор')])
    n = 160
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
    ph = np.stack([40 + 190 * (1 - np.hypot(xx - 0.5, yy - 0.5) * 1.6).clip(0, 1)] * 3, axis=2)
    a = rgb_qimage(ph).scaled(320, 320).convertToFormat(QImage.Format_RGB32)
    b = rgb_qimage(ph * 0.35 + 255 * 0.65).scaled(320, 320).convertToFormat(QImage.Format_RGB32)
    _paint_data(a, petal, nodes=False)
    _paint_data(b, petal, nodes=False)
    save('fade', [(a, 'Осветление 0%'), (b, 'Осветление 65%: белый фон, векторы чёрные')])
    cell = 100.0 / 159
    S = [(0, 0), (0.3, 0.1), (0.7, 0.9), (1, 1)]
    g1 = [(10, 40), (150, 40)]
    g2 = [(10, 120), (150, 120)]
    cfgs = [('Плавный изгиб', dict(mode='bend', angle=0, a=0, b=1.2)), ('Волна', dict(mode='wave', angle=0, amount=0.5, waves=3)), ('Вершина', dict(mode='apex', amount=0.8, apex_x=0.5, apex_y=0.5, radius=70)), ('Между направляющими', dict(mode='guides', hA=0.0, hB=1.0)), ('Выдавливание по двум направляющим', dict(mode='sweep', hA=0.0, hB=0.0)), ('Управляющие точки', dict(mode='points', influence=25))]
    items = [(base, 'До: плоский лепесток')]
    cov = mask.astype(float)
    for cap, c in cfgs:
        p = dict(c, section=S, fixed=True, feather_mm=5)
        if c['mode'] == 'sweep':
            p['profile'] = lambda v: 0.8 * section_eval([(0, 0), (0.5, 1), (1, 0)], v)
        off = deform_offsets(mask, cell, cell, p, [g1, g2], [(60, 80, 0.9), (110, 80, -0.5)], [g1, g2])
        wgt = cov * (feather_weight(cov, 8) if c['mode'] != 'sweep' else 1.0)
        items.append((_big(hillshade(combine_relief(h, off, wgt, 0.0, 'add'), 45, 40)), 'После: ' + cap))
    save('deform', items)
    t, _ = transform_region(h, mask, 0, 0, 25, 0.7, False, False, True, cell, cell)
    cvm = mask.astype(float)
    shp = lambda kind: build_shape(cvm, cell, cell, kind, 45, 'scale', 0.8)
    plateau = np.full_like(h, 0.45)
    save('editor', [(base, 'До'), (_big(hillshade(combine_relief(h, shp('round'), cvm, 0.0, 'add'), 45, 40)), 'Округлая форма, добавить'), (_big(hillshade(combine_relief(h, shp('angular'), cvm, 0.0, 'add'), 45, 40)), 'Угловая форма, добавить'), (_big(hillshade(combine_relief(h, shp('plateau'), cvm, 0.0, 'add'), 45, 40)), 'Плато, добавить'), (_big(hillshade(combine_relief(h, 0 * h, cvm, 0.0, 'zero'), 45, 40)), 'Обнулить внутри вектора'), (_big(hillshade(combine_relief(h, plateau, cvm, 0.0, 'high'), 45, 40)), 'По наибольшему (плато 0.45)'), (_big(hillshade(combine_relief(h, plateau, cvm, 0.0, 'low'), 45, 40)), 'По наименьшему')])
    ms = np.zeros_like(mask)
    ms[40:121, 10:151] = True
    vv, ww, sp = two_rail_param(ms, cell, cell, [(10, 40), (150, 40)], [(10, 120), (150, 120)])
    arch = lambda hh: lambda q: hh * section_eval([(0, 0), (0.5, 1), (1, 0)], q)

    def sweep_img(profs, pos, **kw):
        out = np.zeros_like(h)
        yy_, xx_ = np.where(ms)
        out[yy_, xx_] = sweep_values(vv, ww, sp, profs, pos, dict(scale_w=False, **kw))
        return _big(hillshade(out, 45, 40))
    save('sweep', [(_big(hillshade(np.zeros_like(h), 45, 40)), 'До: плоскость'), (sweep_img([arch(1.0)], [0.0]), 'Одно сечение'), (sweep_img([arch(1.0), arch(0.3)], [0, 1]), 'Два сечения, с переходом'), (sweep_img([arch(1.0), arch(0.3)], [0, 1], transition=False), 'Два сечения, без перехода')])
    save('transform', [(base, 'До'), (_big(hillshade(t, 45, 40)), 'После: копия, повёрнутая и уменьшенная')])
    cs = hillshade(h, 45, 40).copy()
    for pts, _ in height_contours(h, 0.5):
        for x, y in pts:
            cs[int(round(y)), int(round(x))] = 0
    save('contour', [(base, 'До: рельеф'), (_big(cs), 'После: контур на уровне 50% высоты')])
    save('light', [(_big(hillshade(h, 45, 75)), 'Высокое солнце: детали плоские'), (_big(hillshade(h, 200, 20)), 'Низкое боковое солнце: тени подчёркивают рельеф')])
    save('files', [(_card(['Проект без имени*', 'Ctrl+S — сохранить']), 'До'), (_card(['лепесток.srelief', 'фото+векторы+рельеф']), 'После: можно продолжить позже')])
    return out
BOOK = [('draw', 'Рисование вектора', 'Обвести лепесток, лист или орнамент тонкой чёрной линией (прямые и кривые Безье).', ['Кнопка «✎» слева.', 'ЛКМ — поставить точку. Две точки — прямая.', 'Нажать и потянуть — получится гладкий узел с кривой.', 'ПКМ — закончить открытый вектор (он сохранится). Щелчок по первой точке — замкнуть.', 'После завершения программа переходит в режим выделения; Esc отменяет незаконченное.'], ['ЛКМ — точка (тянуть — кривая)', 'ПКМ — закончить открытый', 'Esc — отмена', 'Колесо — масштаб, средняя кнопка — перемещение холста'], 'Обведите лепесток замкнутым вектором — он станет областью деформации.'), ('nodes', 'Узлы и ручки Безье', 'Точная правка формы вектора.', ['Выделите вектор щелчком по линии (Shift или рамка — несколько).', 'Тяните квадратики-узлы и синие ручки.', 'Двойной щелчок по узлу — гладкий/острый.', 'ПКМ по линии — «Добавить узел здесь», по узлу — «Удалить узел».'], ['ЛКМ — тянуть узел/ручку', 'Alt при тяге ручки — независимо от второй', 'ПКМ — меню', 'Delete — удалить'], 'Подгоните контур по краю лепестка, добавляя узлы в изгибах.'), ('segment', 'Удалить участок', 'Убрать часть линии между двумя соседними точками; остальное остаётся независимыми векторами.', ['ПКМ прямо на нужном сегменте.', 'Выберите «Удалить участок».'], ['ПКМ по сегменту — контекстное меню'], 'Замкнутый контур станет открытой линией, открытый — разделится на две части.'), ('fade', 'Осветление фото до белого фона', 'Плавно превратить подложку в белую, не трогая векторы и рельеф.', ['Ползунок над 2D-окном.', '0 — фото, 100 — белый фон.'], ['Только ползунок'], 'Осветлите на 65% и рисуйте мелкие детали, чёрные линии останутся видны.'), ('deform', 'Деформация поверхности', 'Управлять формой всей области: плавный изгиб, волна, вершина, между направляющими, по управляющим точкам.', ['Выделите замкнутый вектор — это область.', 'Выберите вид; задайте высоты сторон/точек; нарисуйте сечение (форму перехода).', '«Закрепить границы» оставляет край; поле «Слияние» — как форма объединяется с рельефом.', 'Для «Между направляющими» выделите два открытых вектора; для точек — поставьте «◉».', 'Перейдите в «Редактирование 3D», нажмите «Предпросмотр», затем «Применить» или «Отмена».'], ['Ctrl+Z / Ctrl+Y — отмена и повтор', 'Остальная поверхность затемняется в предпросмотре'], 'Рисунки справа получены тем же алгоритмом, что и в программе, на учебном лепестке.'), ('editor', 'Редактор формы (F12)', 'Создать форму внутри выделенных замкнутых векторов (округлая, угловая, плато) и объединить её с рельефом; обнулить рельеф внутри или снаружи векторов.', ['Выделите замкнутые векторы и нажмите F12 (Рельефы → Редактор формы).', 'Выберите форму: округлая, угловая или плато; задайте угол (отрицательный — выемка) и начальную высоту.', 'Высота: без ограничений, ограничить по высоте, масштаб по высоте или постоянная высота.', 'Выберите действие: Добавить, Вычитание, По наибольш., По наименьш., Обнул. внутри, Обнул. снаружи.', '«Предпросмотр» показывает результат, «Применить» меняет рельеф, «Сброс» возвращает настройки.'], ['F12 — открыть окно', 'Ctrl+Z — отмена'], 'Округлый лепесток: угол 45°, «Масштаб по высоте» 0,8 мм, действие «Добавить».'), ('sweep', 'Вытягивание по двум направляющим', 'Создать рельеф вытягиванием одного или нескольких сечений вдоль двух направляющих; результат объединяется с рельефом.', ['Нарисуйте открытые векторы: верхнюю и нижнюю направляющие и одно или несколько сечений.', 'Выделите все векторы и нажмите «Верхняя направляющая — Выбор»: первый — верхняя, второй — нижняя, остальные — сечения.', 'Или выбирайте по одному: «Нижняя направляющая», «Граница по Z» (кнопки «Выбор»), «Добавить сечение».', '«Развернуть направляющие» меняет направление, «Переход между сечениями» плавно смешивает сечения.', '«Масштабировать высоту по ширине», «Начальная высота» и «Масштабировать конечную высоту» задают высоту.', 'Выберите «Объединить» (добавить, вычесть, по наибольшему, по наименьшему) и нажмите «Вычислить».'], ['Рельефы → Вытягиваемые профили', 'Сечение идёт от первой точки вектора к последней, вверх — выше'], 'Одно сечение-арка даёт валик вдоль траектории; два сечения с переходом — валик, плавно меняющий форму.'), ('transform', 'Перемещение готового рельефа', 'Сдвинуть, повернуть, масштабировать, копировать, отразить выбранную часть рельефа.', ['Выделите замкнутый вектор.', 'Задайте сдвиг в мм, угол, масштаб.', '«Переместить» или «Копировать в новое место»; «Отразить» — по горизонтали или вертикали.', 'Флажок «вместе со связанными векторами» переносит и контуры.'], ['Ctrl+Z — отмена'], 'Скопируйте лепесток на другое место модели.'), ('contour', 'Векторы по высоте', 'Создать контур по границе заданного уровня рельефа.', ['Вкладка «Контуры по высоте».', 'Задайте уровень в мм (например 7 при высоте 10).', '«Построить контуры» — появятся обычные редактируемые векторы.'], ['—'], 'Уровень 7 мм при высоте 10 мм даёт линию по верхней части.'), ('light', 'Освещение и материалы', 'Менять вид модели светом; STL при этом не меняется.', ['Вкладка «Свет 3D»: материал, яркость, контраст.', 'Солнце: поворот и высота, значок, выключение.', 'Дополнительный свет можно отключить; «Тени» включает тени (медленнее).'], ['ЛКМ — вращать', 'Колесо — масштаб', 'Shift+ЛКМ — сдвиг'], 'Проверяйте рельеф при низком боковом свете — так его увидят на камне.'), ('files', 'Файлы, история и экспорт', 'Сохранение проекта, импорт DXF/SVG, экспорт SVG и STL в мм.', ['Ctrl+S — сохранить проект (фото, векторы, размеры, настройки, геометрия).', 'Ctrl+O — открыть.', 'Файл → Импорт DXF / SVG; Экспорт SVG, Экспорт STL.', 'Толщину основания можно менять в любой момент.'], ['Ctrl+Z / Ctrl+Y', 'F1 — эта книга'], 'Перед экспортом STL смотрите «Замкнутая модель: да».')]

# ================== Рабочее пространство и запуск ==================

import json
import math
import os
import sys
import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QColor, QFont, QKeySequence, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFontComboBox, QFormLayout, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMenu, QMessageBox, QPushButton, QScrollArea, QToolBar, QToolButton, QVBoxLayout, QWidget
STYLE = '\nQMainWindow, QDialog { background: #e7eced; }\nQWidget { color: #293c49; font-family: "Segoe UI", "DejaVu Sans"; font-size: 12px; }\nQMenuBar { background: #263c49; color: #f6f4ee; padding: 3px 8px; }\nQMenuBar::item { color: #f6f4ee; padding: 6px 12px; background: transparent; }\nQMenuBar::item:selected { background: #405968; border-radius: 3px; }\nQMenu { background: #fffdf8; border: 1px solid #bcc9cd; padding: 5px; }\nQMenu::item { padding: 8px 28px; }\nQMenu::item:selected { background: #e2eeec; }\nQToolBar { background: #f4f5f1; border: 0; border-bottom: 1px solid #c6d0d1;\n           spacing: 3px; padding: 5px; }\nQToolBar::separator { background: #ccd4d4; width: 1px; height: 1px; margin: 5px; }\nQToolButton { border: 1px solid transparent; border-radius: 4px; padding: 6px; }\nQToolButton:hover { background: #e1ebe9; border: 1px solid #bdcfcb; }\nQToolButton:checked { background: #d6e8e3; border: 1px solid #799e94; }\nQPushButton { background: #f9faf6; border: 1px solid #b9c7ca; border-radius: 4px; padding: 7px 10px; }\nQPushButton:hover { background: #e4efeb; border-color: #76998f; }\nQPushButton:checked { background: #d7e8e2; border: 1px solid #71998c; }\nQPushButton[primary="true"] { background: #427d70; color: white; border-color: #427d70; }\nQPushButton[primary="true"]:hover { background: #32695e; }\nQPushButton:disabled, QToolButton:disabled { color: #9ca9ad; }\nQLineEdit, QDoubleSpinBox, QSpinBox, QComboBox { background: #fffefa; border: 1px solid #bbc9cc;\n                                            border-radius: 3px; padding: 5px; min-height: 19px; }\nQDoubleSpinBox:focus, QLineEdit:focus { border-color: #4e8a7e; }\nQTabWidget::pane { border: 1px solid #becacc; background: #f3f5f2; }\nQTabBar::tab { background: #dce4e3; padding: 9px 13px; margin-right: 2px; }\nQTabBar::tab:selected { background: #fffdf7; border-top: 3px solid #b6794c; padding-top: 6px; }\nQDockWidget { font-weight: 600; }\nQDockWidget::title { background: #d6e0df; padding: 8px; }\nQScrollArea { border: none; }\nQListWidget { background: #fffefa; border: 1px solid #c1cdce; border-radius: 3px; }\nQStatusBar { background: #263c49; }\nQStatusBar QLabel { color: #e7efeb; padding: 3px 8px; }\nQStatusBar::item { border: none; }\nQToolTip { background: #fffdf3; color: #243b49; border: 1px solid #aebfbe; padding: 5px; }\nQSlider::groove:horizontal { background: #c5d3d0; height: 5px; border-radius: 2px; }\nQSlider::handle:horizontal { background: #427d70; width: 13px; margin: -5px 0; border-radius: 5px; }\nQCheckBox { spacing: 6px; padding: 3px 0; }\n'

class Ruler(QWidget):

    def __init__(self, studio, vertical=False):
        super().__init__()
        self.studio, self.vertical = (studio, vertical)
        self.setFixedWidth(32) if vertical else self.setFixedHeight(26)

    def paintEvent(self, event):
        v = self.studio.view
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#edf1ee'))
        if not v.K or v.transform().m11() <= 0:
            return
        mm_per_pixel = 1 / (v.K * v.transform().m11())
        target = mm_per_pixel * 65
        base = 10 ** math.floor(math.log10(max(target, 1e-09)))
        step = next((x * base for x in (1, 2, 5, 10) if x * base >= target))
        start = v.mapToScene(0, 0)
        end = v.mapToScene(v.viewport().width(), v.viewport().height())
        lo, hi = (start.y() / v.K, end.y() / v.K) if self.vertical else (start.x() / v.K, end.x() / v.K)
        painter.setFont(QFont('DejaVu Sans', 8))
        painter.setPen(QColor('#718486'))
        minor = step / 5
        first, last = (math.floor(lo / minor), math.ceil(hi / minor))
        for i in range(first, min(last + 1, first + 2000)):
            value = i * minor
            if self.vertical:
                pos = v.mapFromScene(QPointF(0, value * v.K)).y()
            else:
                pos = v.mapFromScene(QPointF(value * v.K, 0)).x()
            major = i % 5 == 0
            length = 10 if major else 4
            if self.vertical:
                painter.drawLine(32 - length, pos, 32, pos)
                if major:
                    painter.save()
                    painter.translate(12, pos + 3)
                    painter.rotate(-90)
                    label = self.studio.sp_h.value() - value
                    painter.drawText(0, 0, f'{label:g}')
                    painter.restore()
            else:
                painter.drawLine(pos, 26 - length, pos, 26)
                if major:
                    painter.drawText(pos + 3, 11, f'{value:g}')
        painter.end()

class StoneStudio(Studio):

    def __init__(self):
        self._shape_kind = None
        self._shape_start = self._shape_preview = None
        self._polygon_sides = 6
        self._stroke = None
        self._preview_tool = None
        self.settings = QSettings('StoneReliefStudio', 'Workspace')
        super().__init__()
        self.setStyleSheet(STYLE)
        self.setWindowIcon(icon('dome', True))
        self.setWindowTitle('Stone Relief Studio — векторы и рельеф')
        self.resize(1440, 940)
        self.sweep_dialog = self._build_sweep_dialog()
        self.relief_dialog = self._build_shape_dialog()
        self._restore_dialog(self.sweep_dialog, 'sweep')
        self._restore_dialog(self.relief_dialog, 'shape')
        try:
            self._restore_parameters(json.loads(str(self.settings.value('parameters', '{}'))), bind=False)
        except (ValueError, TypeError, KeyError):
            pass
        for name, widget in [('geometry', self), ('dockState', None)]:
            data = self.settings.value(name)
            if data:
                self.restoreGeometry(data) if widget else self.restoreState(data)
        self.view.viewport().installEventFilter(self)
        self.view.viewport().setMouseTracking(True)
        self._install_sculpt()
        self._saved_sig = self.state_sig()
        self.statusBar().showMessage('Векторы — слева. Формы и рельеф — сверху. F12 — редактор формы.')

    def _action(self, label, name, slot, shortcut=None, checkable=False):
        action = QAction(icon(name), label, self)
        action.setCheckable(checkable)
        action.setToolTip(label + (f' ({shortcut})' if shortcut else ''))
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(lambda checked=False: slot(checked) if checkable else slot())
        return action

    def _toolbar(self, name, area=Qt.TopToolBarArea, large=False):
        toolbar = QToolBar(name, self)
        toolbar.setObjectName(name)
        toolbar.setMovable(True)
        toolbar.setIconSize(QSize(27 if large else 22, 27 if large else 22))
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon if large else Qt.ToolButtonIconOnly)
        self.addToolBar(area, toolbar)
        return toolbar

    def _build_ui(self):
        super()._build_ui()
        self._legacy_toolbars = list(self.findChildren(QToolBar))
        for toolbar in self._legacy_toolbars:
            self.removeToolBar(toolbar)
            toolbar.hide()
        self.menuBar().clear()
        self.setStyleSheet(STYLE)
        self._shell_actions = {}
        actions = self._shell_actions
        for key, label, glyph, slot, shortcut in [('new', 'Новый проект', 'new', self.new_project, 'Ctrl+N'), ('open', 'Открыть проект…', 'open', self.open_project, 'Ctrl+O'), ('save', 'Сохранить', 'save', self.save_project, 'Ctrl+S'), ('saveas', 'Сохранить как…', 'save', self.save_project_as, 'Ctrl+Shift+S'), ('undo', 'Отменить', 'undo', self.undo, 'Ctrl+Z'), ('redo', 'Повторить', 'redo', self.redo, 'Ctrl+Y'), ('copy', 'Создать копию', 'copy', self.copy_selected, 'Ctrl+D'), ('delete', 'Удалить выделенное', 'delete', self.delete_selected, None), ('transform', 'Размер, поворот и положение…', 'transform', self.transform_dialog, None), ('photo', 'Открыть фотографию…', 'photo', self.open_photo, 'Ctrl+I'), ('svg', 'Импорт SVG 1:1…', 'open', self.import_svg, None), ('dxf', 'Импорт DXF 1:1…', 'open', self.import_dxf, None), ('svgout', 'Экспорт SVG…', 'save', self.export_svg, None), ('stlin', 'Открыть STL…', 'stl', self.open_stl, None), ('stl', 'Экспорт STL…', 'stl', self.export_stl, 'Ctrl+E'), ('shape', 'Форма · F12', 'shape', self.open_relief_editor, 'F12'), ('sweep', 'Две направляющие', 'rails', self.open_sweep_dialog, None), ('dome', 'Купол', 'dome', self.create_dome, None), ('plate', 'Плоская основа', 'plate', self.create_plate, None), ('fromphoto', 'Рельеф из фото', 'photo', self.create_relief, None), ('smooth', 'Сглаживание…', 'smooth', lambda: self.relief_op('smooth'), None), ('zero', 'Обнулить…', 'zero', lambda: self.relief_op('zero'), 'Shift+Delete'), ('fit', 'Показать модель целиком', 'fit', lambda: self.view.fit_all(), 'Home'), ('text', 'Текст в кривые…', 'text', self.add_text, None)]:
            actions[key] = self._action(label, glyph, slot, shortcut)
        menus = {}
        for title in ('Файл', 'Правка', 'Вид', 'Модель', 'Векторы', 'Рельефы', 'Справка'):
            menus[title] = self.menuBar().addMenu(title)
        for key in ('new', 'open', 'save', 'saveas', 'photo', 'svg', 'dxf', 'svgout', 'stlin', 'stl'):
            menus['Файл'].addAction(actions[key])
        menus['Файл'].addSeparator()
        menus['Файл'].addAction('Выход', self.close)
        for key in ('undo', 'redo', 'copy', 'delete', 'transform'):
            menus['Правка'].addAction(actions[key])
        for key in ('plate', 'fromphoto'):
            menus['Модель'].addAction(actions[key])
        for key in ('text', 'svg', 'dxf', 'transform'):
            menus['Векторы'].addAction(actions[key])
        menus['Векторы'].addAction('Соединить два открытых вектора', self.join_selected)
        menus['Векторы'].addAction('Группировать', self.group_selected)
        menus['Векторы'].addAction('Разгруппировать', self.ungroup_selected)
        menus['Векторы'].addAction('Зеркально по горизонтали', lambda: self.mirror_selected('h'))
        menus['Векторы'].addAction('Зеркально по вертикали', lambda: self.mirror_selected('v'))
        for key in ('shape', 'sweep', 'dome', 'smooth', 'zero'):
            menus['Рельефы'].addAction(actions[key])
        for title, op in [('Масштаб высоты…', 'scale'), ('Инвертировать', 'invert'), ('Смещение высоты…', 'offset')]:
            menus['Рельефы'].addAction(title, lambda op=op: self.relief_op(op))
        menus['Рельефы'].addAction(actions['stl'])
        menus['Справка'].addAction(self._action('Справочник инструментов', 'new', self.open_help, 'F1'))
        menus['Справка'].addAction('О программе', lambda: QMessageBox.information(self, 'Stone Relief Studio', 'Векторное рисование и моделирование рельефа для камня.\nСамостоятельное оформление и значки.\nВерсия рабочего пространства 3.0.'))
        filebar = self._toolbar('Проект и правка')
        for key in ('new', 'open', 'save', 'undo', 'redo', 'copy', 'delete', 'transform'):
            filebar.addAction(actions[key])
        filebar.addSeparator()
        brand = QLabel('  STONE  /  RELIEF STUDIO  ')
        brand.setStyleSheet('font-size: 15px; font-weight: 700; color:#466b64; padding:4px 12px;')
        filebar.addWidget(brand)
        self.addToolBarBreak()
        relief = self._toolbar('Создание и редактирование рельефа', large=True)
        for key in ('shape', 'sweep', 'dome', 'plate', 'fromphoto'):
            relief.addAction(actions[key])
        relief.addSeparator()
        self.sculpt_action = self._action('Лепка в 3D', 'sculpt', self.toggle_sculpt, checkable=True)
        relief.addAction(self.sculpt_action)
        menus['Рельефы'].addAction(self.sculpt_action)
        for key in ('smooth', 'zero', 'stl'):
            relief.addAction(actions[key])
        vectors = self._toolbar('Векторы', Qt.LeftToolBarArea)
        vectors.setMovable(False)
        self.tool_group = QActionGroup(self)
        self.tool_actions = {}
        for key, label, glyph in [('select', 'Выделение и перемещение', 'select'), ('draw', 'Линии и кривые Безье', 'pen'), ('ctrl', 'Управляющие точки на чертеже', 'point')]:
            action = self._action(label, glyph, lambda checked, k=key: self.set_tool(k), checkable=True)
            self.tool_group.addAction(action)
            vectors.addAction(action)
            self.tool_actions[key] = action
        vectors.addSeparator()
        for kind, label in [('rectangle', 'Прямоугольник'), ('square', 'Квадрат'), ('circle', 'Круг'), ('ellipse', 'Эллипс'), ('star', 'Звезда'), ('polygon', 'Многоугольник')]:
            a = self._action(label + ' — нарисовать мышью', kind, lambda checked, k=kind: self.choose_shape(k), checkable=True)
            self.tool_group.addAction(a)
            self.tool_actions[kind] = a
            vectors.addAction(a)
        vectors.addAction(actions['text'])
        vectors.addSeparator()
        vectors.addAction(actions['fit'])
        menus['Вид'].addAction(actions['fit'])
        snapbar = self._toolbar('Привязки и точность')
        self.removeToolBar(snapbar)
        self.addToolBar(Qt.BottomToolBarArea, snapbar)
        self.snapbar = snapbar
        snapbar.show()
        self.snap_actions = {}
        for key, label in [('nodes', 'Узлы'), ('lines', 'Линии'), ('grid', 'Сетка'), ('showgrid', 'Показать сетку')]:
            a = QAction(label, self)
            a.setCheckable(True)
            a.toggled.connect(lambda on, k=key: self.on_snap(k, on))
            snapbar.addAction(a)
            self.snap_actions[key] = a
        snapbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        snapbar.addSeparator()
        snapbar.addWidget(QLabel(' Шаг узла: '))
        self.nudge = QComboBox()
        self.nudge.addItems(['0,01 мм', '0,1 мм', '1 мм'])
        self.nudge.setCurrentIndex(1)
        self.nudge.currentIndexChanged.connect(lambda i: setattr(self.view, 'keyboard_nudge_mm', (0.01, 0.1, 1.0)[i]))
        snapbar.addWidget(self.nudge)
        for toolbar in (filebar, relief, vectors, snapbar):
            menus['Вид'].addAction(toolbar.toggleViewAction())
        dock = self.findChildren(QDockWidget)[0]
        dock.setObjectName('inspector')
        dock.setWindowTitle('Модель и инструменты')
        dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setMinimumWidth(290)
        self.inspector = dock
        self.inspector_tabs = dock.widget()
        self.inspector_tabs.setTabPosition(QTabWidget.North)
        self.inspector_tabs.setUsesScrollButtons(True)
        model = QWidget()
        form = QFormLayout(model)
        form.setContentsMargins(14, 14, 14, 14)
        title = QLabel('РАЗМЕРЫ И ОСНОВА')
        title.setStyleSheet('font-weight:700;color:#628074;')
        form.addRow(title)
        for label, widget in [('Ширина', self.sp_w), ('Высота', self.sp_h), ('Основание', self.sp_base), ('Высота из фото', self.sp_depth), ('Шаг рельефа', self.sp_step), ('Шаг привязки', self.sp_grid)]:
            form.addRow(label, widget)
            widget.show()
        form.addRow(self.chk_invert)
        form.addRow('Сглаживание фото', self.sp_blur)
        self.chk_invert.show()
        self.sp_blur.show()
        for key in ('plate', 'photo', 'fromphoto', 'svg', 'dxf'):
            b = QPushButton(actions[key].text())
            b.setIcon(actions[key].icon())
            b.clicked.connect(actions[key].trigger)
            form.addRow(b)
        hint = QLabel('Размеры — в миллиметрах.\nКривые: ЛКМ — узел, протянуть — ручки.\nEnter — завершить, Esc — отменить.\nУзлы: Tab, стрелки; Ctrl — 0,01 мм.')
        hint.setWordWrap(True)
        hint.setStyleSheet('color:#60777c;padding-top:10px;')
        form.addRow(hint)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(model)
        self.inspector_tabs.insertTab(0, scroll, 'Модель')
        self.inspector_tabs.setCurrentIndex(0)
        for label in self.inspector_tabs.findChildren(QLabel):
            if len(label.text()) > 60:
                label.setWordWrap(True)
        for combo in self.inspector_tabs.findChildren(QComboBox):
            combo.setMinimumContentsLength(15)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        menus['Вид'].addAction(dock.toggleViewAction())
        menus['Модель'].addAction('Размеры и параметры', lambda: self.show_inspector(0))
        self.view.setBackgroundBrush(QColor('#75858b'))
        parent_layout = self.tabs.widget(0).layout()
        parent_layout.removeWidget(self.view)
        wrap = QWidget()
        grid = QGridLayout(wrap)
        grid.setSpacing(0)
        grid.setContentsMargins(0, 0, 0, 0)
        self.ruler_x = Ruler(self)
        self.ruler_y = Ruler(self, True)
        corner = QLabel('мм')
        corner.setAlignment(Qt.AlignCenter)
        corner.setStyleSheet('background:#edf1ee;color:#748689;')
        grid.addWidget(corner, 0, 0)
        grid.addWidget(self.ruler_x, 0, 1)
        grid.addWidget(self.ruler_y, 1, 0)
        grid.addWidget(self.view, 1, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(1, 1)
        parent_layout.addWidget(wrap, 1)
        self.view.horizontalScrollBar().valueChanged.connect(self.update_rulers)
        self.view.verticalScrollBar().valueChanged.connect(self.update_rulers)
        self.coords = QLabel('X 0,00   Y 0,00 мм')
        self.statusBar().addPermanentWidget(self.coords)
        self.tabs.setTabText(0, 'Чертёж · 2D')
        self.tabs.setTabText(1, 'Поверхность · 3D')
        self.plotter.set_background('#64777c', top='#b7c6c5')
        camera_row = self.tabs.widget(1).layout().itemAt(0).layout()
        while camera_row.count():
            item = camera_row.takeAt(0)
            if item.widget() and item.widget() is not self.cb_mode3d:
                item.widget().deleteLater()
        camera_row.addWidget(self.cb_mode3d)
        self.cb_mode3d.show()
        camera = QComboBox()
        camera.addItems(['Под углом', 'Сверху', 'Спереди', 'Слева', 'Справа', 'Сзади', 'Снизу'])
        camera.currentIndexChanged.connect(lambda i: self.set_view(('iso', 'top', 'front', 'left', 'right', 'back', 'bottom')[i]))
        camera_row.addWidget(camera)
        camera_row.addStretch(1)
        camera_row.addWidget(QLabel('Мышь — вращение · колесо — масштаб'))
        self.sp_w.setValue(300)
        self.sp_h.setValue(500)

    def show_inspector(self, index):
        self.inspector.show()
        self.inspector.raise_()
        self.inspector_tabs.setCurrentIndex(index)

    def update_rulers(self, *_):
        self.ruler_x.update()
        self.ruler_y.update()

    def set_tool(self, tool):
        self._shape_kind = None
        self._shape_start = None
        self._clear_shape_preview()
        super().set_tool(tool)

    def choose_shape(self, kind):
        if kind == 'polygon':
            sides, ok = QInputDialog.getInt(self, 'Многоугольник', 'Количество сторон:', self._polygon_sides, 3, 64)
            if not ok:
                self.set_tool('select')
                return
            self._polygon_sides = sides
        self._shape_kind = kind
        self._shape_start = None
        self._clear_shape_preview()
        self.view.set_mode('shape')
        self.statusBar().showMessage('Нажмите и протяните мышью. Esc — отмена. Размер можно уточнить через «Правка → Размер, поворот и положение».')

    def _clear_shape_preview(self):
        if self._shape_preview is not None:
            self.scene.removeItem(self._shape_preview)
            self._shape_preview = None

    def shape_data(self, a, b):
        x1, y1, x2, y2 = (a.x(), a.y(), b.x(), b.y())
        if self._shape_kind in ('square', 'circle'):
            size = max(abs(x2 - x1), abs(y2 - y1))
            x2 = x1 + math.copysign(size, x2 - x1)
            y2 = y1 + math.copysign(size, y2 - y1)
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        cx, cy = ((left + right) / 2, (top + bottom) / 2)
        rx, ry = ((right - left) / 2, (bottom - top) / 2)
        if self._shape_kind in ('circle', 'ellipse'):
            return _ellipse_path(cx, cy, rx, ry)
        if self._shape_kind in ('square', 'rectangle'):
            points = [(left, top), (right, top), (right, bottom), (left, bottom)]
        else:
            count = 10 if self._shape_kind == 'star' else self._polygon_sides
            points = []
            for i in range(count):
                scale = 0.43 if self._shape_kind == 'star' and i % 2 else 1
                angle = -math.pi / 2 + 2 * math.pi * i / count
                points.append((cx + rx * scale * math.cos(angle), cy + ry * scale * math.sin(angle)))
        return polyline_path(points, True)

    def eventFilter(self, obj, event):
        if obj is self.view.viewport():
            typ = event.type()
            if typ in (QEvent.Resize, QEvent.Wheel, QEvent.Paint):
                if typ != QEvent.Paint:
                    QTimer.singleShot(0, self.update_rulers)
            if typ == QEvent.MouseMove:
                p = self.view.mapToScene(event.position().toPoint())
                self.coords.setText(f'X {p.x() / self.K:.2f}   Y {self.sp_h.value() - p.y() / self.K:.2f} мм')
            if self._shape_kind:
                if typ == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    self._shape_start = self.view.snap_point(self.view.mapToScene(event.position().toPoint()))
                    return True
                if typ == QEvent.MouseMove and self._shape_start is not None:
                    b = self.view.snap_point(self.view.mapToScene(event.position().toPoint()))
                    if self._shape_preview is None:
                        self._shape_preview = QGraphicsPathItem()
                        pen = QPen(QColor('#427d70'), 1, Qt.DashLine)
                        pen.setCosmetic(True)
                        self._shape_preview.setPen(pen)
                        self._shape_preview.setZValue(90)
                        self.scene.addItem(self._shape_preview)
                    self._shape_preview.setPath(build_qpath(self.shape_data(self._shape_start, b)))
                    return True
                if typ == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                    if self._shape_start is not None:
                        b = self.view.snap_point(self.view.mapToScene(event.position().toPoint()))
                        data = self.shape_data(self._shape_start, b)
                        bounds = build_qpath(data).boundingRect()
                        if bounds.width() > 0.0001 and bounds.height() > 0.0001:
                            snap = self.snapshot()
                            self.scene.clearSelection()
                            self._add_vec(data, True)
                            self.push_history('Создание фигуры', snap)
                            self.after_change()
                    self.set_tool('select')
                    return True
                if typ == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                    self.set_tool('select')
                    return True
        if obj in (getattr(self, 'sweep_dialog', None), getattr(self, 'relief_dialog', None)):
            if event.type() in (QEvent.Hide, QEvent.Close):
                name = 'sweep' if obj is self.sweep_dialog else 'shape'
                self.settings.setValue(name + '/geometry', obj.saveGeometry())
                self.settings.setValue('parameters', json.dumps(self._parameters(), ensure_ascii=False))
        return super().eventFilter(obj, event)

    def add_text(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('Надпись → векторные контуры')
        form = QFormLayout(dialog)
        text = QLineEdit('Название')
        font = QFontComboBox()
        height = self._spin(15, 0.5, 500, 2)
        bold = QCheckBox('Полужирный')
        form.addRow('Текст', text)
        form.addRow('Шрифт', font)
        form.addRow('Высота букв', height)
        form.addRow(bold)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.Accepted or not text.text().strip():
            return
        selected_font = font.currentFont()
        selected_font.setPixelSize(100)
        selected_font.setBold(bold.isChecked())
        path = QPainterPath()
        path.addText(0, 0, selected_font, text.text())
        bounds = path.boundingRect()
        if bounds.height() == 0:
            return
        scale = height.value() * self.K / bounds.height()
        paths = []
        nodes = []
        i = 0

        def flush():
            nonlocal nodes
            if len(nodes) >= 2:
                if math.hypot(nodes[0].x - nodes[-1].x, nodes[0].y - nodes[-1].y) < 1e-07:
                    nodes[0].hin = nodes[-1].hin
                    nodes.pop()
                paths.append(PathData(nodes, True))
            nodes = []
        xy = lambda e: ((e.x - bounds.left()) * scale + self.canvas_w * 0.1, (e.y - bounds.top()) * scale + self.canvas_h * 0.3)
        while i < path.elementCount():
            e = path.elementAt(i)
            if e.isMoveTo():
                flush()
                nodes = [Node(*xy(e))]
            elif e.isLineTo():
                nodes.append(Node(*xy(e)))
            elif e.isCurveTo():
                c2 = path.elementAt(i + 1)
                end = path.elementAt(i + 2)
                nodes[-1].hout = xy(e)
                nodes.append(Node(*xy(end), hin=xy(c2)))
                i += 2
            i += 1
        flush()
        snap = self.snapshot()
        self.scene.clearSelection()
        group = self._new_gid()
        for data in paths:
            data.group = group
            self._add_vec(data, True)
        self.push_history('Текст в кривые', snap)
        self.after_change()
        self.tabs.setCurrentIndex(0)

    def _build_sweep_dialog(self):
        dialog = super()._build_sweep_dialog()
        dialog.setWindowTitle('Поверхность по двум направляющим')
        dialog.setSizeGripEnabled(True)
        dialog.resize(560, 720)
        dialog.setModal(False)
        intro = dialog.findChildren(QLabel)[0]
        intro.setText('1. Выберите две открытые направляющие.\n2. Добавьте поперечные сечения и задайте их положение.\n3. Проверьте поверхность в 3D и примените результат.')
        for b in dialog.findChildren(QPushButton):
            if b.text() == 'Вычислить':
                b.setText('Применить')
                b.setProperty('primary', True)
            if b.text().startswith('Положение выбранного'):
                b.setText('Положение сечения вдоль кривых, %')
        scroll = dialog.findChild(QScrollArea)
        layout = scroll.widget().layout()
        footer = layout.takeAt(layout.count() - 1).layout()
        row = QHBoxLayout()
        while footer.count():
            item = footer.takeAt(0)
            if item.widget():
                row.addWidget(item.widget())
        dialog.layout().addLayout(row)
        dialog.installEventFilter(self)
        return dialog

    def _build_shape_dialog(self):
        dialog = super()._build_shape_dialog()
        dialog.setWindowTitle('Форма и высота · F12')
        dialog.setModal(False)
        dialog.setSizeGripEnabled(True)
        dialog.setMinimumSize(440, 480)
        dialog.resize(590, 630)
        content = QWidget()
        content.setLayout(dialog.layout())
        layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        for i, b in enumerate(self.sh_kind_btns):
            b.setText('')
            b.setIcon(icon(('dome', 'polygon', 'plate')[i], True))
            b.setIconSize(QSize(34, 34))
        for b in dialog.findChildren(QPushButton):
            if b.text() == 'Применить':
                b.setProperty('primary', True)
        dialog.installEventFilter(self)
        return dialog

    def _restore_dialog(self, dialog, name):
        geometry = self.settings.value(name + '/geometry')
        if geometry:
            dialog.restoreGeometry(geometry)

    def sweep_state(self):
        if not hasattr(self, 'sw_flip1'):
            return {}
        uid = lambda item: item.data.uid if item is not None and self._alive(item) else None
        return dict(rails=[uid(x) for x in self.sw_rails], sections=[uid(x) for x in self.sw_sections], z=uid(self.sw_zb), positions=dict(self.sw_positions), flip1=self.sw_flip1.isChecked(), flip2=self.sw_flip2.isChecked(), use_z=self.sw_use_z.isChecked(), transition=self.sw_trans.isChecked(), scale_w=self.sw_scalew.isChecked(), scale_end=self.sw_scaleend.isChecked(), start=self.sw_start.value(), end=self.sw_end.value(), mode=next((b.property('key') for b in self.sw_mode_btns if b.isChecked()), 'add'))

    def _parameters(self):
        form = {}
        if hasattr(self, 'sh_kind'):
            form = dict(kind=self.sh_kind, angle=self.sh_angle.value(), start=self.sh_start.value(), scale=self.sh_scale_sl.value(), limit=self.sh_limit.value(), height=self.sh_scaleh.value(), const=self.sh_const.value(), height_mode=self.sh_height_mode()[0], combine=self.sh_mode_key())
        return dict(sweep=self.sweep_state(), form=form)

    def _restore_parameters(self, data, bind=True):
        state = data.get('sweep', {})
        if hasattr(self, 'sw_flip1'):
            for key, widget in [('flip1', self.sw_flip1), ('flip2', self.sw_flip2), ('use_z', self.sw_use_z), ('transition', self.sw_trans), ('scale_w', self.sw_scalew), ('scale_end', self.sw_scaleend)]:
                if key in state:
                    widget.setChecked(state[key])
            self.sw_start.setValue(state.get('start', 0))
            self.sw_end.setValue(state.get('end', 0))
            for b in self.sw_mode_btns:
                b.setChecked(b.property('key') == state.get('mode', 'add'))
            if bind:
                vectors = {v.data.uid: v for v in self.view.vectors()}
                self.sw_rails = [vectors.get(uid) for uid in state.get('rails', [None, None])]
                self.sw_sections = [vectors[uid] for uid in state.get('sections', []) if uid in vectors]
                self.sw_zb = vectors.get(state.get('z'))
                self.sw_positions = dict(state.get('positions', {}))
                self.sw_refresh()
        form = data.get('form', {})
        if form and hasattr(self, 'sh_kind'):
            self.sh_set_kind(int(form.get('kind', 0)))
            for key, widget in [('angle', self.sh_angle), ('start', self.sh_start), ('scale', self.sh_scale_sl), ('limit', self.sh_limit), ('height', self.sh_scaleh), ('const', self.sh_const)]:
                if key in form:
                    widget.setValue(form[key])
            mode = form.get('height_mode', 'free')
            {'free': self.sh_r_free, 'limit': self.sh_r_limit, 'scale': self.sh_r_scale, 'const': self.sh_r_const}.get(mode, self.sh_r_free).setChecked(True)
            for (_, key), b in zip(self.SH_MODES, self.sh_mode_btns):
                b.setChecked(key == form.get('combine', 'add'))

    def _settings(self):
        return dict(super()._settings(), workspace=self._parameters())

    def state_sig(self):
        import hashlib
        return hashlib.sha256((super().state_sig() + json.dumps(self._parameters(), sort_keys=True)).encode()).hexdigest()

    def snapshot(self):
        result = super().snapshot()
        result.update(photo=(self.photo_pil, self.photo_bytes, self.photo_name), mesh=self.stl_mesh, dims=(self.sp_w.value(), self.sp_h.value(), self.sp_step.value()), selected=[v.data.uid for v in self.sel_vectors()], parameters=self._parameters())
        return result

    def restore(self, state):
        for widget, value in zip((self.sp_w, self.sp_h, self.sp_step), state.get('dims', ())):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self.photo_pil, self.photo_bytes, self.photo_name = state.get('photo', (None, None, None))
        self.photo_arr = None
        self.stl_mesh = state.get('mesh')
        super().restore(state)
        self.rebuild_canvas(force=True)
        self._restore_parameters(state.get('parameters', {}))
        selected = set(state.get('selected', []))
        for v in self.view.vectors():
            v.setSelected(v.data.uid in selected)
        self._v_key = None

    def open_project(self, path=None):
        if not isinstance(path, str):
            path, _ = QFileDialog.getOpenFileName(self, 'Открыть проект', '', 'Stone Relief (*.srelief)')
        if not path:
            return
        super().open_project(path)
        if self.project_path == path:
            meta, _, _ = unpack_project(path)
            self._restore_parameters(meta.get('settings', {}).get('workspace', {}))
            self.update_rulers()
            self._saved_sig = self.state_sig()

    def new_project(self):
        super().new_project()
        if not self.has_content():
            self.sw_reset()
            self._v_key = None

    def preview_deform(self):
        self._preview_tool = 'deform'
        super().preview_deform()

    def shape_apply(self, preview=False):
        self._preview_tool = 'shape'
        super().shape_apply(preview)

    def sw_run(self, preview):
        self._preview_tool = 'sweep'
        super().sw_run(preview)

    def _refresh_preview(self):
        if not self.preview_active:
            return
        if self._preview_tool == 'shape':
            self.shape_apply(True)
        elif self._preview_tool == 'sweep':
            self.sw_run(True)
        else:
            super()._refresh_preview()

    def _install_sculpt(self):
        page = QWidget()
        form = QFormLayout(page)
        helptext = QLabel('Выберите замкнутый контур лепестка в 2D, затем включите «Лепка в 3D».\n\nНажмите на поверхность и тяните мышь вверх или вниз. Мелкая фактура сохраняется: изменяется только высота.\n\nБез контура изменяется область кисти. Ctrl+Z отменяет весь жест.')
        helptext.setWordWrap(True)
        form.addRow(helptext)
        self.sculpt_radius = self._spin(15, 0.5, 500, 2)
        self.sculpt_gain = self._spin(0.05, 0.001, 5, 3, ' мм/px', 0.01)
        self.sculpt_fixed = QCheckBox('Зафиксировать край выделенного контура')
        self.sculpt_fixed.setChecked(True)
        form.addRow('Радиус', self.sculpt_radius)
        form.addRow('Сила', self.sculpt_gain)
        form.addRow(self.sculpt_fixed)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.sculpt_tab = self.inspector_tabs.addTab(scroll, 'Лепка')
        self.resizeDocks([self.inspector], [330], Qt.Horizontal)
        self._sculpt_timer = QTimer(self)
        self._sculpt_timer.setSingleShot(True)
        self._sculpt_timer.timeout.connect(self.show_mesh)
        for event, callback in [('LeftButtonPressEvent', self._sculpt_press), ('MouseMoveEvent', self._sculpt_move), ('LeftButtonReleaseEvent', self._sculpt_release)]:
            self.plotter.iren.interactor.AddObserver(event, callback, 1.0)

    def toggle_sculpt(self, enabled):
        if enabled and self.z is None:
            self.sculpt_action.setChecked(False)
            QMessageBox.information(self, 'Лепка', 'Создайте плоскую основу, рельеф из фото или откройте STL.')
            return
        self._sculpt_release()
        interactor = self.plotter.iren.interactor
        if enabled:
            self._camera_style = interactor.GetInteractorStyle()
            from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser
            self._sculpt_style = vtkInteractorStyleUser()
            interactor.SetInteractorStyle(self._sculpt_style)
        elif hasattr(self, '_camera_style'):
            interactor.SetInteractorStyle(self._camera_style)
        if enabled:
            self.cancel_preview()
            self.tabs.setCurrentIndex(1)
            self.show_inspector(self.sculpt_tab)
            self.statusBar().showMessage('ЛЕПКА · нажмите на поверхность и тяните вверх / вниз. Для вращения выключите «Лепка в 3D».')
        else:
            self.statusBar().showMessage('Просмотр 3D: вращение мышью, колесо — масштаб.')

    def _sculpt_press(self, *_):
        if not self.sculpt_action.isChecked() or self.z is None:
            return
        from vtkmodules.vtkRenderingCore import vtkCellPicker
        x, y = self.plotter.iren.interactor.GetEventPosition()
        picker = vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.PickFromListOn()
        picker.AddPickList(self.actor)
        if not picker.Pick(x, y, 0, self.plotter.renderer):
            return
        px, py, _ = picker.GetPickPosition()
        cx, cy = self.cell()
        center = ((px + self.sp_w.value() / 2) / cx, (py + self.sp_h.value() / 2) / cy)
        mask = None
        if self._closed_polys():
            mask = self.region_cov() > 0.5
        weight = sculpt_weight(self.z.shape, center, (cx, cy), self.sculpt_radius.value(), mask, self.sculpt_fixed.isChecked())
        self._stroke = dict(y=y, weight=weight, base=self.z, snapshot=self.snapshot(), changed=False)

    def _sculpt_move(self, *_):
        if not self._stroke:
            return
        _, y = self.plotter.iren.interactor.GetEventPosition()
        delta = (y - self._stroke['y']) * self.sculpt_gain.value()
        self.z = np.maximum(self._stroke['base'] + delta * self._stroke['weight'], 0.05)
        self._stroke['changed'] = abs(delta) > 1e-08
        if not self._sculpt_timer.isActive():
            self._sculpt_timer.start(60)

    def _sculpt_release(self, *_):
        if not self._stroke:
            return
        stroke, self._stroke = (self._stroke, None)
        self._sculpt_timer.stop()
        if stroke['changed']:
            self.push_history('Лепка поверхности', stroke['snapshot'])
            self.after_change()
        self.show_mesh()

    def closeEvent(self, event):
        if not self.confirm_discard():
            event.ignore()
            return
        self._sculpt_release()
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('dockState', self.saveState())
        for key, dialog in [('sweep', self.sweep_dialog), ('shape', self.relief_dialog)]:
            self.settings.setValue(key + '/geometry', dialog.saveGeometry())
        self.settings.setValue('parameters', json.dumps(self._parameters(), ensure_ascii=False))
        self.settings.sync()
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        if self.book_dir:
            shutil.rmtree(self.book_dir, ignore_errors=True)
        self.plotter.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName('Stone Relief Studio')
    app.setOrganizationName('StoneReliefStudio')
    window = StoneStudio()
    window.show()
    arg = next((p for p in sys.argv[1:] if p.lower().endswith(EXT)), None)
    if arg:
        QTimer.singleShot(0, lambda: window.open_project(arg))
    sys.exit(app.exec())
if __name__ == '__main__':
    main()
