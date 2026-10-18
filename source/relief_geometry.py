"""Continuous surface coordinates and additive sculpting, in millimetres."""
import io
import math
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.ndimage import distance_transform_edt, gaussian_filter
from input_limits import open_checked_image


def photo_heights(raw, nx, ny, blur=0, invert=False):
    """Keep the source height precision; RGB previews never feed this calculation."""
    original = open_checked_image(raw)
    tags = getattr(original, 'tag_v2', {})
    bits, sample_format = tags.get(258), tags.get(339)
    if isinstance(bits, tuple): bits = bits[0]
    if isinstance(sample_format, tuple): sample_format = sample_format[0]
    # Some Pillow versions expose 16-bit PNG as mode I (signed int32 container).
    if raw[:8] == b'\x89PNG\r\n\x1a\n' and len(raw) > 24:
        bits, sample_format = raw[24], 1
    image = ImageOps.exif_transpose(original)
    if image.mode in ('I;16', 'I;16L', 'I;16B', 'I', 'F'):
        array = np.asarray(image)
        if not np.isfinite(array).all():
            raise ValueError('Карта высот содержит NaN или бесконечность.')
        if array.dtype.kind == 'f':
            values = array.astype(np.float64)
            if values.min() < 0 or values.max() > 1:
                minimum, maximum = float(values.min()), float(values.max())
                values -= minimum
                values /= max(maximum-minimum, 1e-12)
        else:
            bits = int(bits or array.dtype.itemsize*8)
            if bits < 1 or bits > 64:
                raise ValueError('Неподдерживаемая разрядность карты высот.')
            signed = sample_format == 2 if sample_format is not None else array.dtype.kind == 'i'
            # Unsigned 32-bit TIFF can be decoded into Pillow's signed I container.
            if not signed and array.dtype.kind == 'i' and bits == array.dtype.itemsize*8:
                array = array.view(np.dtype(array.dtype.str.replace('i','u')))
            values = array.astype(np.float64)
            minimum = -(2**(bits-1)) if signed else 0
            maximum = 2**(bits-1)-1 if signed else 2**bits-1
            values -= minimum
            values /= maximum-minimum
    else:
        if 'A' in image.getbands() or 'transparency' in image.info:
            rgba = image.convert('RGBA')
            bg = Image.new('RGBA', rgba.size, 'white')
            image = Image.alpha_composite(bg, rgba).convert('RGB')
        values = np.asarray(image.convert('L'), dtype=np.float32)/255.0
    if not np.isfinite(values).all():
        raise ValueError('Карта высот содержит недопустимые значения.')
    values = np.asarray(values, dtype=np.float32)
    if blur > 0:
        values = gaussian_filter(values, blur)
    resized = Image.fromarray(values).resize((nx, ny), Image.Resampling.LANCZOS)
    result = np.flipud(np.clip(np.asarray(resized, dtype=np.float32), 0, 1))
    return 1-result if invert else result


def project_ruled(px, py, a, b):
    """Invert neighbouring bilinear patches instead of quantizing to a rail sample."""
    points = np.column_stack((px, py))
    d = b-a
    dd = np.maximum(np.sum(d*d, axis=1), 1e-18)
    outputs = []
    for start in range(0, len(points), 800):
        q = points[start:start+800]
        # Exact nearest transverse segment; bound temporaries independently of
        # rail sample count. Equal-distance ties keep the first sample as before.
        nearest = np.full(len(q), np.inf)
        k = np.zeros(len(q), dtype=np.int64)
        initial_v = np.zeros(len(q))
        for offset in range(0, len(a), 256):
            section = slice(offset, offset+256)
            r = q[:, None, :]-a[None, section, :]
            v = np.clip(np.sum(r*d[None, section, :], axis=2)/dd[None, section], 0, 1)
            err = np.sum((r-v[:, :, None]*d[None, section, :])**2, axis=2)
            local_k = np.argmin(err, axis=1)
            values = err[np.arange(len(q)), local_k]
            take = values < nearest
            nearest[take], k[take] = values[take], local_k[take]+offset
            initial_v[take] = v[np.arange(len(q)), local_k][take]
        best = np.full(len(q), np.inf)
        best_v = np.zeros(len(q))
        best_s = np.zeros(len(q))
        best_w = np.zeros(len(q))
        for j in (np.maximum(k-1, 0), np.minimum(k, len(a)-2)):
            aa, da = a[j], a[j+1]-a[j]
            d0, delta = d[j], d[j+1]-d[j]
            t = np.full(len(q), .5)
            u = initial_v.copy()
            for _ in range(8):
                dv = d0+t[:, None]*delta
                ds = da+u[:, None]*delta
                residual = aa+t[:, None]*da+u[:, None]*dv-q
                det = ds[:, 0]*dv[:, 1]-ds[:, 1]*dv[:, 0]
                good = np.abs(det) > 1e-14
                safe = np.where(good, det, 1)
                dt = (residual[:, 0]*dv[:, 1]-residual[:, 1]*dv[:, 0])/safe
                du = (ds[:, 0]*residual[:, 1]-ds[:, 1]*residual[:, 0])/safe
                t = np.clip(t-np.where(good, dt, 0), 0, 1)
                u = np.clip(u-np.where(good, du, 0), 0, 1)
            width_vec = d0+t[:, None]*delta
            err = np.sum((aa+t[:, None]*da+u[:, None]*width_vec-q)**2, axis=1)
            take = err < best
            best[take] = err[take]
            best_v[take] = u[take]
            best_s[take] = (j[take]+t[take])/(len(a)-1)
            best_w[take] = np.linalg.norm(width_vec[take], axis=1)
        outputs.append((best_v, best_w, best_s))
    if not outputs:
        return np.empty(0), np.empty(0), np.empty(0)
    return tuple(np.concatenate([out[i] for out in outputs]) for i in range(3))


def coverage(polys, nx, ny, ss=None):
    """Union overlapping regions; alternate fill for completely nested contours."""
    ss = ss or (4 if nx*ny <= 1200000 else 2)
    shapes = [Polygon(p) for p in polys if len(p) >= 3]
    if any(not p.is_valid for p in shapes):
        raise ValueError('Контур пересекает сам себя. Исправьте узлы перед построением рельефа.')
    ordered = []
    for shape in sorted(shapes, key=lambda p:p.area, reverse=True):
        if not any(shape.equals(previous) for previous in ordered):
            ordered.append(shape)
    children = [[] for _ in ordered]
    roots = []
    for index, shape in enumerate(ordered):
        parents = [j for j in range(index) if shape.within(ordered[j])]
        if parents:
            children[min(parents, key=lambda j:ordered[j].area)].append(index)
        else:
            roots.append(index)
    regions = list(ordered)
    for index in range(len(ordered)-1, -1, -1):
        if children[index]:
            regions[index] = ordered[index].difference(unary_union([regions[j] for j in children[index]]))
    geometry = unary_union([regions[j] for j in roots])
    image = Image.new('L', (nx*ss, ny*ss), 0)
    draw = ImageDraw.Draw(image)
    def paint(polygon):
        if polygon.geom_type == 'Polygon':
            convert = lambda ring: [((x+.5)*ss-.5, (y+.5)*ss-.5) for x,y in ring.coords]
            draw.polygon(convert(polygon.exterior), fill=255)
            for ring in polygon.interiors:
                draw.polygon(convert(ring), fill=0)
        elif hasattr(polygon, 'geoms'):
            for child in polygon.geoms:
                paint(child)
    paint(geometry)
    return np.asarray(image).reshape(ny, ss, nx, ss).sum(axis=(1, 3), dtype=np.float32)/(255*ss*ss)


def sculpt_weight(shape, center, cell, radius, mask=None, fixed=True):
    """Compact-support C2 displacement. Adds height without resampling fine detail."""
    x, y = center
    cx, cy = cell
    ny, nx = shape
    x0, x1 = max(0, int(math.floor(x-radius/cx))), min(nx, int(math.ceil(x+radius/cx))+1)
    y0, y1 = max(0, int(math.floor(y-radius/cy))), min(ny, int(math.ceil(y+radius/cy))+1)
    weight = np.zeros(shape, dtype=np.float32)
    if x0 >= x1 or y0 >= y1:
        return weight
    dx = (np.arange(x0, x1, dtype=np.float32)-x)*cx
    dy = (np.arange(y0, y1, dtype=np.float32)-y)*cy
    local = dy[:, None]**2+dx[None, :]**2
    local /= -max(radius*radius, 1e-12)
    local += 1
    np.maximum(local, 0, out=local)
    np.power(local, 3, out=local)
    weight[y0:y1, x0:x1] = local
    if mask is not None:
        weight *= mask
        if fixed:
            distance = distance_transform_edt(np.pad(mask, 1), sampling=(cy, cx))[1:-1, 1:-1]
            # First ring is exactly pinned; smooth transition over 20% of brush radius.
            t = np.clip((distance-max(cx, cy))/max(radius*.2, max(cx, cy)), 0, 1)
            weight *= t*t*(3-2*t)
    return weight


def history_storage_bytes(*stacks):
    """Estimate retained model buffers; shared arrays and photos are counted once."""
    seen = set()
    def size(value):
        if isinstance(value, np.ndarray):
            while isinstance(value.base, np.ndarray): value = value.base
            if id(value) in seen: return 0
            seen.add(id(value))
            return value.nbytes
        if id(value) in seen: return 0
        seen.add(id(value))
        if isinstance(value, dict): return sum(size(item) for item in value.values())
        if isinstance(value, (list, tuple)): return sum(size(item) for item in value)
        if isinstance(value, bytes): return len(value)
        if isinstance(value, Image.Image):
            return value.width*value.height*len(value.getbands())*(4 if value.mode in ('I','F') else 1)
        if hasattr(value, 'vertices') and hasattr(value, 'faces'):
            return size(value.vertices)+size(value.faces)
        return 0
    return sum(size(stack) for stack in stacks)


def trim_history_stacks(undo, redo, budget, max_steps=25):
    """Keep nearby undo/redo states under a soft budget, retaining one last state."""
    removed = 0
    for stack in (undo, redo):
        while len(stack) > max_steps:
            stack.pop(0); removed += 1
    while len(undo)+len(redo) > 1 and history_storage_bytes(undo, redo) > budget:
        (undo if len(undo) >= len(redo) else redo).pop(0)
        removed += 1
    return removed
