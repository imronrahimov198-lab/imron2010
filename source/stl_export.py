"""Validated, atomic STL export with bounded temporary memory."""
from dataclasses import dataclass
from contextlib import ExitStack
from pathlib import Path as FilePath
import math
import os
import shutil
import sqlite3
import struct
import tempfile

import numpy as np


class ExportCancelled(Exception):
    """The user cancelled before the destination was replaced."""


@dataclass(frozen=True)
class ExportReport:
    triangles: int
    dimensions: tuple
    volume: float


_STL_RAM_FACE_LIMIT = 200000

_STL_RECORD = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])


def _export_progress(callback, phase, current, total):
    if callback:
        callback(phase, current, total)


def validate_stl_mesh(vertices, faces, progress=None, chunk_size=8192, precision_tolerance=.001):
    """Check indexed geometry; bound the RAM edge index and spill large meshes to disk.

    Detects invalid coordinates/indices, degenerate triangles, open/nonmanifold
    edges, inconsistent winding and nonpositive signed volume. It does not detect
    arbitrary triangle self-intersections.
    """
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if not math.isfinite(precision_tolerance) or precision_tolerance<=0:
        raise ValueError('Допуск координат STL должен быть положительным конечным числом в мм.')
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError('В модели нет вершин XYZ.')
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
        raise ValueError('Для замкнутой STL-модели нужны треугольные грани.')
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError('Индексы треугольников должны быть целыми числами.')
    if len(faces) > 0xffffffff:
        raise ValueError('Число треугольников превышает предел двоичного STL.')
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for start in range(0, len(vertices), chunk_size):
        block = vertices[start:start+chunk_size]
        if not np.isfinite(block).all() or np.max(np.abs(block)) > np.finfo(np.float32).max:
            raise ValueError('Координаты содержат NaN, бесконечность или выходят за диапазон STL.')
        error=float(np.max(np.abs(block.astype(np.float64)-block.astype(np.float32).astype(np.float64))))
        if error > precision_tolerance:
            raise ValueError(f'Преобразование в float32 изменяет координаты на {error:.6g} мм, больше допуска {precision_tolerance:g} мм. Переместите модель ближе к началу координат или выберите обоснованный допуск.')
        low, high = np.minimum(low, block.min(axis=0)), np.maximum(high, block.max(axis=0))
    if np.any(high <= low):
        raise ValueError('Модель должна иметь положительные размеры по X, Y и Z.')
    reference = low/2+high/2
    volume_parts = []
    use_ram = len(faces) <= _STL_RAM_FACE_LIMIT and len(vertices)**2 <= np.iinfo(np.int64).max
    with ExitStack() as resources:
        if use_ram:
            all_edges = np.empty((len(faces)*3, 2), dtype=np.int64)
        else:
            folder = resources.enter_context(tempfile.TemporaryDirectory(prefix='stone_stl_check_'))
            database = sqlite3.connect(str(FilePath(folder)/'edges.sqlite'))
            resources.callback(database.close)
            # This is a disposable edge index, never the user's model or project.
            database.execute('PRAGMA journal_mode=OFF')
            database.execute('PRAGMA synchronous=OFF')
            database.execute('PRAGMA cache_size=-8192')
            database.execute('PRAGMA temp_store=FILE')
            database.execute('CREATE TABLE edges(a INTEGER, b INTEGER, n INTEGER, d INTEGER, PRIMARY KEY(a,b)) WITHOUT ROWID')
        for start in range(0, len(faces), chunk_size):
            block = faces[start:start+chunk_size]
            if block.min() < 0 or block.max() >= len(vertices):
                raise ValueError('Треугольник ссылается на несуществующую вершину.')
            # Validate the coordinates actually written to the float32 STL.
            triangle = np.asarray(vertices[block], dtype=np.float32).astype(np.float64)
            normal = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
            if np.any(np.linalg.norm(normal, axis=1) == 0):
                raise ValueError('Обнаружены вырожденные треугольники. Исправьте сетку перед экспортом.')
            local = triangle-reference
            volume_parts.append(float(np.einsum('ij,ij->i', local[:, 0], np.cross(local[:, 1], local[:, 2])).sum()/6))
            edges = np.concatenate((block[:, (0, 1)], block[:, (1, 2)], block[:, (2, 0)]))
            if use_ram:
                all_edges[3*start:3*(start+len(block))] = edges
            else:
                # Merge repeated edges in a batch before inserting into SQLite.
                pairs, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0,
                                                   return_inverse=True, return_counts=True)
                directions = np.bincount(inverse, weights=np.where(edges[:,0] < edges[:,1], 1, -1)).astype(np.int64)
                database.executemany(
                    'INSERT INTO edges VALUES(?,?,?,?) ON CONFLICT(a,b) DO UPDATE SET n=n+excluded.n,d=d+excluded.d',
                    ((int(a), int(b), int(n), int(d)) for (a,b),n,d in zip(pairs,counts,directions)))
            _export_progress(progress, 'check', min(start+chunk_size, len(faces)), len(faces))
        if use_ram:
            directions = np.where(all_edges[:,0] < all_edges[:,1], 1, -1)
            keys = np.minimum(all_edges[:,0],all_edges[:,1])*len(vertices)+np.maximum(all_edges[:,0],all_edges[:,1])
            del all_edges
            _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
            bad_edges = int(np.count_nonzero(counts != 2))
            bad_directions = int(np.count_nonzero(np.bincount(inverse, weights=directions)))
        else:
            bad_edges = database.execute('SELECT COUNT(*) FROM edges WHERE n!=2').fetchone()[0]
            bad_directions = database.execute('SELECT COUNT(*) FROM edges WHERE d!=0').fetchone()[0]
        _export_progress(progress, 'check', len(faces), len(faces))
        if bad_edges:
            raise ValueError(f'Сетка незамкнута или содержит неманифолдные рёбра: {bad_edges}. Экспорт отменён. Исправьте сетку.')
        if bad_directions:
            raise ValueError(f'Нарушена ориентация граней у {bad_directions} рёбер. Исправьте нормали перед экспортом.')
    volume = math.fsum(volume_parts)
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError('Объём модели нулевой или нормали направлены внутрь. Исправьте сетку перед экспортом.')
    return ExportReport(len(faces), tuple(float(x) for x in high-low), volume)


def _stl_records(triangles):
    triangles = np.asarray(triangles, dtype=np.float32)
    wide = triangles.astype(np.float64)
    normals = np.cross(wide[:, 1]-wide[:, 0], wide[:, 2]-wide[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normals).all() or np.any(lengths == 0):
        raise ValueError('Невозможно записать вырожденный треугольник.')
    normals /= lengths[:, None]
    records = np.zeros(len(triangles), dtype=_STL_RECORD)
    records['normal'], records['vertices'] = normals, triangles
    return records


def _atomic_stl(path, report, batches, progress=None):
    path = FilePath(path)
    expected_size = 84+50*report.triangles
    if shutil.disk_usage(path.parent).free < expected_size+4*1024**2:
        raise OSError('Недостаточно свободного места для безопасной записи STL. Старый файл сохранён.')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.'+path.name+'.', suffix='.tmp', dir=path.parent, delete=False) as stream:
            temporary = FilePath(stream.name)
            stream.write(b'Stone Relief Studio 3.3.1; millimetres'.ljust(80, b' '))
            stream.write(struct.pack('<I', report.triangles))
            written = 0
            for triangles in batches:
                stream.write(_stl_records(triangles).tobytes())
                written += len(triangles)
                _export_progress(progress, 'write', written, report.triangles)
            if written != report.triangles or stream.tell() != expected_size:
                raise ValueError('Число записанных треугольников не совпало с расчётом. Старый файл сохранён.')
            stream.flush()
            os.fsync(stream.fileno())
        # A cancellation or write error above leaves an existing destination intact.
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report


def export_indexed_stl(path, vertices, faces, progress=None, chunk_size=8192, precision_tolerance=.001):
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    report = validate_stl_mesh(vertices, faces, progress, chunk_size, precision_tolerance)
    batches = (vertices[faces[start:start+chunk_size]] for start in range(0, len(faces), chunk_size))
    return _atomic_stl(path, report, batches, progress)


def _heightfield_batches(z, xs, ys, chunk_size=8192):
    ny, nx = z.shape
    for row in range(ny-1):
        for start in range(0, nx-1, chunk_size):
            end = min(start+chunk_size, nx-1)
            count = end-start
            a = np.column_stack((xs[start:end], np.full(count, ys[row]), z[row, start:end]))
            b = np.column_stack((xs[start+1:end+1], np.full(count, ys[row]), z[row, start+1:end+1]))
            c = np.column_stack((xs[start:end], np.full(count, ys[row+1]), z[row+1, start:end]))
            d = np.column_stack((xs[start+1:end+1], np.full(count, ys[row+1]), z[row+1, start+1:end+1]))
            yield np.concatenate((np.stack((a,b,d), axis=1), np.stack((a,d,c), axis=1)))
    # The perimeter is O(nx+ny); there is no full vertex or face array.
    ix = np.concatenate((np.arange(nx-1), np.full(ny-1, nx-1), np.arange(nx-1, 0, -1), np.zeros(ny-1, int)))
    iy = np.concatenate((np.zeros(nx-1, int), np.arange(ny-1), np.full(nx-1, ny-1), np.arange(ny-1, 0, -1)))
    perimeter = np.column_stack((xs[ix], ys[iy], z[iy, ix]))
    for start in range(0, len(perimeter), chunk_size):
        indices = np.arange(start, min(start+chunk_size, len(perimeter)))
        a, b = perimeter[indices], perimeter[(indices+1) % len(perimeter)]
        bottom_a, bottom_b = a.copy(), b.copy()
        bottom_a[:, 2] = bottom_b[:, 2] = 0
        centre = np.zeros_like(a)
        yield np.concatenate((np.stack((a,bottom_a,bottom_b), axis=1),
                              np.stack((a,bottom_b,b), axis=1),
                              np.stack((centre,bottom_b,bottom_a), axis=1)))


def export_heightmap_stl(path, z, width, height, progress=None, minimum_thickness=.05, precision_tolerance=.001):
    z = np.asarray(z)
    if z.ndim != 2 or min(z.shape) < 2:
        raise ValueError('Рельеф должен содержать минимум 2×2 точки.')
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError('Ширина и высота модели должны быть положительными числами в миллиметрах.')
    if not math.isfinite(minimum_thickness) or minimum_thickness<.05 or not math.isfinite(precision_tolerance) or precision_tolerance<=0:
        raise ValueError('Проверьте минимальную толщину (от 0,05 мм) и положительный допуск STL.')
    ny, nx = z.shape
    count = 2*(nx-1)*(ny-1)+3*(2*nx+2*ny-4)
    if count > 0xffffffff:
        raise ValueError('Слишком много треугольников для двоичного STL.')
    maximum = 0.0
    for start in range(0, ny, 64):
        block = z[start:start+64]
        if not np.isfinite(block).all() or np.max(np.abs(block)) > np.finfo(np.float32).max:
            raise ValueError('Карта высот содержит NaN, бесконечность или слишком большие значения.')
        if np.any(block <= 0) or np.any(np.asarray(block, dtype=np.float32) <= 0):
            raise ValueError('Поверхность касается или пересекает дно Z=0. Увеличьте толщину основания перед экспортом.')
        if np.min(np.asarray(block,dtype=np.float32)) < minimum_thickness-1e-7:
            raise ValueError(f'Толщина основания меньше {minimum_thickness:g} мм. Поднимите поверхность или увеличьте основание; ослабляйте предел только осознанно.')
        error=float(np.max(np.abs(block.astype(np.float64)-block.astype(np.float32).astype(np.float64))))
        if error>precision_tolerance:
            raise ValueError(f'Потеря точности Z при записи STL: {error:g} мм. Уменьшите высоту или измените допуск.')
        maximum = max(maximum, float(block.max()))
        _export_progress(progress, 'check', min(start+64, ny), ny)
    xs = np.linspace(-width/2, width/2, nx, dtype=np.float32)
    ys = np.linspace(-height/2, height/2, ny, dtype=np.float32)
    error=max(float(np.max(np.abs(xs.astype(float)-np.linspace(-width/2,width/2,nx)))),
              float(np.max(np.abs(ys.astype(float)-np.linspace(-height/2,height/2,ny)))))
    if error>precision_tolerance:
        raise ValueError(f'Потеря точности X/Y: {error:g} мм. Уменьшите размеры модели или измените допуск STL.')
    if np.any(np.diff(xs) <= 0) or np.any(np.diff(ys) <= 0) or not np.isfinite(xs).all() or not np.isfinite(ys).all():
        raise ValueError('Шаг координат нельзя представить в STL без вырождения треугольников.')
    # A positive single-valued heightfield with strictly increasing X/Y has a
    # closed oriented top, sides and flat bottom by construction.
    integral = 0.0
    for start in range(0, ny-1, 64):
        end = min(start+64, ny-1)
        integral += (2*np.sum(z[start:end, :-1], dtype=np.float64)+np.sum(z[start:end, 1:], dtype=np.float64)
                     +np.sum(z[start+1:end+1, :-1], dtype=np.float64)+2*np.sum(z[start+1:end+1, 1:], dtype=np.float64))
    volume = integral*(width/(nx-1))*(height/(ny-1))/6
    report = ExportReport(count, (float(width), float(height), maximum), float(volume))
    return _atomic_stl(path, report, _heightfield_batches(z, xs, ys), progress)
