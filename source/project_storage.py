"""Project validation and atomic, lossless storage, independent of Qt."""
import io
import json
import math
import os
import tempfile
import zipfile

import numpy as np

FORMAT, VERSION, EXT = 'StoneReliefStudioPro', 5, '.srelief'
MAX_PROJECT_BYTES = 384*1024**2
MAX_ARRAY_BYTES = 256*1024**2
MAX_METADATA_BYTES = 8*1024**2


class ProjectCancelled(Exception):
    pass


def _project_number(value, label, minimum=None, maximum=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Некорректное число в проекте: {label}.')
    if ((minimum is not None and value < minimum) or
            (maximum is not None and value > maximum) or (integer and type(value) is not int)):
        raise ValueError(f'Значение вне допустимого диапазона: {label}.')


def _project_point(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f'Некорректные координаты: {label}.')
    for coordinate in value:
        _project_number(coordinate, label)


def validate_project_payload(meta, arrays):
    """Reject broken payloads before a GUI replaces the current document."""
    if not isinstance(meta, dict) or not isinstance(arrays, dict):
        raise ValueError('Повреждена структура проекта.')
    settings = meta.get('settings', {})
    if not isinstance(settings, dict):
        raise ValueError('Повреждены настройки проекта.')
    export=settings.get('export_options',{})
    if not isinstance(export,dict): raise ValueError('Повреждены параметры экспорта STL.')
    for key,low,high in [('minimum_thickness',.05,200),('precision',.000001,1)]:
        if key in export: _project_number(export[key],key,low,high)
    ranges = dict(w=(1e-6, 1e9), h=(1e-6, 1e9), base=(.5, 200), depth=(.5, 200),
                  step=(.05, 2), grid=(.5, 500), blur=(0, 15), photo_fade=(0, 100),
                  bright=(0, 100), contrast=(0, 100), az=(-180, 180), el=(3, 90),
                  level=(0, 500), material=(0, 3))
    integers = {'photo_fade', 'bright', 'contrast', 'az', 'el', 'material'}
    for key, bounds in ranges.items():
        if key in settings:
            _project_number(settings[key], key, *bounds, integer=key in integers)
    for key in ('invert', 'sun', 'fill'):
        if key in settings and not isinstance(settings[key], bool):
            raise ValueError(f'Некорректный переключатель: {key}.')
    vectors = meta.get('vec', [])
    if not isinstance(vectors, list):
        raise ValueError('Повреждён список векторов.')
    uids = set()
    for vector in vectors:
        if not isinstance(vector, dict) or not isinstance(vector.get('n'), list):
            raise ValueError('Повреждён вектор.')
        for node in vector['n']:
            if not isinstance(node, (list, tuple)) or len(node) != 4:
                raise ValueError('Повреждён узел вектора.')
            _project_point(node[:2], 2, 'узел')
            for handle in node[2:]:
                if handle is not None:
                    _project_point(handle, 2, 'ручка Безье')
        _project_number(vector.get('g', 0), 'группа векторов', integer=True)
        if 'c' in vector and not isinstance(vector['c'], bool):
            raise ValueError('Повреждён признак замыкания вектора.')
        uid = vector.get('uid')
        if uid is not None:
            if not isinstance(uid, str) or not uid or uid in uids:
                raise ValueError('Некорректный или повторный идентификатор вектора.')
            uids.add(uid)
    controls = meta.get('ctrl', [])
    if not isinstance(controls, list):
        raise ValueError('Повреждён список управляющих точек.')
    for point in controls:
        _project_point(point, 3, 'управляющая точка')
    workspace = settings.get('workspace', {})
    if not isinstance(workspace, dict):
        raise ValueError('Повреждены параметры рабочего пространства.')
    for name in ('sweep', 'form'):
        if not isinstance(workspace.get(name, {}), dict):
            raise ValueError('Повреждены параметры инструмента: ' + name)
    sweep, form = workspace.get('sweep', {}), workspace.get('form', {})
    rails, sections = sweep.get('rails', [None, None]), sweep.get('sections', [])
    if (not isinstance(rails, list) or len(rails) != 2 or not isinstance(sections, list)
            or any(x is not None and not isinstance(x, str) for x in rails)
            or any(not isinstance(x, str) for x in sections)
            or (sweep.get('z') is not None and not isinstance(sweep['z'], str))):
        raise ValueError('Повреждены ссылки на направляющие или сечения.')
    positions = sweep.get('positions', {})
    if not isinstance(positions, dict) or any(not isinstance(x, str) for x in positions):
        raise ValueError('Повреждены положения сечений.')
    for value in positions.values():
        _project_number(value, 'положение сечения', 0, 100)
    for key in ('flip1', 'flip2', 'use_z', 'transition', 'scale_w', 'scale_end'):
        if key in sweep and not isinstance(sweep[key], bool):
            raise ValueError('Повреждён переключатель инструмента: ' + key)
    for key in ('start', 'end'):
        if key in sweep:
            _project_number(sweep[key], key, -500, 500)
    if sweep.get('mode', 'add') not in ('add', 'sub', 'high', 'low'):
        raise ValueError('Неизвестный режим объединения сечений.')
    form_ranges = dict(angle=(-90,90), scale=(1,50), kind=(0,2), start=(-500,500),
                       limit=(0,500), height=(0,500), exact=(0,500), const=(-500,500))
    for key, bounds in form_ranges.items():
        if key in form:
            _project_number(form[key], key, *bounds, integer=key in ('angle', 'scale', 'kind'))
    if form.get('kind', 0) not in (0, 1, 2):
        raise ValueError('Неизвестная форма F12.')
    if form.get('height_mode', 'free') not in ('free', 'limit', 'scale', 'const', 'exact'):
        raise ValueError('Неизвестный режим высоты F12.')
    if form.get('combine', 'add') not in ('add', 'sub', 'high', 'low', 'zero_in', 'zero_out', 'replace'):
        raise ValueError('Неизвестный режим объединения F12.')
    if set(arrays) - {'z', 'mesh_vertices', 'mesh_faces'}:
        raise ValueError('Неизвестные данные геометрии проекта.')
    for key, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
            raise ValueError('Недопустимые значения геометрии: ' + key)
    if 'z' in arrays:
        z = arrays['z']
        if (z.ndim != 2 or min(z.shape) < 2
                or max(abs(float(z.min())), abs(float(z.max()))) > float(np.finfo(np.float32).max)):
            raise ValueError('Некорректная карта высот проекта или превышен диапазон высот.')
    has_vertices, has_faces = 'mesh_vertices' in arrays, 'mesh_faces' in arrays
    if has_vertices != has_faces or ('z' in arrays and has_vertices):
        raise ValueError('Несогласованные данные сетки проекта.')
    if has_vertices:
        vertices, faces = arrays['mesh_vertices'], arrays['mesh_faces']
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or faces.dtype.kind not in 'iu' or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError('Повреждены вершины или индексы сетки проекта.')
    interactive=settings.get('interactive',{})
    if not isinstance(interactive,dict):
        raise ValueError('Повреждены параметры Interactive Relief Manipulation.')
    from interactive_geometry import validate_control_points
    validate_control_points(interactive.get('controls',[]))
    bindings=interactive.get('mask_vectors',[])
    if not isinstance(bindings,list) or any(not isinstance(uid,str) for uid in bindings):
        raise ValueError('Повреждены ссылки на контуры области деформации.')
    for key in ('pins','selected'):
        indices=interactive.get(key,[])
        if not isinstance(indices,list) or len(indices)>100000:
            raise ValueError('Некорректный список вершин Mesh Edit.')
        if any(type(i) is not int or i<0 or not has_vertices or i>=len(arrays['mesh_vertices']) for i in indices):
            raise ValueError('Mesh Edit ссылается на отсутствующие вершины.')
    options=interactive.get('options',{})
    if not isinstance(options,dict): raise ValueError('Повреждены параметры деформации.')
    for key,low,high in [('radius',.01,10000),('strength',0,2),('gain',.001,10),('feather',.01,1000)]:
        if key in options: _project_number(options[key],key,low,high)
    for key in ('fixed','use_mask','soft'):
        if key in options and type(options[key]) is not bool: raise ValueError('Некорректный переключатель деформации.')
    for key in ('selection_kind','drag_axis'):
        if key in options: _project_number(options[key],key,0,2,integer=True)
    if options.get('target','relief') not in ('relief','mesh') or options.get('surface_mode','local') not in ('local','full'):
        raise ValueError('Неизвестный режим Interactive Relief Manipulation.')
    if 'mesh_delta' in options:
        _project_point(options['mesh_delta'],3,'смещение сетки')
        if any(abs(x)>100000 for x in options['mesh_delta']): raise ValueError('Слишком большое смещение сетки.')


def pack_project(path, meta, photo=None, photo_name=None, arrays=None, progress=None):
    arrays = {key: np.asarray(value) for key, value in (arrays or {}).items()}
    validate_project_payload(meta, arrays)
    if sum(v.nbytes for v in arrays.values()) > MAX_ARRAY_BYTES:
        raise ValueError('Геометрия проекта больше 256 МиБ. Уменьшите разрешение или размер сетки.')
    meta = dict(meta, format=FORMAT, version=VERSION, photo_file='photo.bin' if photo else None,
                arrays=sorted(arrays), photo_name=os.path.basename(photo_name or 'photo.png'))
    metadata = json.dumps(meta, ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(metadata) > MAX_METADATA_BYTES:
        raise ValueError('Метаданные проекта больше 8 МиБ. Разделите векторный рисунок на проекты.')
    total = sum(v.nbytes for v in arrays.values())+len(photo or b'')+len(metadata)
    done = 0
    def write_chunks(stream, data):
        nonlocal done
        view = memoryview(data).cast('B')
        for start in range(0, len(view), 1024**2):
            if progress: progress(done, total)
            part = view[start:start+1024**2]
            stream.write(part); done += len(part)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.' + os.path.basename(path) + '.', suffix='.tmp',
                                         dir=os.path.dirname(os.path.abspath(path)), delete=False) as stream:
            temporary = stream.name
            with zipfile.ZipFile(stream, 'w', zipfile.ZIP_STORED) as archive:
                archive.writestr('project.json', metadata)
                if photo:
                    with archive.open(meta['photo_file'], 'w', force_zip64=True) as target:
                        write_chunks(target, photo)
                if arrays:
                    # A nested NPZ is streamed: no whole compressed NPZ in BytesIO.
                    with archive.open('arrays.npz', 'w', force_zip64=True) as target:
                        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as payload:
                            for key, array in arrays.items():
                                array = np.ascontiguousarray(array)
                                with payload.open(key+'.npy', 'w', force_zip64=True) as member:
                                    np.lib.format.write_array_header_2_0(member, dict(
                                        descr=np.lib.format.dtype_to_descr(array.dtype),
                                        fortran_order=False, shape=array.shape))
                                    write_chunks(member, array)
            if progress: progress(total, total)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path); temporary = None
    finally:
        if temporary is not None: os.unlink(temporary)


def _project_zip_budget(archive, limit, members=32):
    infos = archive.infolist()
    if len(infos) > members or len({x.filename for x in infos}) != len(infos):
        raise ValueError('В архиве проекта слишком много или повторяются записи. Файл не открыт.')
    if sum(x.file_size for x in infos) > limit or any(x.flag_bits & 1 for x in infos):
        raise ValueError('Превышен безопасный объём распаковки проекта или архив зашифрован.')


def unpack_project(path):
    if os.path.getsize(path) > MAX_PROJECT_BYTES:
        raise ValueError('Файл проекта больше 384 МиБ. Используйте проект меньшего размера.')
    with zipfile.ZipFile(path) as archive:
        _project_zip_budget(archive, MAX_PROJECT_BYTES)
        if archive.getinfo('project.json').file_size > MAX_METADATA_BYTES:
            raise ValueError('Метаданные проекта больше 8 МиБ. Открытие отменено.')
        meta = json.loads(archive.read('project.json').decode('utf-8'))
        if not isinstance(meta, dict) or meta.get('format') != FORMAT:
            raise ValueError('Неизвестный формат проекта.')
        version = meta.get('version')
        if type(version) is not int or version < 2 or version > VERSION:
            raise ValueError('Неподдерживаемая версия проекта. Обновите программу.')
        photo = None
        if meta.get('photo_file'):
            from input_limits import MAX_IMAGE_BYTES
            if archive.getinfo(meta['photo_file']).file_size > MAX_IMAGE_BYTES:
                raise ValueError('Фотография в проекте больше 64 МиБ. Открытие отменено.')
            photo = archive.read(meta['photo_file'])
        arrays = {}
        if meta.get('arrays'):
            if archive.getinfo('arrays.npz').file_size > MAX_ARRAY_BYTES:
                raise ValueError('Архив геометрии больше 256 МиБ.')
            with zipfile.ZipFile(io.BytesIO(archive.read('arrays.npz'))) as payload:
                _project_zip_budget(payload, MAX_ARRAY_BYTES, members=8)
                descriptors = []
                for info in payload.infolist():
                    if info.filename not in ('z.npy','mesh_vertices.npy','mesh_faces.npy'):
                        raise ValueError('Неизвестный массив геометрии проекта.')
                    with payload.open(info) as member:
                        magic = np.lib.format.read_magic(member)
                        reader = np.lib.format.read_array_header_1_0 if magic == (1,0) else np.lib.format.read_array_header_2_0
                        if magic not in ((1,0),(2,0)):
                            raise ValueError('Неподдерживаемый заголовок массива проекта.')
                        shape, fortran, dtype = reader(member, max_header_size=65536)
                        size = math.prod(shape)*dtype.itemsize
                        if (len(shape)>2 or dtype.kind not in 'fiu' or dtype.hasobject
                                or size > MAX_ARRAY_BYTES or size != info.file_size-member.tell()):
                            raise ValueError('Размер или тип массива не соответствует данным. Открытие отменено до выделения памяти.')
                        descriptors.append((info, size))
                if sum(size for _,size in descriptors) > MAX_ARRAY_BYTES:
                    raise ValueError('Геометрия проекта требует больше 256 МиБ памяти.')
                for info, _ in descriptors:
                    with payload.open(info) as member:
                        arrays[info.filename[:-4]] = np.load(member, allow_pickle=False, max_header_size=65536)
            if sorted(arrays) != sorted(meta['arrays']):
                raise ValueError('Список геометрии проекта не совпадает с данными.')
    validate_project_payload(meta, arrays)
    return meta, photo, arrays
