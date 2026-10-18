"""Deterministic additive relief constraints and lossless mesh vertex editing."""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree


class ManipulationCancelled(Exception):
    pass


def _manipulation_progress(callback, current, total):
    if callback:
        callback(current, total)


def validate_control_points(controls):
    if not isinstance(controls, list) or len(controls) > 128:
        raise ValueError('Допустимо до 128 контрольных точек. Разделите деформацию на несколько операций.')
    identifiers = set()
    for point in controls:
        if not isinstance(point, dict):
            raise ValueError('Повреждена контрольная точка.')
        for key in ('x','y','z'):
            value = point.get(key)
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError('Координаты контрольной точки должны быть конечными числами в мм.')
        if type(point.get('pinned',False)) is not bool:
            raise ValueError('Повреждён признак закрепления точки.')
        uid = point.get('id')
        if not isinstance(uid,str) or not uid or uid in identifiers:
            raise ValueError('Некорректный идентификатор контрольной точки.')
        identifiers.add(uid)


def surface_samples(z, xy, width, height):
    xy = np.asarray(xy,dtype=np.float64).reshape(-1,2)
    indices = np.array([(xy[:,1]+height/2)*(z.shape[0]-1)/height,
                        (xy[:,0]+width/2)*(z.shape[1]-1)/width])
    return map_coordinates(np.asarray(z),indices,order=1,mode='nearest',output=np.float64)


def deform_control_surface(z, controls, width, height, mask=None, fixed=True,
                           mode='local', radius=15., strength=1., feather=3., progress=None):
    """Interpolate displacement constraints, then add to the untouched detailed Z.

    Full uses a smooth Gaussian RBF with an affine tail when constraints span XY.
    Local uses the compact C2 Wendland kernel (exactly zero beyond radius).
    A smooth Dirichlet taper fixes the first boundary ring. Kernel targets are
    compensated for that taper, so interior control points retain their target Z.
    """
    validate_control_points(controls)
    z = np.asarray(z)
    if z.ndim!=2 or min(z.shape)<2 or not np.isfinite(z).all():
        raise ValueError('Нужна корректная карта высот размером не менее 2×2.')
    if mode not in ('local','full') or min(width,height,radius,feather)<=0 or not 0<=strength<=2:
        raise ValueError('Проверьте режим, радиус, плавность и силу деформации.')
    if not controls:
        raise ValueError('Добавьте контрольную точку щелчком по поверхности.')
    mask = np.ones(z.shape,bool) if mask is None else np.asarray(mask,dtype=bool)
    if mask.shape!=z.shape or not mask.any():
        raise ValueError('Выделенная область пуста или не совпадает с рельефом.')
    xy = np.array([[p['x'],p['y']] for p in controls],dtype=np.float64)
    if np.any(np.abs(xy)>[width/2+1e-8,height/2+1e-8]):
        raise ValueError('Контрольная точка за пределами плиты. Переместите её на рельеф.')
    cx, cy = width/(z.shape[1]-1), height/(z.shape[0]-1)
    indices=np.array([(xy[:,1]+height/2)/cy,(xy[:,0]+width/2)/cx])
    if np.any(map_coordinates(mask.astype(float),indices,order=0,mode='nearest')<.5):
        raise ValueError('Точка находится вне выбранной области. Измените маску или удалите точку.')
    base = surface_samples(z,xy,width,height)
    delta = np.array([p['z'] for p in controls])-base
    delta *= np.array([1. if p.get('pinned') else strength for p in controls])
    taper=mask.astype(np.float32)
    if fixed:
        distance=distance_transform_edt(np.pad(mask,1),sampling=(cy,cx))[1:-1,1:-1]
        t=np.clip((distance-max(cx,cy))/max(feather,max(cx,cy)),0,1)
        taper*=t*t*(3-2*t)
    at_control=map_coordinates(taper,indices,order=1,mode='nearest')
    if np.any((at_control<1e-6)&(np.abs(delta)>1e-6)):
        raise ValueError('Точка на закреплённом краю не может менять высоту. Отключите фиксацию или сдвиньте точку внутрь.')
    targets=delta/np.maximum(at_control,1e-6)
    distances=np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=2)
    if np.any((distances+np.eye(len(xy)))<1e-6):
        raise ValueError('Контрольные точки совпадают. Оставьте одну точку в этом месте.')
    scale = radius if mode=='local' else max(width,height,radius)
    def kernel(distance):
        r=distance/scale
        return np.maximum(1-r,0)**4*(4*r+1) if mode=='local' else np.exp(-r*r)
    matrix=kernel(distances)
    polynomial=None
    if mode=='full':
        affine=np.column_stack((np.ones(len(xy)),xy/scale))
        polynomial=affine if np.linalg.matrix_rank(affine)==3 else affine[:,:1]
        n=polynomial.shape[1]
        matrix=np.block([[matrix,polynomial],[polynomial.T,np.zeros((n,n))]])
        rhs=np.r_[targets,np.zeros(n)]
    else:
        rhs=targets
    _manipulation_progress(progress,0,z.size)
    try:
        coefficients=np.linalg.solve(matrix,rhs)
    except np.linalg.LinAlgError as error:
        raise ValueError('Ограничения плохо обусловлены. Раздвиньте близкие точки или уменьшите их число.') from error
    result=np.array(z,dtype=np.float32,copy=True)
    for start in range(0,z.size,8192):
        ids=np.arange(start,min(start+8192,z.size))
        yy,xx=np.divmod(ids,z.shape[1])
        query=np.column_stack((xx*cx-width/2,yy*cy-height/2))
        values=kernel(np.linalg.norm(query[:,None,:]-xy[None,:,:],axis=2))@coefficients[:len(xy)]
        if polynomial is not None:
            p=np.column_stack((np.ones(len(query)),query/scale))[:,:polynomial.shape[1]]
            values+=p@coefficients[len(xy):]
        result.ravel()[ids]+=values*taper.ravel()[ids]
        _manipulation_progress(progress,min(start+8192,z.size),z.size)
    if not np.isfinite(result).all() or np.min(result)<.05:
        raise ValueError('Деформация пересекает дно плиты (Z<0,05 мм). Уменьшите смещение или увеличьте основание.')
    return result


def deform_mesh_vertices(vertices, selected, delta, radius=0., strength=1., pinned=(), progress=None):
    """Move exact indexed vertices in XYZ; never rasterize or alter connectivity."""
    vertices=np.asarray(vertices,dtype=np.float64)
    selected=np.unique(np.asarray(selected,dtype=np.int64))
    pinned=np.unique(np.asarray(pinned,dtype=np.int64))
    delta=np.asarray(delta,dtype=np.float64)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():
        raise ValueError('Некорректная исходная сетка.')
    if not len(selected) or selected.min()<0 or selected.max()>=len(vertices):
        raise ValueError('Выберите вершины, ребро или грань сетки.')
    if len(pinned) and (pinned.min()<0 or pinned.max()>=len(vertices)):
        raise ValueError('Некорректные закреплённые вершины.')
    if delta.shape!=(3,) or not np.isfinite(delta).all() or not math.isfinite(radius) or radius<0 or not 0<=strength<=2:
        raise ValueError('Проверьте смещения X/Y/Z, радиус и силу.')
    selected=np.setdiff1d(selected,pinned)
    if not len(selected):
        raise ValueError('Все выбранные вершины закреплены. Снимите фиксацию.')
    tree=cKDTree(vertices[selected]) if radius>0 else None
    result=vertices.copy()
    for start in range(0,len(vertices),65536):
        end=min(start+65536,len(vertices))
        if tree is not None:
            distance=tree.query(vertices[start:end])[0]/radius
            weight=np.maximum(1-distance,0)**4*(4*distance+1)
            result[start:end]+=weight[:,None]*delta*strength
        _manipulation_progress(progress,end,len(vertices))
    result[selected]=vertices[selected]+delta*strength
    result[pinned]=vertices[pinned]
    if not np.isfinite(result).all():
        raise ValueError('Смещение вышло за диапазон координат. Уменьшите его.')
    return result
