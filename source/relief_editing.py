"""Millimetre-space region editing and workspace sampling for the existing editor."""
import math
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter


def reset_relief_region(z, coverage, base, outside=False):
    """Reset sampled cell centres to the base, leaving all other bytes unchanged."""
    mask = np.asarray(coverage) > .5
    if outside:
        mask = ~mask
    return np.where(mask, float(base), np.asarray(z, dtype=np.float64)), mask


def slope_delta(coverage, cell_x, cell_y, amount, direction=90., start=0.):
    """A plane with a specified height difference in mm across the selected area."""
    coverage = np.asarray(coverage)
    inside = coverage > .5
    if not inside.any():
        raise ValueError('Выделите замкнутый вектор достаточного размера.')
    angle = math.radians(direction)
    ny,nx = coverage.shape
    projection = (np.arange(nx)[None,:]*cell_x*math.cos(angle)
                  + np.arange(ny)[:,None]*cell_y*math.sin(angle))
    low,high = float(projection[inside].min()),float(projection[inside].max())
    if high-low < 1e-10:
        raise ValueError('Область слишком узкая для наклона в этом направлении.')
    plane = np.clip((projection-low)/(high-low),0,1)*float(amount)
    return np.where(coverage > .001, float(start)+plane, 0.)


def relief_details(z, coverage, base, cell_x, cell_y, radius):
    """High-frequency residual; normalized filtering never mixes unselected relief."""
    mask = (np.asarray(coverage) > .001).astype(float)
    sigma = (max(.1,float(radius)/cell_y),max(.1,float(radius)/cell_x))
    weight = gaussian_filter(mask,sigma,mode='constant',cval=0)
    numerator = gaussian_filter((np.asarray(z)-base)*mask,sigma,mode='constant',cval=0)
    low = numerator/np.maximum(weight,1e-12)
    return np.where(mask>0,np.asarray(z)-base-low,0.)


def resample_world_grid(values, source_rect, target_rect, target_shape, fill=0.):
    """Axis-aligned grids anchored at world lower left; no XY scaling of contents."""
    sx,sy,sw,sh = map(float,source_rect)
    tx,ty,tw,th = map(float,target_rect)
    values = np.asarray(values,dtype=np.float64)
    ny,nx = map(int,target_shape)
    if min(sw,sh,tw,th)<=0 or min(values.shape)<2 or min(nx,ny)<2:
        raise ValueError('Для пересчёта нужны положительные размеры и сетка не меньше 2×2.')
    source_dy,source_dx=sh/(values.shape[0]-1),sw/(values.shape[1]-1)
    matrix=np.diag([th/(ny-1)/source_dy,tw/(nx-1)/source_dx])
    offset=((ty-sy)/source_dy,(tx-sx)/source_dx)
    result=affine_transform(values,matrix,offset=offset,output_shape=(ny,nx),order=1,
                            mode='nearest',prefilter=False)
    # Nearest protects endpoints against roundoff. Explicit world masks fill
    # genuinely new workspace, rather than extending the old edge indefinitely.
    xs=np.linspace(tx,tx+tw,nx); ys=np.linspace(ty,ty+th,ny)
    result[:,(xs<sx-1e-9)|(xs>sx+sw+1e-9)]=fill
    result[(ys<sy-1e-9)|(ys>sy+sh+1e-9),:]=fill
    return result
