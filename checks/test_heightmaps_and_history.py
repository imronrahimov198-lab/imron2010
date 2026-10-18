import io
import struct

import numpy as np
import pytest
from PIL import Image

from stone_relief_studio_fixed import photo_heights, history_storage_bytes, trim_history_stacks


def integer_tiff(array, bits, sample_format):
    """Minimal uncompressed grayscale TIFF preserving signedness and bit depth."""
    dtype = '<'+('u' if sample_format == 1 else 'i')+str(bits//8)
    pixels = np.asarray(array, dtype=dtype)
    tags = [(256,4,pixels.shape[1]),(257,4,pixels.shape[0]),(258,3,bits),
            (259,3,1),(262,3,1),(273,4,134),(277,3,1),
            (278,4,pixels.shape[0]),(279,4,pixels.nbytes),(339,3,sample_format)]
    return (b'II'+struct.pack('<HIH',42,8,len(tags))
            +b''.join(struct.pack('<HHII',tag,kind,1,value) for tag,kind,value in tags)
            +struct.pack('<I',0)+pixels.tobytes())


@pytest.mark.parametrize('bits,sample_format',[(16,1),(16,2),(32,1),(32,2)])
def test_integer_heightmaps_use_declared_range(bits, sample_format):
    minimum = 0 if sample_format==1 else -2**(bits-1)
    maximum = 2**bits-1 if sample_format==1 else 2**(bits-1)-1
    middle = (minimum+maximum)//2
    raw = integer_tiff([[minimum,middle],[middle+1,maximum]],bits,sample_format)
    heights = np.flipud(photo_heights(raw,2,2))
    assert heights[0,0]==0
    assert heights[1,1]==1
    assert heights[0,1]==pytest.approx(.5,abs=1/65535)
    if bits==16:
        assert heights[1,0]-heights[0,1]==pytest.approx(1/65535,abs=1e-7)


def test_float_heightmap_normalizes_observed_range():
    buffer=io.BytesIO()
    Image.fromarray(np.array([[-10,0],[10,30]],np.float32)).save(buffer,format='TIFF')
    result=np.flipud(photo_heights(buffer.getvalue(),2,2))
    assert result==pytest.approx(np.array([[0,.25],[.5,1]]))


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_nonfinite_float_heightmaps_are_rejected(bad):
    buffer=io.BytesIO()
    Image.fromarray(np.array([[0,bad],[.5,1]],np.float32)).save(buffer,format='TIFF')
    with pytest.raises(ValueError,match='NaN|бесконечность'):
        photo_heights(buffer.getvalue(),2,2)


def test_16bit_png_preserves_declared_unsigned_range():
    buffer=io.BytesIO()
    Image.fromarray(np.array([[0,16384],[32768,65535]],np.uint16)).save(buffer,format='PNG')
    result=np.flipud(photo_heights(buffer.getvalue(),2,2))
    assert result==pytest.approx(np.array([[0,16384/65535],[32768/65535,1]]),abs=1e-7)


def test_history_counts_shared_views_once():
    data=np.arange(20,dtype=np.float32)
    undo=[('first',{'z':data}),('second',{'z':data[::-1]})]
    redo=[('third',{'z':data.reshape(4,5)})]
    assert history_storage_bytes(undo,redo)==data.nbytes


def test_history_drops_oldest_distinct_states_to_fit_budget():
    arrays=[np.full(10,index,dtype=np.float32) for index in range(4)]
    undo=[(str(index),{'z':a}) for index,a in enumerate(arrays)]
    assert trim_history_stacks(undo,[],80)==2
    assert [label for label,_ in undo]==['2','3']
    assert history_storage_bytes(undo)==80


def test_history_keeps_one_undo_when_a_single_state_exceeds_budget():
    undo=[('last',{'z':np.ones(100,np.float32)})]
    assert trim_history_stacks(undo,[],10)==0
    assert len(undo)==1


def test_undo_redo_shared_buffers_do_not_trigger_false_eviction():
    data=np.ones(100,np.float32)
    undo=[('first',{'z':data})]; redo=[('next',{'z':data[::-1]})]
    assert trim_history_stacks(undo,redo,data.nbytes)==0
    assert len(undo)==len(redo)==1
