"""Bound untrusted raster inputs before decompression or pixel allocation."""
import io
import os
import warnings
from PIL import Image

MAX_IMAGE_PIXELS = 24_000_000
MAX_IMAGE_SIDE = 16384
MAX_IMAGE_BYTES = 64*1024**2
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def read_checked_file(path, limit=MAX_IMAGE_BYTES):
    if os.path.getsize(path) > limit:
        raise ValueError(f'Файл превышает предел {limit//1024**2} МиБ. Уменьшите размер исходного файла.')
    with open(path, 'rb') as stream:
        data = stream.read(limit+1)
    if len(data) > limit:
        raise ValueError('Размер файла изменился во время чтения. Повторите открытие.')
    return data


def open_checked_image(raw):
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError('Изображение больше 64 МиБ. Сохраните уменьшенную копию.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(raw))
        if max(image.size) > MAX_IMAGE_SIDE or image.width*image.height > MAX_IMAGE_PIXELS:
            image.close()
            raise ValueError('Изображение превышает 24 млн пикселей или сторону 16384 px. Уменьшите разрешение.')
        return image
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError('Слишком большое изображение. Уменьшите разрешение до 24 млн пикселей.') from error
