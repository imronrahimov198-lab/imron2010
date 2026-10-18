"""Original geometric icons drawn for Stone Relief Studio."""
from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

PATHS = {
    'book': '<path d="M3 6q7-3 13 1 6-4 13-1v20q-7-3-13 1-6-4-13-1z M16 7v20 M6 11q4-1 7 1 M6 16q4-1 7 1 M19 12q3-2 7-1 M19 17q3-2 7-1"/>',
    'new': '<path d="M8 4h12l5 5v19H8z M20 4v6h5 M12 18h9 M16.5 13.5v9"/>',
    'open': '<path d="M4 10h10l3 4h12l-4 13H5z M5 10V7h10l3 3"/>',
    'save': '<path d="M6 5h18l3 3v19H6z M11 5v8h11V5 M11 27v-9h11v9"/>',
    'undo': '<path d="M12 7l-7 7 7 6 M6 14h13q8 0 8 7v4"/>',
    'redo': '<path d="M20 7l7 7-7 6 M26 14H13q-8 0-8 7v4"/>',
    'select': '<path d="M8 4l19 15-10 1-4 9z"/>',
    'nodes': '<path d="M6 24C6 6 26 27 26 8 M6 24l5-17 M26 8l-4 18"/><rect x="3" y="21" width="6" height="6"/><rect x="23" y="5" width="6" height="6"/>',
    'pen': '<path d="M7 25l2-8L23 3l6 6-14 14z M9 17l6 6 M21 5l6 6"/>',
    'rectangle': '<rect x="4" y="8" width="24" height="17" rx="1"/>',
    'square': '<rect x="6" y="6" width="20" height="20" rx="1"/>',
    'ellipse': '<ellipse cx="16" cy="16" rx="13" ry="8"/>',
    'circle': '<circle cx="16" cy="16" r="11"/>',
    'polygon': '<path d="M16 3l12 9-5 15H9L4 12z"/>',
    'star': '<path d="M16 3l4 8 9 2-7 7 1 9-7-4-8 4 2-9-7-7 9-2z"/>',
    'text': '<path d="M5 8V5h22v3 M16 5v22 M11 27h10"/>',
    'point': '<circle cx="16" cy="16" r="5"/><path d="M16 2v6 M16 24v6 M2 16h6 M24 16h6"/>',
    'shape': '<path d="M3 25h26 M5 23C7 4 25 4 27 23 M11 23C12 12 20 12 21 23"/>',
    'rails': '<path d="M4 26Q5 10 13 4 M17 28Q30 18 26 5 M5 23Q17 11 27 19 M7 15Q18 5 27 12"/>',
    'dome': '<path d="M4 23q12-28 24 0 M4 23q12 9 24 0 M16 6v21"/>',
    'plate': '<path d="M3 14l17-8 10 9-17 9z M3 14v5l10 9 17-8v-5 M13 24v4"/>',
    'sculpt': '<path d="M3 26q7-19 13-12t13 5 M17 6l4-3 6 8-5 3z M17 6l-5 10 10-2"/>',
    'smooth': '<path d="M3 23l5-10 5 7 6-14 10 12 M3 27q11-13 26-4"/>',
    'zero': '<path d="M3 24h26 M8 8l16 11 M24 8L8 19"/>',
    'stl': '<path d="M16 3L3 11v14l13 5 13-9V9z M3 11l13 7 13-9 M16 18v12 M16 3v15"/>',
    'photo': '<rect x="3" y="5" width="26" height="22" rx="2"/><circle cx="10" cy="12" r="3"/><path d="M4 25l8-8 5 4 6-9 6 13"/>',
    'fit': '<path d="M11 4H4v7 M21 4h7v7 M4 21v7h7 M28 21v7h-7"/><rect x="9" y="10" width="14" height="12"/>',
    'copy': '<rect x="5" y="4" width="16" height="19"/><path d="M12 23v5h15V10h-6"/>',
    'delete': '<path d="M5 8h22 M12 8V4h8v4 M8 8l2 20h12l2-20 M13 12v11 M19 12v11"/>',
    'transform': '<path d="M4 13V4h9 M4 4l9 9 M19 28h9v-9 M28 28l-9-9"/><rect x="11" y="11" width="10" height="10"/>',
}


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
