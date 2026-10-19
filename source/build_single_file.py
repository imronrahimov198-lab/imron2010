"""Build the self-contained Python application from its development modules."""
import ast
from pathlib import Path


MODULES = ('input_limits', 'vector_import', 'relief_geometry', 'relief_editing', 'studio_icons', 'stl_export', 'interactive_geometry', 'project_storage', 'studio_core', 'interactive_tools', 'stone_relief_studio')


class FlattenStudio(ast.NodeTransformer):
    def visit_Import(self, node):
        node.names = [name for name in node.names if name.name not in MODULES]
        return node if node.names else None

    def visit_ImportFrom(self, node):
        if node.module in MODULES:
            return None
        if node.module == 'pathlib':
            node.names = [name for name in node.names if name.asname == 'FilePath']
            return node if node.names else None
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'core':
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node


def build(output):
    root = Path(__file__).resolve().parent
    header = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stone Relief Studio — редактор векторов и рельефов, версия 3.4.0 (этап P0).

Вся программа находится в этом файле. Другие файлы исходного кода не нужны.
Запуск: python stone_relief_studio_fixed.py
Установка библиотек:
python -m pip install PySide6 numpy Pillow scipy trimesh pyvista pyvistaqt qtpy ezdxf rtree svgelements shapely "vtk>=9.5,<9.6"

Рабочий этап Pro 3.3: аудит, Interactive Relief Manipulation и Mesh Edit.
3.4.0: F12 редактирует готовый рельеф и фото; обнуление возвращает к основанию.
Подъём/опускание на N мм сохраняют фактуру; наклон, выпуклость/вогнутость,
сохранение деталей при замене, отчёт Z min/max, применение и отмена.
Панель направляющих остаётся видимой. Размеры, сетка и оценка памяти;
изменение рабочей области или масштабирование композиции с Undo/Redo.
Точная высота не умножается на геометрический масштаб; 3D XYZ = 1:1.
Существующие Studio, F12, две направляющие, векторы и «Лепка в 3D» сохранены.
Контрольные точки: Z в мм, закрепление, Local/Full Surface, маска и края,
фоновый предпросмотр, применение одной операцией Undo/Redo.
Mesh Edit открывает исходный STL без карты высот и смещает вершины X/Y/Z.
Проект формата 5 хранит точки и закрепления; открываются форматы 2, 3 и 4.
Новый проект не открывается старыми версиями программы.
Файлы распаковки ограничены 384 МиБ, массивы — 256 МиБ, фото — 64 МиБ
и 24 млн пикселей. Предыдущий файл сохраняется при отмене записи.
Минимальная толщина карты высот и допуск float32 настраиваются в «Модель».
STL интерпретируется в мм; полной проверки самопересечений и толщины
произвольной сетки в этом этапе ещё нет.
Независимые слои, управляющие линии Безье, новые кисти и полный STL Inspector
остаются следующими этапами. Это рабочий этап, а не выполнение всего ТЗ.
Windows 10/11: требуется 64-битный Python 3.12 и OpenGL. Windows не проверен.
Windows 7 не поддерживается текущим PySide6/Qt6 и требует отдельного порта.
"""
'''
    sections = [header]
    for module in MODULES:
        tree = ast.parse((root/(module+'.py')).read_text(encoding='utf-8'))
        if tree.body and isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str):
            tree.body.pop(0)
        tree = ast.fix_missing_locations(FlattenStudio().visit(tree))
        sections.append('\n# ==================== '+module+' ====================\n\n'+ast.unparse(tree)+'\n')
    source = ''.join(sections)
    compile(source,str(output),'exec')
    output.write_text(source,encoding='utf-8')
    return output


if __name__=='__main__':
    import sys
    output=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/workspace/stone_relief_studio_fixed.py')
    print(build(output))
