"""Stone Relief Studio — original classic CAD workspace for vectors and stone reliefs.

Run: python stone_relief_studio.py
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import (QAction, QActionGroup, QColor, QFont, QKeySequence,
                           QPainter, QPainterPath, QPen)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QDialogButtonBox,
                               QDockWidget, QDoubleSpinBox, QFileDialog,
                               QFontComboBox, QFormLayout, QGridLayout,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QMenu, QMessageBox, QPushButton, QScrollArea,
                               QToolBar, QToolButton, QVBoxLayout, QWidget)

import studio_core as core
from studio_icons import icon
from relief_geometry import sculpt_weight
from interactive_tools import InteractiveManipulationMixin


STYLE = '''
QMainWindow, QDialog { background: #e7eced; }
QWidget { color: #293c49; font-family: "Segoe UI", "DejaVu Sans"; font-size: 12px; }
QMenuBar { background: #263c49; color: #f6f4ee; padding: 3px 8px; }
QMenuBar::item { color: #f6f4ee; padding: 6px 12px; background: transparent; }
QMenuBar::item:selected { background: #405968; border-radius: 3px; }
QMenu { background: #fffdf8; border: 1px solid #bcc9cd; padding: 5px; }
QMenu::item { padding: 8px 28px; }
QMenu::item:selected { background: #e2eeec; }
QToolBar { background: #f4f5f1; border: 0; border-bottom: 1px solid #c6d0d1;
           spacing: 3px; padding: 5px; }
QToolBar::separator { background: #ccd4d4; width: 1px; height: 1px; margin: 5px; }
QToolButton { border: 1px solid transparent; border-radius: 4px; padding: 6px; }
QToolButton:hover { background: #e1ebe9; border: 1px solid #bdcfcb; }
QToolButton:checked { background: #d6e8e3; border: 1px solid #799e94; }
QPushButton { background: #f9faf6; border: 1px solid #b9c7ca; border-radius: 4px; padding: 7px 10px; }
QPushButton:hover { background: #e4efeb; border-color: #76998f; }
QPushButton:checked { background: #d7e8e2; border: 1px solid #71998c; }
QPushButton[primary="true"] { background: #427d70; color: white; border-color: #427d70; }
QPushButton[primary="true"]:hover { background: #32695e; }
QPushButton:disabled, QToolButton:disabled { color: #9ca9ad; }
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox { background: #fffefa; border: 1px solid #bbc9cc;
                                            border-radius: 3px; padding: 5px; min-height: 19px; }
QDoubleSpinBox:focus, QLineEdit:focus { border-color: #4e8a7e; }
QTabWidget::pane { border: 1px solid #becacc; background: #f3f5f2; }
QTabBar::tab { background: #dce4e3; padding: 9px 13px; margin-right: 2px; }
QTabBar::tab:selected { background: #fffdf7; border-top: 3px solid #b6794c; padding-top: 6px; }
QDockWidget { font-weight: 600; }
QDockWidget::title { background: #d6e0df; padding: 8px; }
QScrollArea { border: none; }
QListWidget { background: #fffefa; border: 1px solid #c1cdce; border-radius: 3px; }
QStatusBar { background: #263c49; }
QStatusBar QLabel { color: #e7efeb; padding: 3px 8px; }
QStatusBar::item { border: none; }
QToolTip { background: #fffdf3; color: #243b49; border: 1px solid #aebfbe; padding: 5px; }
QSlider::groove:horizontal { background: #c5d3d0; height: 5px; border-radius: 2px; }
QSlider::handle:horizontal { background: #427d70; width: 13px; margin: -5px 0; border-radius: 5px; }
QCheckBox { spacing: 6px; padding: 3px 0; }
'''


class Ruler(QWidget):
    def __init__(self, studio, vertical=False):
        super().__init__()
        self.studio, self.vertical = studio, vertical
        self.setFixedWidth(32) if vertical else self.setFixedHeight(26)

    def paintEvent(self, event):
        v = self.studio.view
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#edf1ee'))
        if not v.K or v.transform().m11() <= 0:
            return
        mm_per_pixel = 1/(v.K*v.transform().m11())
        target = mm_per_pixel*65
        base = 10**math.floor(math.log10(max(target, 1e-9)))
        step = next(x*base for x in (1, 2, 5, 10) if x*base >= target)
        start = v.mapToScene(0, 0)
        end = v.mapToScene(v.viewport().width(), v.viewport().height())
        lo, hi = ((start.y()/v.K, end.y()/v.K) if self.vertical
                  else (start.x()/v.K, end.x()/v.K))
        painter.setFont(QFont('DejaVu Sans', 8))
        painter.setPen(QColor('#718486'))
        minor = step/5
        first, last = math.floor(lo/minor), math.ceil(hi/minor)
        for i in range(first, min(last+1, first+2000)):
            value = i*minor
            if self.vertical:
                pos = v.mapFromScene(QPointF(0, value*v.K)).y()
            else:
                pos = v.mapFromScene(QPointF(value*v.K, 0)).x()
            major = i % 5 == 0
            length = 10 if major else 4
            if self.vertical:
                painter.drawLine(32-length, pos, 32, pos)
                if major:
                    painter.save()
                    painter.translate(12, pos+3)
                    painter.rotate(-90)
                    label = self.studio.sp_h.value()-value
                    painter.drawText(0, 0, f'{label:g}')
                    painter.restore()
            else:
                painter.drawLine(pos, 26-length, pos, 26)
                if major:
                    painter.drawText(pos+3, 11, f'{value:g}')
        painter.end()


class StoneStudio(InteractiveManipulationMixin, core.Studio):
    def __init__(self):
        self._shape_kind = None
        self._shape_start = self._shape_preview = None
        self._polygon_sides = 6
        self._stroke = None
        self._preview_tool = None
        self.settings = QSettings('StoneReliefStudio', 'Workspace')
        self._recent_projects = []
        try:
            recent = json.loads(str(self.settings.value('recentProjects', '[]')))
            if isinstance(recent, list):
                self._recent_projects = list(dict.fromkeys(p for p in recent if isinstance(p, str) and p))[:6]
        except (ValueError, TypeError):
            pass
        try:
            self.history_max_bytes = max(64, min(4096, int(self.settings.value('historyBudgetMiB',256))))*1024**2
        except (ValueError, TypeError):
            self.history_max_bytes = 256*1024**2
        super().__init__()
        self.setStyleSheet(STYLE)
        self.setWindowIcon(icon('dome', True))
        self.setWindowTitle('Stone Relief Studio Pro 3.3.1 — векторы и рельеф')
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
        self._install_manipulation()
        self._restore_user_preferences()
        self._saved_sig = self.state_sig()
        self.statusBar().showMessage('Векторы — слева. Формы и рельеф — сверху. F12 — редактор формы.')

    def _action(self, label, name, slot, shortcut=None, checkable=False):
        action = QAction(icon(name), label, self)
        action.setCheckable(checkable)
        action.setToolTip(label+(f' ({shortcut})' if shortcut else ''))
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
        for key, label, glyph, slot, shortcut in [
            ('new','Новый проект','new',self.new_project,'Ctrl+N'),
            ('open','Открыть проект…','open',self.open_project,'Ctrl+O'),
            ('save','Сохранить','save',self.save_project,'Ctrl+S'),
            ('saveas','Сохранить как…','save',self.save_project_as,'Ctrl+Shift+S'),
            ('undo','Отменить','undo',self.undo,'Ctrl+Z'),
            ('redo','Повторить','redo',self.redo,'Ctrl+Y'),
            ('copy','Создать копию','copy',self.copy_selected,'Ctrl+D'),
            ('delete','Удалить выделенное','delete',self.delete_selected,None),
            ('transform','Размер, поворот и положение…','transform',self.transform_dialog,None),
            ('photo','Открыть фотографию…','photo',self.open_photo,'Ctrl+I'),
            ('svg','Импорт SVG 1:1…','open',self.import_svg,None),
            ('dxf','Импорт DXF 1:1…','open',self.import_dxf,None),
            ('svgout','Экспорт SVG…','save',self.export_svg,None),
            ('stlin','Открыть STL…','stl',self.open_stl,None),
            ('stl','Экспорт STL…','stl',self.export_stl,'Ctrl+E'),
            ('shape','Форма · F12','shape',self.open_relief_editor,'F12'),
            ('sweep','Две направляющие','rails',self.open_sweep_dialog,None),
            ('dome','Купол','dome',self.create_dome,None),
            ('plate','Плоская основа','plate',self.create_plate,None),
            ('fromphoto','Рельеф из фото','photo',self.create_relief,None),
            ('smooth','Сглаживание…','smooth',lambda:self.relief_op('smooth'),None),
            ('zero','Обнулить…','zero',lambda:self.relief_op('zero'),'Shift+Delete'),
            ('fit','Показать модель целиком','fit',self.fit_model,'Home'),
            ('fit_selection','Приблизить выделенное · 2D','fit',self.fit_selection,'Ctrl+Home'),
            ('text','Текст в кривые…','text',self.add_text,None),
        ]:
            actions[key] = self._action(label, glyph, slot, shortcut)
        menus = {}
        for title in ('Файл','Правка','Вид','Модель','Векторы','Рельефы','Справка'):
            menus[title] = self.menuBar().addMenu(title)
        for key in ('new','open','save','saveas','photo','svg','dxf','svgout','stlin','stl'):
            menus['Файл'].addAction(actions[key])
        self.recent_menu = QMenu('Недавние проекты', self)
        self.recent_menu.setToolTipsVisible(True)
        menus['Файл'].insertMenu(actions['save'], self.recent_menu)
        self._refresh_recent_projects()
        menus['Файл'].addSeparator()
        menus['Файл'].addAction('Выход', self.close)
        for key in ('undo','redo','copy','delete','transform'):
            menus['Правка'].addAction(actions[key])
        for key in ('plate','fromphoto'):
            menus['Модель'].addAction(actions[key])
        self.dimensions_action = QAction('РАЗМЕРЫ МОДЕЛИ', self)
        self.dimensions_action.triggered.connect(lambda: self.model_dimensions_dialog())
        menus['Модель'].addAction(self.dimensions_action)
        for key in ('text','svg','dxf','transform'):
            menus['Векторы'].addAction(actions[key])
        menus['Векторы'].addAction('Соединить два открытых вектора', self.join_selected)
        menus['Векторы'].addAction('Группировать', self.group_selected)
        menus['Векторы'].addAction('Разгруппировать', self.ungroup_selected)
        menus['Векторы'].addAction('Зеркально по горизонтали', lambda:self.mirror_selected('h'))
        menus['Векторы'].addAction('Зеркально по вертикали', lambda:self.mirror_selected('v'))
        for key in ('shape','sweep','dome','smooth','zero'):
            menus['Рельефы'].addAction(actions[key])
        for title, op in [('Масштаб высоты…','scale'),('Инвертировать','invert'),('Смещение высоты…','offset')]:
            menus['Рельефы'].addAction(title, lambda op=op:self.relief_op(op))
        menus['Рельефы'].addAction('Отразить рельеф горизонтально', lambda:self.relief_op('mirror_h'))
        menus['Рельефы'].addAction('Отразить рельеф вертикально', lambda:self.relief_op('mirror_v'))
        menus['Рельефы'].addAction(actions['stl'])
        menus['Справка'].addAction(self._action('Справочник инструментов','new',self.open_help,'F1'))
        self.help_button=QToolButton(self.menuBar())
        self.help_button.setIcon(icon('book',True)); self.help_button.setIconSize(QSize(24,24))
        self.help_button.setToolTip('Книга инструментов на русском · F1')
        self.help_button.clicked.connect(self.open_help)
        self.menuBar().setCornerWidget(self.help_button,Qt.TopRightCorner)
        menus['Справка'].addAction('О программе', lambda:QMessageBox.information(self, 'Stone Relief Studio',
            'Векторное рисование и моделирование рельефа для камня.\nСамостоятельное оформление и значки.\nВерсия рабочего пространства 3.3.1.'))

        filebar = self._toolbar('Проект и правка')
        for key in ('new','open','save','undo','redo','copy','delete','transform'):
            filebar.addAction(actions[key])
        filebar.addSeparator()
        self.dimensions_button = QPushButton('РАЗМЕРЫ МОДЕЛИ')
        self.dimensions_button.setProperty('primary', True)
        self.dimensions_button.clicked.connect(lambda: self.model_dimensions_dialog())
        filebar.addWidget(self.dimensions_button)
        brand = QLabel('  STONE  /  RELIEF STUDIO  ')
        brand.setStyleSheet('font-size: 15px; font-weight: 700; color:#466b64; padding:4px 12px;')
        filebar.addWidget(brand)
        self.addToolBarBreak()
        relief = self._toolbar('Создание и редактирование рельефа', large=True)
        for key in ('shape','sweep','dome','plate','fromphoto'):
            relief.addAction(actions[key])
        relief.addSeparator()
        self.sculpt_action = self._action('Лепка в 3D', 'sculpt', self.toggle_sculpt, checkable=True)
        relief.addAction(self.sculpt_action)
        menus['Рельефы'].addAction(self.sculpt_action)
        for key in ('smooth','zero','stl'):
            relief.addAction(actions[key])

        vectors = self._toolbar('Векторы', Qt.LeftToolBarArea)
        vectors.setMovable(False)
        self.tool_group = QActionGroup(self)
        self.tool_actions = {}
        for key, label, glyph in [('select','Выделение и перемещение','select'),
                                  ('draw','Линии и кривые Безье','pen'),
                                  ('ctrl','Управляющие точки на чертеже','point')]:
            action = self._action(label, glyph, lambda checked,k=key:self.set_tool(k), checkable=True)
            self.tool_group.addAction(action)
            vectors.addAction(action)
            self.tool_actions[key] = action
        vectors.addSeparator()
        for kind, label in [('rectangle','Прямоугольник'),('square','Квадрат'),
                            ('circle','Круг'),('ellipse','Эллипс'),('star','Звезда'),('polygon','Многоугольник')]:
            a = self._action(label+' — нарисовать мышью', kind,
                             lambda checked,k=kind:self.choose_shape(k), checkable=True)
            self.tool_group.addAction(a)
            self.tool_actions[kind] = a
            vectors.addAction(a)
        vectors.addAction(actions['text'])
        vectors.addSeparator()
        vectors.addAction(actions['fit'])
        vectors.addAction(actions['fit_selection'])
        menus['Вид'].addAction(actions['fit'])
        menus['Вид'].addAction(actions['fit_selection'])

        snapbar = self._toolbar('Привязки и точность')
        self.removeToolBar(snapbar)
        self.addToolBar(Qt.BottomToolBarArea, snapbar)
        self.snapbar = snapbar
        snapbar.show()
        self.snap_actions = {}
        for key,label in [('nodes','Узлы'),('lines','Линии'),('grid','Сетка'),('showgrid','Показать сетку')]:
            a=QAction(label,self); a.setCheckable(True)
            a.toggled.connect(lambda on,k=key:self.on_snap(k,on))
            snapbar.addAction(a); self.snap_actions[key]=a
        snapbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        snapbar.addSeparator()
        snapbar.addWidget(QLabel(' Шаг узла: '))
        self.nudge = core.QComboBox()
        self.nudge.addItems(['0,01 мм','0,1 мм','1 мм'])
        self.nudge.setCurrentIndex(1)
        self.nudge.currentIndexChanged.connect(lambda i:setattr(self.view,'keyboard_nudge_mm',(0.01,0.1,1.0)[i]))
        snapbar.addWidget(self.nudge)
        for toolbar in (filebar,relief,vectors,snapbar):
            menus['Вид'].addAction(toolbar.toggleViewAction())

        dock = self.findChildren(QDockWidget)[0]
        dock.setObjectName('inspector')
        dock.setWindowTitle('Модель и инструменты')
        dock.setFeatures(QDockWidget.DockWidgetClosable|QDockWidget.DockWidgetMovable|QDockWidget.DockWidgetFloatable)
        dock.setMinimumWidth(290)
        self.inspector = dock
        self.inspector_tabs = dock.widget()
        self.inspector_tabs.setTabPosition(core.QTabWidget.North)
        self.inspector_tabs.setUsesScrollButtons(True)
        model = QWidget(); form=QFormLayout(model)
        form.setContentsMargins(14,14,14,14)
        title=QLabel('РАЗМЕРЫ И ОСНОВА'); title.setStyleSheet('font-weight:700;color:#628074;')
        form.addRow(title)
        self.selection_info = QLabel('Выделите вектор — здесь появятся его размеры.')
        self.selection_info.setWordWrap(True)
        form.addRow(self.selection_info)
        for label,widget in [('Ширина',self.sp_w),('Высота',self.sp_h),('Основание',self.sp_base),
                             ('Высота рельефа / фото',self.sp_depth),('Шаг рельефа',self.sp_step),('Шаг привязки',self.sp_grid)]:
            form.addRow(label,widget)
            widget.show()
        self.stl_min_wall=self._spin(.5,.05,200,3,' мм',.1)
        self.stl_precision=self._spin(.001,.000001,1,6,' мм',.001)
        self.stl_min_wall.setToolTip('Экспорт карты высот требует не меньшего расстояния до плоского дна. Для произвольной сетки такой проверки толщины пока нет.')
        self.stl_precision.setToolTip('Максимальная допустимая ошибка координаты при записи STL в float32.')
        form.addRow('STL: мин. основание',self.stl_min_wall)
        form.addRow('STL: допуск координат',self.stl_precision)
        form.addRow(self.chk_invert)
        form.addRow('Сглаживание фото',self.sp_blur)
        self.chk_invert.show(); self.sp_blur.show()
        for key in ('plate','photo','fromphoto','svg','dxf'):
            b=QPushButton(actions[key].text()); b.setIcon(actions[key].icon())
            b.clicked.connect(actions[key].trigger); form.addRow(b)
        hint=QLabel('Размеры — в миллиметрах.\nКривые: ЛКМ — узел, протянуть — ручки.\nEnter — завершить, Esc — отменить.\nУзлы: Tab, стрелки; Ctrl — 0,01 мм.'); hint.setWordWrap(True)
        hint.setStyleSheet('color:#60777c;padding-top:10px;'); form.addRow(hint)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(model)
        self.inspector_tabs.insertTab(0,scroll,'Модель')
        self.inspector_tabs.setCurrentIndex(0)
        for label in self.inspector_tabs.findChildren(QLabel):
            if len(label.text())>60: label.setWordWrap(True)
        for combo in self.inspector_tabs.findChildren(core.QComboBox):
            combo.setMinimumContentsLength(15)
            combo.setSizeAdjustPolicy(core.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        menus['Вид'].addAction(dock.toggleViewAction())
        menus['Модель'].addAction('Размеры и параметры', lambda:self.show_inspector(0))

        history_layout = self.hist.parentWidget().layout()
        budget_row = QWidget(); budget_form = QFormLayout(budget_row)
        self.history_budget = core.QSpinBox()
        self.history_budget.setRange(64,4096); self.history_budget.setSingleStep(64)
        self.history_budget.setSuffix(' МБ'); self.history_budget.setValue(self.history_max_bytes//1024**2)
        self.history_budget.valueChanged.connect(self.set_history_budget)
        budget_form.addRow('Лимит буферов истории',self.history_budget)
        self.lb_history_memory = QLabel()
        self.lb_history_memory.setWordWrap(True)
        history_layout.insertWidget(0,budget_row)
        history_layout.insertWidget(1,self.lb_history_memory)
        self.refresh_history()

        self.view.setBackgroundBrush(QColor('#75858b'))
        # Add millimetre rulers around the existing view, preserving its scene and tools.
        parent_layout=self.tabs.widget(0).layout()
        parent_layout.removeWidget(self.view)
        wrap=QWidget(); grid=QGridLayout(wrap); grid.setSpacing(0); grid.setContentsMargins(0,0,0,0)
        self.ruler_x=Ruler(self); self.ruler_y=Ruler(self,True)
        corner=QLabel('мм'); corner.setAlignment(Qt.AlignCenter); corner.setStyleSheet('background:#edf1ee;color:#748689;')
        grid.addWidget(corner,0,0); grid.addWidget(self.ruler_x,0,1)
        grid.addWidget(self.ruler_y,1,0); grid.addWidget(self.view,1,1)
        grid.setColumnStretch(1,1); grid.setRowStretch(1,1)
        parent_layout.addWidget(wrap,1)
        self.view.horizontalScrollBar().valueChanged.connect(self.update_rulers)
        self.view.verticalScrollBar().valueChanged.connect(self.update_rulers)
        self.coords=QLabel('X 0,00   Y 0,00 мм')
        self.statusBar().addPermanentWidget(self.coords)
        self.tabs.setTabText(0,'Чертёж · 2D'); self.tabs.setTabText(1,'Поверхность · 3D')
        self.plotter.set_background('#64777c',top='#b7c6c5')
        camera_row=self.tabs.widget(1).layout().itemAt(0).layout()
        while camera_row.count():
            item=camera_row.takeAt(0)
            if item.widget() and item.widget() is not self.cb_mode3d:
                item.widget().deleteLater()
        camera_row.addWidget(self.cb_mode3d)
        self.cb_mode3d.show()
        camera=core.QComboBox()
        camera.addItems(['Под углом','Сверху','Спереди','Слева','Справа','Сзади','Снизу'])
        camera.currentIndexChanged.connect(lambda i:self.set_view(('iso','top','front','left','right','back','bottom')[i]))
        camera_row.addWidget(camera)
        fit_button = QToolButton()
        fit_button.setDefaultAction(actions['fit'])
        fit_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        camera_row.addWidget(fit_button)
        camera_row.addStretch(1)
        camera_row.addWidget(QLabel('Мышь — вращение · колесо — масштаб'))
        self.sp_w.setValue(300); self.sp_h.setValue(500)

    def fit_model(self):
        if self.tabs.currentIndex() == 1:
            self._sculpt_release()
            self.plotter.reset_camera()
            self.plotter.render()
        else:
            self.view.fit_all()
            self.update_rulers()

    def _selected_bounds(self):
        bounds = None
        for item in self.sel_vectors():
            # Exclude pens and node handles: these are geometric dimensions.
            rect = item.mapRectToScene(core.build_qpath(item.data).boundingRect())
            if bounds is None:
                bounds = QRectF(rect)
            else:
                left, top = min(bounds.left(),rect.left()), min(bounds.top(),rect.top())
                right, bottom = max(bounds.right(),rect.right()), max(bounds.bottom(),rect.bottom())
                bounds = QRectF(left,top,right-left,bottom-top)
        return bounds

    def fit_selection(self):
        bounds = self._selected_bounds()
        if bounds is None:
            self.statusBar().showMessage('Сначала выделите один или несколько векторов в 2D.')
            return
        self._sculpt_release()
        self.tabs.setCurrentIndex(0)
        margin = max(bounds.width(), bounds.height()) * .12 + self.K
        self.view.fitInView(bounds.adjusted(-margin,-margin,margin,margin), Qt.KeepAspectRatio)
        self.update_rulers()

    def update_selection_ui(self):
        super().update_selection_ui()
        if not hasattr(self, 'selection_info'):
            return
        bounds = self._selected_bounds()
        if bounds is None:
            self.selection_info.setText('Выделите вектор — здесь появятся его размеры.')
        else:
            count = len(self.sel_vectors())
            self.selection_info.setText(f'Выделено векторов: {count}\n'
                f'Размер: {bounds.width()/self.K:.2f} × {bounds.height()/self.K:.2f} мм\n'
                f'Левый нижний угол: X {bounds.left()/self.K:.2f}; '
                f'Y {self.sp_h.value()-bounds.bottom()/self.K:.2f} мм')
        self._shell_actions['fit_selection'].setEnabled(bounds is not None)

    def _refresh_recent_projects(self):
        self.recent_menu.clear()
        if not self._recent_projects:
            self.recent_menu.addAction('Пока нет проектов').setEnabled(False)
            return
        for path in self._recent_projects:
            action = self.recent_menu.addAction(os.path.basename(path).replace('&', '&&'))
            action.setToolTip(path)
            action.setStatusTip(path)
            action.setData(path)
            action.triggered.connect(lambda checked=False, p=path: self.open_project(p))
        self.recent_menu.addSeparator()
        self.recent_menu.addAction('Очистить список', self._clear_recent_projects)

    def _remember_project(self, path):
        path = os.path.abspath(path)
        self._recent_projects = [path] + [p for p in self._recent_projects if os.path.normcase(p) != os.path.normcase(path)]
        self._recent_projects = self._recent_projects[:6]
        self.settings.setValue('recentProjects', json.dumps(self._recent_projects, ensure_ascii=False))
        self._refresh_recent_projects()

    def _clear_recent_projects(self):
        self._recent_projects = []
        self.settings.setValue('recentProjects', '[]')
        self._refresh_recent_projects()

    def _restore_user_preferences(self):
        try:
            data = json.loads(str(self.settings.value('userPreferences', '{}')))
        except (ValueError, TypeError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        for key, widget in [('radius',self.sculpt_radius), ('gain',self.sculpt_gain)]:
            value = data.get(key)
            if type(value) in (int,float) and widget.minimum() <= value <= widget.maximum():
                widget.setValue(value)
        for key, widget in [('fixed',self.sculpt_fixed), ('protect_base',self.sculpt_protect_base)]:
            if type(data.get(key)) is bool:
                widget.setChecked(data[key])
        index = data.get('nudge')
        if type(index) is int and 0 <= index < self.nudge.count():
            self.nudge.setCurrentIndex(index)
        for key, action in self.snap_actions.items():
            if type(data.get(key)) is bool:
                action.setChecked(data[key])
        for widget in (self.sculpt_radius,self.sculpt_gain):
            widget.valueChanged.connect(self._save_user_preferences)
        for widget in (self.sculpt_fixed,self.sculpt_protect_base):
            widget.toggled.connect(self._save_user_preferences)
        self.nudge.currentIndexChanged.connect(self._save_user_preferences)
        for action in self.snap_actions.values():
            action.toggled.connect(self._save_user_preferences)
        self.update_selection_ui()

    def _save_user_preferences(self, *_):
        data = dict(radius=self.sculpt_radius.value(),gain=self.sculpt_gain.value(),
                    fixed=self.sculpt_fixed.isChecked(),protect_base=self.sculpt_protect_base.isChecked(),
                    nudge=self.nudge.currentIndex(),**{k:a.isChecked() for k,a in self.snap_actions.items()})
        self.settings.setValue('userPreferences',json.dumps(data))

    def show_inspector(self, index):
        self.inspector.show(); self.inspector.raise_(); self.inspector_tabs.setCurrentIndex(index)

    def set_history_budget(self, mib):
        self.history_max_bytes = int(mib)*1024**2
        self.settings.setValue('historyBudgetMiB',int(mib))
        self._trim_history()
        self.refresh_history()

    def update_rulers(self, *_):
        self.ruler_x.update(); self.ruler_y.update()

    def set_tool(self, tool):
        self._shape_kind=None; self._shape_start=None
        self._clear_shape_preview()
        super().set_tool(tool)

    def choose_shape(self, kind):
        if kind=='polygon':
            sides,ok=QInputDialog.getInt(self,'Многоугольник','Количество сторон:',self._polygon_sides,3,64)
            if not ok:
                self.set_tool('select'); return
            self._polygon_sides=sides
        self._shape_kind=kind; self._shape_start=None
        self._clear_shape_preview(); self.view.set_mode('shape')
        self.statusBar().showMessage('Нажмите и протяните мышью. Esc — отмена. Размер можно уточнить через «Правка → Размер, поворот и положение».')

    def _clear_shape_preview(self):
        if self._shape_preview is not None:
            self.scene.removeItem(self._shape_preview)
            self._shape_preview=None

    def shape_data(self, a, b):
        x1,y1,x2,y2=a.x(),a.y(),b.x(),b.y()
        if self._shape_kind in ('square','circle'):
            size=max(abs(x2-x1),abs(y2-y1))
            x2=x1+math.copysign(size,x2-x1); y2=y1+math.copysign(size,y2-y1)
        left,right=sorted((x1,x2)); top,bottom=sorted((y1,y2))
        cx,cy=(left+right)/2,(top+bottom)/2
        rx,ry=(right-left)/2,(bottom-top)/2
        if self._shape_kind in ('circle','ellipse'):
            return core._ellipse_path(cx,cy,rx,ry)
        if self._shape_kind in ('square','rectangle'):
            points=[(left,top),(right,top),(right,bottom),(left,bottom)]
        else:
            count=10 if self._shape_kind=='star' else self._polygon_sides
            points=[]
            for i in range(count):
                scale=.43 if self._shape_kind=='star' and i%2 else 1
                angle=-math.pi/2+2*math.pi*i/count
                points.append((cx+rx*scale*math.cos(angle),cy+ry*scale*math.sin(angle)))
        return core.polyline_path(points,True)

    def eventFilter(self, obj, event):
        if obj is self.view.viewport():
            typ=event.type()
            if typ in (QEvent.Resize,QEvent.Wheel,QEvent.Paint):
                if typ!=QEvent.Paint:
                    QTimer.singleShot(0,self.update_rulers)
            if typ==QEvent.MouseMove:
                p=self.view.mapToScene(event.position().toPoint())
                self.coords.setText(f'X {p.x()/self.K:.2f}   Y {self.sp_h.value()-p.y()/self.K:.2f} мм')
            if self._shape_kind:
                if typ==QEvent.MouseButtonPress and event.button()==Qt.LeftButton:
                    self._shape_start=self.view.snap_point(self.view.mapToScene(event.position().toPoint())); return True
                if typ==QEvent.MouseMove and self._shape_start is not None:
                    b=self.view.snap_point(self.view.mapToScene(event.position().toPoint()))
                    if self._shape_preview is None:
                        self._shape_preview=core.QGraphicsPathItem()
                        pen=QPen(QColor('#427d70'),1,Qt.DashLine); pen.setCosmetic(True)
                        self._shape_preview.setPen(pen); self._shape_preview.setZValue(90)
                        self.scene.addItem(self._shape_preview)
                    self._shape_preview.setPath(core.build_qpath(self.shape_data(self._shape_start,b))); return True
                if typ==QEvent.MouseButtonRelease and event.button()==Qt.LeftButton:
                    if self._shape_start is not None:
                        b=self.view.snap_point(self.view.mapToScene(event.position().toPoint()))
                        data=self.shape_data(self._shape_start,b)
                        bounds=core.build_qpath(data).boundingRect()
                        if bounds.width()>1e-4 and bounds.height()>1e-4:
                            snap=self.snapshot(); self.scene.clearSelection(); self._add_vec(data,True)
                            self.push_history('Создание фигуры',snap); self.after_change()
                    self.set_tool('select'); return True
                if typ==QEvent.MouseButtonPress and event.button()==Qt.RightButton:
                    self.set_tool('select'); return True
        if obj in (getattr(self,'sweep_dialog',None),getattr(self,'relief_dialog',None)):
            if event.type() in (QEvent.Hide,QEvent.Close):
                name='sweep' if obj is self.sweep_dialog else 'shape'
                self.settings.setValue(name+'/geometry',obj.saveGeometry())
                self.settings.setValue('parameters',json.dumps(self._parameters(),ensure_ascii=False))
        return super().eventFilter(obj,event)

    def add_text(self):
        dialog=QDialog(self); dialog.setWindowTitle('Надпись → векторные контуры')
        form=QFormLayout(dialog)
        text=QLineEdit('Название'); font=QFontComboBox()
        height=self._spin(15,0.5,500,2); bold=QCheckBox('Полужирный')
        form.addRow('Текст',text); form.addRow('Шрифт',font); form.addRow('Высота букв',height); form.addRow(bold)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); form.addRow(buttons)
        if dialog.exec()!=QDialog.Accepted or not text.text().strip():
            return
        selected_font=font.currentFont(); selected_font.setPixelSize(100); selected_font.setBold(bold.isChecked())
        path=QPainterPath(); path.addText(0,0,selected_font,text.text())
        bounds=path.boundingRect()
        if bounds.height()==0:
            return
        scale=height.value()*self.K/bounds.height()
        # QPainterPath exposes exact cubic outlines, including Cyrillic font glyphs.
        paths=[]; nodes=[]; i=0
        def flush():
            nonlocal nodes
            if len(nodes)>=2:
                if math.hypot(nodes[0].x-nodes[-1].x,nodes[0].y-nodes[-1].y)<1e-7:
                    nodes[0].hin=nodes[-1].hin; nodes.pop()
                paths.append(core.PathData(nodes,True))
            nodes=[]
        xy=lambda e:((e.x-bounds.left())*scale+self.canvas_w*.1,
                      (e.y-bounds.top())*scale+self.canvas_h*.3)
        while i<path.elementCount():
            e=path.elementAt(i)
            if e.isMoveTo():
                flush(); nodes=[core.Node(*xy(e))]
            elif e.isLineTo():
                nodes.append(core.Node(*xy(e)))
            elif e.isCurveTo():
                c2=path.elementAt(i+1); end=path.elementAt(i+2)
                nodes[-1].hout=xy(e); nodes.append(core.Node(*xy(end),hin=xy(c2))); i+=2
            i+=1
        flush()
        snap=self.snapshot(); self.scene.clearSelection(); group=self._new_gid()
        for data in paths:
            data.group=group; self._add_vec(data,True)
        self.push_history('Текст в кривые',snap); self.after_change(); self.tabs.setCurrentIndex(0)

    def _build_sweep_dialog(self):
        dialog=super()._build_sweep_dialog()
        dialog.setWindowTitle('Поверхность по двум направляющим')
        dialog.setSizeGripEnabled(True); dialog.resize(560,720)
        dialog.setModal(False)
        intro=dialog.findChildren(QLabel)[0]
        intro.setText('1. Нажмите «Выбор», затем один раз щёлкните открытый вектор в 2D — Enter не нужен.\n'
                      'Первая направляющая синяя, вторая оранжевая; сечения фиолетовые.\n'
                      '2. Добавьте поперечные сечения и задайте их положение.\n3. Проверьте поверхность в 3D и примените результат.')
        for b in dialog.findChildren(QPushButton):
            if b.text()=='Вычислить': b.setText('Применить'); b.setProperty('primary',True)
            if b.text().startswith('Положение выбранного'): b.setText('Положение сечения вдоль кривых, %')
        scroll=dialog.findChild(QScrollArea)
        layout=scroll.widget().layout()
        footer=layout.takeAt(layout.count()-1).layout()
        row=QHBoxLayout()
        while footer.count():
            item=footer.takeAt(0)
            if item.widget(): row.addWidget(item.widget())
        dialog.layout().addLayout(row)
        dialog.installEventFilter(self)
        return dialog

    def _build_shape_dialog(self):
        dialog=super()._build_shape_dialog()
        dialog.setWindowTitle('Форма и высота · F12'); dialog.setModal(False)
        dialog.setSizeGripEnabled(True); dialog.setMinimumSize(440,480); dialog.resize(590,630)
        # Existing controls remain connected; the scroll area makes every action reachable.
        content=QWidget()
        content.setLayout(dialog.layout())
        layout=QVBoxLayout(dialog)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(content); layout.addWidget(scroll)
        for i,b in enumerate(self.sh_kind_btns):
            b.setText(''); b.setIcon(icon(('dome','polygon','plate')[i],True)); b.setIconSize(QSize(34,34))
        for b in dialog.findChildren(QPushButton):
            if b.text()=='Применить': b.setProperty('primary',True)
        dialog.installEventFilter(self)
        return dialog

    def _restore_dialog(self, dialog, name):
        geometry=self.settings.value(name+'/geometry')
        if geometry: dialog.restoreGeometry(geometry)

    def sweep_state(self):
        if not hasattr(self,'sw_flip1'):
            return {}
        uid=lambda item:item.data.uid if item is not None and self._alive(item) else None
        return dict(rails=[uid(x) for x in self.sw_rails], sections=[uid(x) for x in self.sw_sections],
                    z=uid(self.sw_zb), positions=dict(self.sw_positions),
                    flip1=self.sw_flip1.isChecked(),flip2=self.sw_flip2.isChecked(),use_z=self.sw_use_z.isChecked(),
                    transition=self.sw_trans.isChecked(),scale_w=self.sw_scalew.isChecked(),
                    scale_end=self.sw_scaleend.isChecked(),start=self.sw_start.value(),end=self.sw_end.value(),
                    mode=next((b.property('key') for b in self.sw_mode_btns if b.isChecked()),'add'))

    def _parameters(self):
        form={}
        if hasattr(self,'sh_kind'):
            form=dict(kind=self.sh_kind,angle=self.sh_angle.value(),start=self.sh_start.value(),
                      scale=self.sh_scale_sl.value(),limit=self.sh_limit.value(),height=self.sh_scaleh.value(),
                      const=self.sh_const.value(),height_mode=self.sh_height_mode()[0],combine=self.sh_mode_key())
            form['exact'] = self.sh_exact.value()
        return dict(sweep=self.sweep_state(),form=form)

    def _restore_parameters(self, data, bind=True):
        state=data.get('sweep',{})
        if hasattr(self,'sw_flip1'):
            for key,widget in [('flip1',self.sw_flip1),('flip2',self.sw_flip2),('use_z',self.sw_use_z),
                               ('transition',self.sw_trans),('scale_w',self.sw_scalew),('scale_end',self.sw_scaleend)]:
                if key in state: widget.setChecked(state[key])
            self.sw_start.setValue(state.get('start',0)); self.sw_end.setValue(state.get('end',0))
            for b in self.sw_mode_btns: b.setChecked(b.property('key')==state.get('mode','add'))
            if bind:
                vectors={v.data.uid:v for v in self.view.vectors()}
                self.sw_rails=[vectors.get(uid) for uid in state.get('rails',[None,None])]
                self.sw_sections=[vectors[uid] for uid in state.get('sections',[]) if uid in vectors]
                self.sw_zb=vectors.get(state.get('z')); self.sw_positions=dict(state.get('positions',{}))
                self.sw_refresh()
        form=data.get('form',{})
        if form and hasattr(self,'sh_kind'):
            self.sh_set_kind(int(form.get('kind',0)))
            for key,widget in [('angle',self.sh_angle),('start',self.sh_start),('scale',self.sh_scale_sl),
                               ('limit',self.sh_limit),('height',self.sh_scaleh),('const',self.sh_const),('exact',self.sh_exact)]:
                if key in form: widget.setValue(form[key])
            mode=form.get('height_mode','free')
            {'exact':self.sh_r_exact,'free':self.sh_r_free,'limit':self.sh_r_limit,'scale':self.sh_r_scale,'const':self.sh_r_const}.get(mode,self.sh_r_free).setChecked(True)
            for (_,key),b in zip(self.SH_MODES,self.sh_mode_btns): b.setChecked(key==form.get('combine','add'))

    def _settings(self):
        return dict(super()._settings(),workspace=self._parameters(),export_options=self._export_options())

    def _export_options(self):
        return dict(minimum_thickness=self.stl_min_wall.value(),precision=self.stl_precision.value())

    def _write(self, path):
        self._sculpt_release()
        saved = super()._write(path)
        if saved:
            self._remember_project(path)
        return saved

    def confirm_discard(self):
        # Commit the active gesture to history before any Save/Discard/Cancel decision.
        self._sculpt_release()
        return super().confirm_discard()

    def state_sig(self):
        import hashlib
        return hashlib.sha256((super().state_sig()+json.dumps(dict(workspace=self._parameters(),export=self._export_options()),sort_keys=True)).encode()).hexdigest()

    def snapshot(self):
        result=super().snapshot()
        result.update(photo=(self.photo_pil,self.photo_bytes,self.photo_name),mesh=self.stl_mesh,
                      dims=(self.sp_w.value(),self.sp_h.value(),self.sp_step.value()),
                      depth=self.sp_depth.value(),
                      selected=[v.data.uid for v in self.sel_vectors()],parameters=self._parameters())
        return result

    def restore(self, state):
        for widget,value in zip((self.sp_w,self.sp_h,self.sp_step),state.get('dims',())):
            widget.blockSignals(True); widget.setValue(value); widget.blockSignals(False)
        self.sp_depth.blockSignals(True)
        self.sp_depth.setValue(state.get('depth', self.sp_depth.value()))
        self.sp_depth.blockSignals(False)
        self.photo_pil,self.photo_bytes,self.photo_name=state.get('photo',(None,None,None))
        self.photo_arr=None; self.stl_mesh=state.get('mesh')
        super().restore(state)
        self.rebuild_canvas(force=True)
        self._restore_parameters(state.get('parameters',{}))
        selected=set(state.get('selected',[]))
        for v in self.view.vectors(): v.setSelected(v.data.uid in selected)
        self._v_key=None

    def open_project(self, path=None):
        if not isinstance(path,str):
            path,_=QFileDialog.getOpenFileName(self,'Открыть проект','','Stone Relief (*.srelief)')
        if not path: return
        if super().open_project(path):
            self._restore_parameters(self._loaded_project_settings.get('workspace',{}))
            self._irm_restore_state(self._loaded_project_settings.get('interactive',{}))
            export=self._loaded_project_settings.get('export_options',{})
            self.stl_min_wall.setValue(export.get('minimum_thickness',.5))
            self.stl_precision.setValue(export.get('precision',.001))
            self.update_rulers()
            self._saved_sig=self.state_sig()
            self._remember_project(path)
            return True
        return False

    def model_dimensions_dialog(self, new_project=False):
        dialog = QDialog(self)
        dialog.setWindowTitle('Новый проект — размеры модели' if new_project else 'РАЗМЕРЫ МОДЕЛИ')
        dialog.setSizeGripEnabled(True)
        layout = QVBoxLayout(dialog)
        note = QLabel('Все размеры в мм. X/Y масштабируют композицию; высота рельефа задаёт предел-предупреждение F12 и высоту из фото.\n'
                      'Существующий рельеф не обнуляется. Изменение основания сдвигает поверхность по Z, сохраняя рисунок.\n'
                      'Увеличение шага сетки может уменьшить детализацию.')
        note.setWordWrap(True); layout.addWidget(note)
        form = QFormLayout(); layout.addLayout(form)
        fields = []
        for label, original in [('Ширина X',self.sp_w),('Высота Y',self.sp_h),('Толщина основания',self.sp_base),
                                ('Высота рельефа',self.sp_depth),('Шаг сетки',self.sp_step)]:
            widget = self._spin(original.value(), original.minimum(), original.maximum(), 6)
            fields.append(widget); form.addRow(label + ' (мм)', widget)
        if self.stl_mesh is not None and not new_project:
            fields[2].setEnabled(False)
            fields[2].setToolTip('Для произвольной сетки толщина основания не определяется одним числом. Используйте Mesh Edit.')
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        self.dimensions_dialog = dialog
        self.dimensions_fields = fields
        if dialog.exec() != QDialog.Accepted:
            return False
        values = [field.value() for field in fields]
        if new_project:
            # Confirm/Save while the old project is intact; Cancel preserves it.
            if not self.confirm_discard():
                return False
            self._clear_all(); self.project_path = None
            self.setWindowTitle(core.APP)
            self.sw_reset(); self._v_key = None
        changed = self.apply_model_dimensions(*values, confirm=not new_project)
        if new_project:
            self.update_lock(); self.rebuild_canvas(force=True)
            self._saved_sig = self.state_sig()
        return changed

    def apply_model_dimensions(self, width, height, base, depth, step, confirm=True):
        """Resize the existing composition; preserve Z detail and history in mm."""
        widgets = (self.sp_w,self.sp_h,self.sp_base,self.sp_depth,self.sp_step)
        values = tuple(float(v) for v in (width,height,base,depth,step))
        if any(not math.isfinite(v) or not w.minimum() <= v <= w.maximum() for w,v in zip(widgets,values)):
            raise ValueError('Размеры вне допустимых пределов. Проверьте значения в мм.')
        old = tuple(w.value() for w in widgets)
        if values == old:
            return True
        if self.stl_mesh is not None and base != old[2]:
            raise ValueError('Основание произвольной сетки изменяется в Mesh Edit.')
        if confirm and self.has_content():
            answer = QMessageBox.question(self, 'Изменение размеров модели',
                f'Размер X/Y: {old[0]:g} × {old[1]:g} → {width:g} × {height:g} мм.\n'
                f'Основание: {old[2]:g} → {base:g} мм; шаг: {old[4]:g} → {step:g} мм.\n'
                'Композиция будет масштабирована по X/Y; высоты рельефа сохранятся. '
                'При изменении шага сетка пересчитывается, возможна потеря мелких деталей.\nПродолжить? Ctrl+Z отменит операцию.',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return False
        self._sculpt_release(); self._irm_release(); self._irm_cancel(silent=True)
        self.cancel_preview(silent=True)
        snap = self.snapshot()
        new_z = self.z
        if new_z is not None:
            if (width,height,step) != (old[0],old[1],old[4]):
                from scipy.ndimage import zoom
                nx,ny,used = core.grid_from_step(width,height,step)
                new_z = zoom(new_z, (ny/new_z.shape[0],nx/new_z.shape[1]), order=1, prefilter=False)
            new_z = new_z + base-old[2]
            if float(new_z.min()) < .05:
                QMessageBox.warning(self,'Толщина основания','Изменение создаёт Z ниже 0,05 мм. Увеличьте основание; исходная модель сохранена.')
                return False
        new_mesh = self.stl_mesh
        if new_mesh is not None:
            new_mesh = new_mesh.copy()
            vertices = np.array(new_mesh.vertices, dtype=float, copy=True)
            origin = vertices[:,:2].min(axis=0)
            vertices[:,:2] = origin + (vertices[:,:2]-origin)*[width/old[0],height/old[1]]
            new_mesh.vertices = vertices
        old_canvas = (self.canvas_w,self.canvas_h)
        for widget,value in zip(widgets,values):
            widget.blockSignals(True); widget.setValue(value); widget.blockSignals(False)
        self._base_prev = base
        self.z, self.stl_mesh = new_z,new_mesh
        self.rebuild_canvas(force=True)
        fx,fy = self.canvas_w/old_canvas[0],self.canvas_h/old_canvas[1]
        for vector in self.view.vectors():
            vector.data.transform(lambda x,y:(x*fx,y*fy)); vector.refresh()
        for point in self.view.ctrl_points():
            point.setPos(point.pos().x()*fx,point.pos().y()*fy)
        for point in self._irm_controls:
            point['x'] *= width/old[0]; point['y'] *= height/old[1]; point['z'] += base-old[2]
        self.push_history('Размеры модели (мм)',snap)
        self.sw_refresh(); self.show_mesh(reset=True)
        self.after_change(); self.update_rulers(); self._irm_refresh_list()
        self.statusBar().showMessage(f'Модель {width:g} × {height:g} мм; основание {base:g} мм. Ctrl+Z — отменить.')
        return True

    def new_project(self):
        return self.model_dimensions_dialog(new_project=True)

    def undo(self):
        self._sculpt_release()
        self._irm_release()
        self._irm_cancel()
        return super().undo()

    def redo(self):
        self._sculpt_release()
        self._irm_release()
        self._irm_cancel()
        return super().redo()

    def preview_deform(self):
        self._preview_tool='deform'; super().preview_deform()

    def shape_apply(self, preview=False):
        self._preview_tool='shape'; super().shape_apply(preview)

    def sw_run(self, preview):
        self._preview_tool='sweep'; super().sw_run(preview)

    def _refresh_preview(self):
        if not self.preview_active: return
        if self._preview_tool=='shape': self.shape_apply(True)
        elif self._preview_tool=='sweep': self.sw_run(True)
        else: super()._refresh_preview()

    def _install_sculpt(self):
        page=QWidget(); form=QFormLayout(page)
        helptext=QLabel('Выберите замкнутый контур лепестка в 2D, затем включите «Лепка в 3D».\n\nНажмите на поверхность и тяните мышь вверх или вниз. Мелкая фактура сохраняется: изменяется только высота.\n\nБез контура изменяется область кисти. Ctrl+Z отменяет весь жест.\n\n0 рельефа — верх основания. Для углублений ниже этого уровня отключите защиту толщины. Дно плиты находится на Z=0.')
        helptext.setWordWrap(True); form.addRow(helptext)
        self.sculpt_radius=self._spin(15,.5,500,2)
        self.sculpt_gain=self._spin(.05,.001,5,3,' мм/px',.01)
        self.sculpt_fixed=QCheckBox('Зафиксировать край выделенного контура'); self.sculpt_fixed.setChecked(True)
        self.sculpt_protect_base=QCheckBox('Сохранять толщину основания'); self.sculpt_protect_base.setChecked(True)
        form.addRow('Радиус',self.sculpt_radius); form.addRow('Сила',self.sculpt_gain); form.addRow(self.sculpt_fixed)
        form.addRow(self.sculpt_protect_base)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(page)
        self.sculpt_tab=self.inspector_tabs.addTab(scroll,'Лепка')
        self.resizeDocks([self.inspector],[330],Qt.Horizontal)
        self._sculpt_timer=QTimer(self); self._sculpt_timer.setSingleShot(True)
        self._sculpt_timer.timeout.connect(self.show_mesh)
        for event,callback in [('LeftButtonPressEvent',self._sculpt_press),('MouseMoveEvent',self._sculpt_move),('LeftButtonReleaseEvent',self._sculpt_release)]:
            # Use higher priority than the camera style, which captures drag events.
            self.plotter.iren.interactor.AddObserver(event,callback,1.0)

    def toggle_sculpt(self, enabled):
        if enabled and getattr(self,'_irm_enabled',False):
            self.irm_enable_button.setChecked(False)
        if enabled and self.z is None:
            self.sculpt_action.setChecked(False)
            QMessageBox.information(self,'Лепка','Создайте плоскую основу, рельеф из фото или откройте STL.'); return
        self._sculpt_release()
        interactor=self.plotter.iren.interactor
        if enabled:
            self._camera_style=interactor.GetInteractorStyle()
            from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser
            self._sculpt_style=vtkInteractorStyleUser()
            interactor.SetInteractorStyle(self._sculpt_style)
        elif hasattr(self,'_camera_style'):
            interactor.SetInteractorStyle(self._camera_style)
        if enabled:
            self.cancel_preview(); self.tabs.setCurrentIndex(1); self.show_inspector(self.sculpt_tab)
            self.statusBar().showMessage('ЛЕПКА · нажмите на поверхность и тяните вверх / вниз. Для вращения выключите «Лепка в 3D».')
        else:
            self.statusBar().showMessage('Просмотр 3D: вращение мышью, колесо — масштаб.')

    def _sculpt_press(self, *_):
        if not self.sculpt_action.isChecked() or self.z is None: return
        from vtkmodules.vtkRenderingCore import vtkCellPicker
        x,y=self.plotter.iren.interactor.GetEventPosition()
        picker=vtkCellPicker(); picker.SetTolerance(.005)
        picker.PickFromListOn(); picker.AddPickList(self.actor)
        if not picker.Pick(x,y,0,self.plotter.renderer): return
        px,py,_=picker.GetPickPosition(); cx,cy=self.cell()
        center=((px+self.sp_w.value()/2)/cx,(py+self.sp_h.value()/2)/cy)
        mask=None
        if self._closed_polys(): mask=self.region_cov()>.5
        weight=sculpt_weight(self.z.shape,center,(cx,cy),self.sculpt_radius.value(),mask,self.sculpt_fixed.isChecked())
        self._stroke=dict(y=y,weight=weight,base=self.z,snapshot=self.snapshot(),changed=False)

    def _sculpt_move(self, *_):
        if not self._stroke: return
        _,y=self.plotter.iren.interactor.GetEventPosition()
        delta=(y-self._stroke['y'])*self.sculpt_gain.value()
        base, weight = self._stroke['base'], self._stroke['weight']
        floor = self.sp_base.value() if self.sculpt_protect_base.isChecked() else .05
        # Preserve existing depressions and all points outside the brush. Protection
        # stops downward motion at the base without raising untouched geometry.
        floor = np.minimum(base, floor)
        self.z=np.maximum(base+delta*weight,floor)
        self._stroke['changed']=not np.array_equal(self.z,base)
        if not self._sculpt_timer.isActive(): self._sculpt_timer.start(60)

    def _sculpt_release(self, *_):
        if not self._stroke: return
        stroke,self._stroke=self._stroke,None
        self._sculpt_timer.stop()
        if stroke['changed']:
            self.push_history('Лепка поверхности',stroke['snapshot']); self.after_change()
        self.show_mesh()

    def closeEvent(self,event):
        if getattr(self,'_closed',False):
            event.accept(); return
        self._sculpt_release()
        if not self.confirm_discard(): event.ignore(); return
        self.settings.setValue('geometry',self.saveGeometry()); self.settings.setValue('dockState',self.saveState())
        for key,dialog in [('sweep',self.sweep_dialog),('shape',self.relief_dialog)]:
            self.settings.setValue(key+'/geometry',dialog.saveGeometry())
        self.settings.setValue('parameters',json.dumps(self._parameters(),ensure_ascii=False)); self.settings.sync()
        self.sh_timer.stop(); self._sculpt_timer.stop()
        self.sweep_dialog.hide(); self.relief_dialog.hide()
        self._closed=True
        core.shutil.rmtree(self.tmpdir,ignore_errors=True)
        if self.book_dir: core.shutil.rmtree(self.book_dir,ignore_errors=True)
        self.plotter.close(); event.accept()


def main():
    app=QApplication(sys.argv); app.setStyle('Fusion')
    app.setApplicationName('Stone Relief Studio'); app.setOrganizationName('StoneReliefStudio')
    window=StoneStudio(); window.show()
    arg=next((p for p in sys.argv[1:] if p.lower().endswith(core.EXT)),None)
    if arg: QTimer.singleShot(0,lambda:window.open_project(arg))
    sys.exit(app.exec())


if __name__=='__main__':
    main()
