import os
import sys
import numpy as np

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtGui import QKeySequence

import pyvista as pv
from pyvistaqt import QtInteractor

from vtkmodules.vtkRenderingCore import vtkCellPicker
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget


# ---------------------------
# Harja-apurit
# ---------------------------
def gaussian_falloff(r: np.ndarray, radius: float) -> np.ndarray:
    radius = max(float(radius), 1e-9)
    sigma = radius / 3.0
    w = np.exp(-(r * r) / (2.0 * sigma * sigma))
    w[r > radius] = 0.0
    return w


def safe_unit_axis(mode: str) -> np.ndarray | None:
    if mode == "X":
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if mode == "Y":
        return np.array([0.0, 1.0, 0.0], dtype=float)
    if mode == "Z":
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return None


def grab_deform(points: np.ndarray, anchor: np.ndarray, delta: np.ndarray, radius: float, strength: float) -> np.ndarray:
    d = points - anchor
    r = np.linalg.norm(d, axis=1)
    w = gaussian_falloff(r, radius) * float(strength)
    return points + (w[:, None] * delta[None, :])


def inflate_deflate(points: np.ndarray, anchor: np.ndarray, normals: np.ndarray, radius: float, amount: float) -> np.ndarray:
    d = points - anchor
    r = np.linalg.norm(d, axis=1)
    w = gaussian_falloff(r, radius) * float(amount)
    return points + normals * w[:, None]


def local_smooth(points: np.ndarray, anchor: np.ndarray, radius: float, strength: float, drag_len: float) -> np.ndarray:
    pts = points
    r = np.linalg.norm(pts - anchor, axis=1)
    idxs = np.where(r <= radius)[0]
    if idxs.size == 0:
        return pts

    nb_r = max(radius * 0.40, 0.8)
    amt = float(np.clip(strength * drag_len * 6.0, 0.0, 0.65))

    out = pts.copy()
    for vi in idxs:
        rr = np.linalg.norm(pts - pts[vi], axis=1)
        nb = np.where(rr <= nb_r)[0]
        if nb.size < 8:
            continue
        avg = pts[nb].mean(axis=0)
        out[vi] = (1 - amt) * pts[vi] + amt * avg
    return out


# ---------------------------
# UI-teema (harmaa)
# ---------------------------
def apply_grey_ui_theme(app: QtWidgets.QApplication, main: QtWidgets.QMainWindow):
    app.setStyle("Fusion")
    f = app.font()
    f.setPointSize(9)
    app.setFont(f)

    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#d7d9dd"))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#111827"))
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#f3f4f6"))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#e5e7eb"))
    pal.setColor(QtGui.QPalette.Text, QtGui.QColor("#111827"))
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor("#e5e7eb"))
    pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#111827"))
    pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#2563eb"))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    app.setPalette(pal)

    main.setStyleSheet("""
        QMainWindow { background: #d7d9dd; }

        QMenuBar { background: #cfd2d8; border-bottom: 1px solid #b7bcc6; }
        QMenuBar::item { padding: 5px 8px; font-weight: 600; }
        QMenuBar::item:selected { background: #e5e7eb; border-radius: 6px; }
        QMenu { background: #f3f4f6; border: 1px solid #b7bcc6; }
        QMenu::item { padding: 6px 12px; font-weight: 500; }
        QMenu::item:selected { background: #e5e7eb; }

        QWidget#PaletteRoot {
            background: #f3f4f6;
            border: 1px solid #b7bcc6;
            border-radius: 10px;
        }
        QLabel#PaletteTitle {
            background: #cfd2d8;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            padding: 6px 10px;
            font-weight: 800;
            color: #111827;
        }
        QLabel#StatusLabel {
            background: #ffffff;
            border: 1px solid #b7bcc6;
            border-radius: 8px;
            padding: 8px;
            color: #111827;
            font-weight: 600;
        }
        QLabel#SubLabel { color: #374151; font-weight: 700; padding-left: 2px; }

        QGroupBox {
            border: 1px solid #b7bcc6;
            border-radius: 8px;
            margin-top: 9px;
            padding: 8px;
            background: #f3f4f6;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px 0 6px;
            font-weight: 700;
            color: #111827;
        }

        QPushButton {
            background: #e5e7eb;
            border: 1px solid #b7bcc6;
            border-radius: 8px;
            padding: 5px 7px;
            font-weight: 600;
            color: #111827;
        }
        QPushButton:hover { background: #eef0f3; }
        QPushButton:pressed { background: #dfe3ea; }

        QToolButton#MiniBtn {
            background: #e5e7eb;
            border: 1px solid #b7bcc6;
            border-radius: 8px;
            padding: 4px 6px;
            font-weight: 600;
            color: #111827;
        }
        QToolButton#MiniBtn:hover { background: #eef0f3; }
        QToolButton#MiniBtn:checked {
            background: #bfc6d2;
            border: 2px solid #111827;
            padding: 3px 5px;
        }

        QToolButton#AxisMiniBtn {
            background: #e5e7eb;
            border: 1px solid #b7bcc6;
            border-radius: 8px;
            padding: 2px 5px;
            font-weight: 600;
            color: #111827;
        }
        QToolButton#AxisMiniBtn:hover { background: #eef0f3; }
        QToolButton#AxisMiniBtn:checked {
            background: #bfc6d2;
            border: 2px solid #111827;
            padding: 1px 4px;
        }
        QToolButton#AxisMiniBtn:disabled {
            color: #6b7280;
            background: #eceff3;
            border: 1px dashed #b7bcc6;
        }

        QSlider::groove:horizontal { height: 6px; border-radius: 3px; background: #c7ccd6; }
        QSlider::handle:horizontal { width: 14px; margin: -5px 0; border-radius: 7px; background: #ffffff; border: 1px solid #b7bcc6; }
        QSpinBox { padding: 3px 6px; border-radius: 6px; border: 1px solid #b7bcc6; background: #ffffff; }

        QStatusBar { background: #cfd2d8; border-top: 1px solid #b7bcc6; color: #111827; }
    """)


# ---------------------------
# Leijuva työkalupaletti (EI always-on-top)
# ---------------------------
class ToolPalette(QtWidgets.QDialog):
    def __init__(self, owner: QtWidgets.QWidget, title: str, content: QtWidgets.QWidget):
        super().__init__(owner)
        self.setWindowTitle(title)

        # EI WindowStaysOnTopHint -> ei jää muiden ohjelmien päälle
        self.setWindowFlags(
            QtCore.Qt.Tool |
            QtCore.Qt.CustomizeWindowHint |
            QtCore.Qt.WindowTitleHint |
            QtCore.Qt.WindowCloseButtonHint
        )

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.setSizeGripEnabled(False)

        root = QtWidgets.QWidget()
        root.setObjectName("PaletteRoot")
        lay = QtWidgets.QVBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setObjectName("PaletteTitle")
        lay.addWidget(title_lbl)

        inner = QtWidgets.QWidget()
        inner_lay = QtWidgets.QVBoxLayout(inner)
        inner_lay.setContentsMargins(10, 10, 10, 10)
        inner_lay.setSpacing(8)
        inner_lay.addWidget(content)
        inner_lay.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        lay.addWidget(inner)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(root)

        # kasvaa juuri sopivaksi + lukitaan
        self.adjustSize()
        self.setFixedSize(self.sizeHint())


class _VisWatcher(QtCore.QObject):
    def __init__(self, cb):
        super().__init__()
        self.cb = cb

    def eventFilter(self, obj, event):
        if event.type() in (QtCore.QEvent.Show, QtCore.QEvent.Hide, QtCore.QEvent.Close):
            QtCore.QTimer.singleShot(0, self.cb)
        return False


# ---------------------------
# Ctrl+Z/Y/O/S “hard override” (VTK fokusongelmaan)
# ---------------------------
class ShortcutEater(QtCore.QObject):
    def __init__(self, editor: "STLEditor"):
        super().__init__(editor)
        self.ed = editor

    def _match(self, ev: QtGui.QKeyEvent, key: QtCore.Qt.Key, ctrl: bool = False) -> bool:
        if ev.key() != key:
            return False
        mods = ev.modifiers()
        need = QtCore.Qt.NoModifier
        if ctrl:
            need |= QtCore.Qt.ControlModifier
        return mods == need

    def eventFilter(self, obj, event):
        et = event.type()
        if et in (QtCore.QEvent.ShortcutOverride, QtCore.QEvent.KeyPress):
            if isinstance(event, QtGui.QKeyEvent):
                ev = event
                if self._match(ev, QtCore.Qt.Key_Z, ctrl=True):
                    self.ed.do_undo()
                    ev.accept()
                    return True
                if self._match(ev, QtCore.Qt.Key_Y, ctrl=True):
                    self.ed.do_redo()
                    ev.accept()
                    return True
                if self._match(ev, QtCore.Qt.Key_O, ctrl=True):
                    self.ed.open_dialog()
                    ev.accept()
                    return True
                if self._match(ev, QtCore.Qt.Key_S, ctrl=True):
                    self.ed.save_as_dialog()
                    ev.accept()
                    return True
        return False


# ---------------------------
# Pääohjelma
# ---------------------------
class STLEditor(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STL Sculpt/CAD")
        self._auto_size_to_screen()

        self.mesh: pv.PolyData | None = None
        self.points: np.ndarray | None = None
        self.file_path: str | None = None
        self.mesh_actor = None

        self.display_mode = "CAD"

        self.brush = "TARTU"
        self.axis_mode = "VAPAA"
        self.axis: np.ndarray | None = None
        self.radius = 10.0
        self.strength = 1.0
        self.smooth_amount = 0.50

        self.anchor: np.ndarray | None = None
        self.anchor_actor_name = "ANKKURI"

        self.hud_actor_name = "HUD_TEXT"
        self.ruler_actor_name = "RULER_LINE"
        self._sculpt_total = 0.0
        self._axis_total_signed = 0.0

        self.sculpting = False
        self.last_pick: np.ndarray | None = None
        self._lmb_down = False
        self._press_pos = None
        self._moved = False
        self._click_move_threshold_px = 6
        self._drag_consumed_by_sculpt = False

        self._rmb_down = False
        self._rmb_last = None

        self.undo: list[np.ndarray] = []
        self.redo: list[np.ndarray] = []
        self.stack_max = 30

        self.picker = vtkCellPicker()
        self.picker.SetTolerance(0.002)

        self._axes_actor = vtkAxesActor()
        self._orient_widget = None

        self._sculpt_timer = QtCore.QElapsedTimer()
        self._sculpt_timer.start()
        self._sculpt_min_dt_ms = 16

        self._normals_timer = QtCore.QElapsedTimer()
        self._normals_timer.start()
        self._normals_min_dt_ms = 80
        self._cached_normals: np.ndarray | None = None

        self._max_step_world = 2.0
        self._max_amount_world = 2.0

        # Värit (CAD: vaalea kappale + musta verkko + vaalea harmaa tausta)
        self.viewport_bg = "#e2e4e8"
        self.mesh_color_cad = "#f0f1f3"
        self.edge_color_cad = "#000000"
        self.mesh_color_print = "#9aa0a8"
        self.anchor_color = "#f59e0b"
        self.hud_color = "black"
        self.ruler_color = "black"

        apply_grey_ui_theme(QtWidgets.QApplication.instance(), self)

        self._build_actions()
        self._build_menu()
        self._build_layout()
        self._install_axes_gizmo()
        self._bind_shortcuts()

        self._shortcut_eater = ShortcutEater(self)
        QtWidgets.QApplication.instance().installEventFilter(self._shortcut_eater)

        self._set_status(
            "Avaa STL (Ctrl+O). Ankkuri: klikkaus pintaan. Muovaa: SHIFT+vedä. "
            "Orbit: vasen vedä. Pan: oikea vedä. Zoom: rulla."
        )
        self._update_hud("", clear=True)

        QtCore.QTimer.singleShot(0, self._place_palettes_default)

    def _auto_size_to_screen(self):
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        w = int(screen.width() * 0.92)
        h = int(screen.height() * 0.92)
        self.resize(max(1100, w), max(740, h))
        self.setMinimumSize(980, 680)

    # ---------------------------
    # actions/menu
    # ---------------------------
    def _build_actions(self):
        style = self.style()

        self.act_open = QtGui.QAction(style.standardIcon(QtWidgets.QStyle.SP_DialogOpenButton), "Avaa…", self)
        self.act_open.setShortcut(QKeySequence("Ctrl+O"))
        self.act_open.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        self.act_open.triggered.connect(self.open_dialog)

        self.act_save = QtGui.QAction(style.standardIcon(QtWidgets.QStyle.SP_DialogSaveButton), "Tallenna nimellä…", self)
        self.act_save.setShortcut(QKeySequence("Ctrl+S"))
        self.act_save.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        self.act_save.triggered.connect(self.save_as_dialog)

        self.act_undo = QtGui.QAction(style.standardIcon(QtWidgets.QStyle.SP_ArrowBack), "Kumoa", self)
        self.act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        self.act_undo.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        self.act_undo.triggered.connect(self.do_undo)

        self.act_redo = QtGui.QAction(style.standardIcon(QtWidgets.QStyle.SP_ArrowForward), "Tee uudelleen", self)
        self.act_redo.setShortcut(QKeySequence("Ctrl+Y"))
        self.act_redo.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        self.act_redo.triggered.connect(self.do_redo)

        self.act_shortcuts = QtGui.QAction("Pikanäppäimet…", self)
        self.act_shortcuts.triggered.connect(self.show_shortcuts)

        self.act_view_cad = QtGui.QAction("CAD (verkko)", self, checkable=True)
        self.act_view_print = QtGui.QAction("Tulostettu (matta)", self, checkable=True)
        self.act_view_cad.setChecked(True)
        self.act_view_cad.triggered.connect(lambda: self.set_display_mode("CAD"))
        self.act_view_print.triggered.connect(lambda: self.set_display_mode("PRINT"))

    def _build_menu(self):
        menu = self.menuBar()

        fm = menu.addMenu("Tiedosto")
        fm.addAction(self.act_open)
        fm.addAction(self.act_save)
        fm.addSeparator()
        fm.addAction("Lopeta", self.close)

        em = menu.addMenu("Muokkaa")
        em.addAction(self.act_undo)
        em.addAction(self.act_redo)

        vm = menu.addMenu("Näyttö")
        view_group = QtGui.QActionGroup(self)
        view_group.setExclusive(True)
        view_group.addAction(self.act_view_cad)
        view_group.addAction(self.act_view_print)
        vm.addAction(self.act_view_cad)
        vm.addAction(self.act_view_print)

        self.win_menu = menu.addMenu("Ikkunat")

        hm = menu.addMenu("Ohje")
        hm.addAction(self.act_shortcuts)

    # ---------------------------
    # layout/paletit
    # ---------------------------
    def _build_layout(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        self.view = QtInteractor(central)
        root.addWidget(self.view.interactor, 1)
        self.view.set_background(self.viewport_bg)
        self.view.interactor.installEventFilter(self)

        try:
            self.view.enable_trackball_style()
        except Exception:
            pass

        self._setup_studio_lighting()

        self.palette_status = ToolPalette(self, "Status", self._build_status_panel())
        self.palette_sculpt = ToolPalette(self, "Muovailu", self._section_muovailu_compact())
        self.palette_mesh = ToolPalette(self, "Verkko", self._section_verkko_compact())
        self.palette_camera = ToolPalette(self, "Kamera", self._section_kamera_compact_small_text())

        self._rebuild_windows_menu()

    def _place_palettes_default(self):
        g = self.geometry()
        x0 = g.x() + 18
        y0 = g.y() + 58
        gap = 10

        self.palette_status.move(x0, y0)
        self.palette_status.show()

        self.palette_sculpt.move(x0, y0 + self.palette_status.height() + gap)
        self.palette_sculpt.show()

        self.palette_mesh.move(x0, y0 + self.palette_status.height() + gap + self.palette_sculpt.height() + gap)
        self.palette_mesh.show()

        cam_x = g.x() + g.width() - self.palette_camera.width() - 22
        cam_y = y0
        self.palette_camera.move(cam_x, cam_y)
        self.palette_camera.show()

    def _rebuild_windows_menu(self):
        self.win_menu.clear()

        def add_toggle(title: str, dlg: QtWidgets.QDialog):
            act = QtGui.QAction(title, self, checkable=True)
            act.setChecked(dlg.isVisible())
            act.toggled.connect(lambda on: dlg.setVisible(on))

            def sync():
                act.blockSignals(True)
                act.setChecked(dlg.isVisible())
                act.blockSignals(False)

            watcher = _VisWatcher(sync)
            dlg.installEventFilter(watcher)
            if not hasattr(self, "_vis_watchers"):
                self._vis_watchers = []
            self._vis_watchers.append(watcher)

            self.win_menu.addAction(act)

        add_toggle("Status", self.palette_status)
        add_toggle("Muovailu", self.palette_sculpt)
        add_toggle("Verkko", self.palette_mesh)
        add_toggle("Kamera", self.palette_camera)

        self.win_menu.addSeparator()
        act_reset = QtGui.QAction("Palauta palettien sijainti", self)
        act_reset.triggered.connect(self._place_palettes_default)
        self.win_menu.addAction(act_reset)

    # ---------------------------
    # mini toolbuttons
    # ---------------------------
    def _mini_toolbtn(self, text: str) -> QtWidgets.QToolButton:
        b = QtWidgets.QToolButton()
        b.setText(text)
        b.setCheckable(True)
        b.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        b.setObjectName("MiniBtn")
        b.setMinimumHeight(26)
        return b

    def _axis_toolbtn(self, text: str) -> QtWidgets.QToolButton:
        b = QtWidgets.QToolButton()
        b.setText(text)
        b.setCheckable(True)
        b.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        b.setObjectName("AxisMiniBtn")
        f = b.font()
        f.setPointSize(8)
        b.setFont(f)
        b.setMinimumHeight(22)
        b.setToolTip("Akseli koskee vain Tartu-harjaa (SHIFT+vedä).")
        return b

    # ---------------------------
    # status panel
    # ---------------------------
    def _build_status_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.status_lbl = QtWidgets.QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setObjectName("StatusLabel")
        lay.addWidget(self.status_lbl)

        gb = QtWidgets.QGroupBox("Näyttö")
        gl = QtWidgets.QHBoxLayout(gb)
        gl.setContentsMargins(10, 10, 10, 10)
        gl.setSpacing(10)

        self.rb_cad = QtWidgets.QRadioButton("CAD")
        self.rb_print = QtWidgets.QRadioButton("Tulostettu")
        self.rb_cad.setChecked(True)

        self.rb_cad.toggled.connect(lambda on: on and self.set_display_mode("CAD"))
        self.rb_print.toggled.connect(lambda on: on and self.set_display_mode("PRINT"))

        gl.addWidget(self.rb_cad)
        gl.addWidget(self.rb_print)
        lay.addWidget(gb)
        return w

    # ---------------------------
    # sections
    # ---------------------------
    def _section_muovailu_compact(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        gb = QtWidgets.QGroupBox("Muovailu")
        gl = QtWidgets.QVBoxLayout(gb)
        gl.setSpacing(6)

        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(6)
        self.btn_br_grab = self._mini_toolbtn("Tartu")
        self.btn_br_infl = self._mini_toolbtn("Pullista")
        self.btn_br_defl = self._mini_toolbtn("Paina")
        self.btn_br_smoo = self._mini_toolbtn("Silota")
        row1.addWidget(self.btn_br_grab)
        row1.addWidget(self.btn_br_infl)
        row1.addWidget(self.btn_br_defl)
        row1.addWidget(self.btn_br_smoo)
        gl.addLayout(row1)

        self.grp_brush = QtWidgets.QButtonGroup(self)
        self.grp_brush.setExclusive(True)
        for b in (self.btn_br_grab, self.btn_br_infl, self.btn_br_defl, self.btn_br_smoo):
            self.grp_brush.addButton(b)

        self.btn_br_grab.clicked.connect(lambda: self.set_brush("TARTU"))
        self.btn_br_infl.clicked.connect(lambda: self.set_brush("PULLISTA"))
        self.btn_br_defl.clicked.connect(lambda: self.set_brush("PAINA"))
        self.btn_br_smoo.clicked.connect(lambda: self.set_brush("SILOTA"))
        self.btn_br_grab.setChecked(True)

        sub = QtWidgets.QLabel("Akseli / viivain (vain Tartu-harja)")
        sub.setObjectName("SubLabel")
        gl.addWidget(sub)

        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(6)
        self.btn_ax_free = self._axis_toolbtn("Vapaa")
        self.btn_ax_x = self._axis_toolbtn("X")
        self.btn_ax_y = self._axis_toolbtn("Y")
        self.btn_ax_z = self._axis_toolbtn("Z")
        row2.addWidget(self.btn_ax_free, 2)
        row2.addWidget(self.btn_ax_x, 1)
        row2.addWidget(self.btn_ax_y, 1)
        row2.addWidget(self.btn_ax_z, 1)
        gl.addLayout(row2)

        self.grp_axis = QtWidgets.QButtonGroup(self)
        self.grp_axis.setExclusive(True)
        for b in (self.btn_ax_free, self.btn_ax_x, self.btn_ax_y, self.btn_ax_z):
            self.grp_axis.addButton(b)

        self.btn_ax_free.clicked.connect(lambda: self.set_axis_mode("VAPAA"))
        self.btn_ax_x.clicked.connect(lambda: self.set_axis_mode("X"))
        self.btn_ax_y.clicked.connect(lambda: self.set_axis_mode("Y"))
        self.btn_ax_z.clicked.connect(lambda: self.set_axis_mode("Z"))
        self.btn_ax_free.setChecked(True)

        self.radius_lbl = QtWidgets.QLabel(f"Säde: {self.radius:.1f} mm")
        self.radius_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.radius_sl.setRange(1, 600)
        self.radius_sl.setValue(int(self.radius))
        self.radius_sl.valueChanged.connect(self._on_radius)

        self.str_lbl = QtWidgets.QLabel(f"Voima: {self.strength:.2f}")
        self.str_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.str_sl.setRange(1, 250)
        self.str_sl.setValue(int(min(self.strength, 2.50) * 100))
        self.str_sl.valueChanged.connect(self._on_strength)

        self.sm_lbl = QtWidgets.QLabel(f"Silotus: {self.smooth_amount:.2f}")
        self.sm_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sm_sl.setRange(5, 100)
        self.sm_sl.setValue(int(self.smooth_amount * 100))
        self.sm_sl.valueChanged.connect(self._on_smooth)

        gl.addWidget(self.radius_lbl)
        gl.addWidget(self.radius_sl)
        gl.addWidget(self.str_lbl)
        gl.addWidget(self.str_sl)
        gl.addWidget(self.sm_lbl)
        gl.addWidget(self.sm_sl)

        layout.addWidget(gb)

        self._sync_axis_controls_enabled()
        return w

    def _section_verkko_compact(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        gb = QtWidgets.QGroupBox("Verkko")
        vl = QtWidgets.QVBoxLayout(gb)
        vl.setSpacing(6)

        row = QtWidgets.QHBoxLayout()
        self.sub_iter = QtWidgets.QSpinBox()
        self.sub_iter.setRange(1, 3)
        self.sub_iter.setValue(1)
        row.addWidget(QtWidgets.QLabel("Tihennys:"))
        row.addWidget(self.sub_iter)
        row.addStretch(1)
        vl.addLayout(row)

        self.btn_subdivide = QtWidgets.QPushButton("Tihennä (Loop)")
        self.btn_subdivide.setMinimumHeight(26)
        self.btn_subdivide.clicked.connect(self.subdivide_mesh)

        self.dec_lbl = QtWidgets.QLabel("Kevennys: 50 %")
        self.dec_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.dec_sl.setRange(5, 90)
        self.dec_sl.setValue(50)
        self.dec_sl.valueChanged.connect(self._on_decimate)

        self.btn_decimate = QtWidgets.QPushButton("Kevennä")
        self.btn_decimate.setMinimumHeight(26)
        self.btn_decimate.clicked.connect(self.decimate_mesh)

        vl.addWidget(self.btn_subdivide)
        vl.addWidget(self.dec_lbl)
        vl.addWidget(self.dec_sl)
        vl.addWidget(self.btn_decimate)

        layout.addWidget(gb)
        return w

    def _section_kamera_compact_small_text(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        gb = QtWidgets.QGroupBox("Kamera / Suunnat")
        v = QtWidgets.QVBoxLayout(gb)
        v.setSpacing(6)

        btn_font = QtGui.QFont()
        btn_font.setPointSize(8)
        btn_font.setWeight(QtGui.QFont.Medium)

        def mk(text: str, fn):
            b = QtWidgets.QPushButton(text)
            b.setMinimumHeight(22)
            b.setFont(btn_font)
            b.clicked.connect(fn)
            return b

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)

        grid.addWidget(mk("Sovita", self.camera_fit), 0, 0)
        grid.addWidget(mk("Iso", self.camera_iso), 0, 1)

        grid.addWidget(mk("Etu", lambda: self.camera_axis_view("front")), 1, 0)
        grid.addWidget(mk("Taka", lambda: self.camera_axis_view("back")), 1, 1)
        grid.addWidget(mk("Vasen", lambda: self.camera_axis_view("left")), 2, 0)
        grid.addWidget(mk("Oikea", lambda: self.camera_axis_view("right")), 2, 1)
        grid.addWidget(mk("Ylä", lambda: self.camera_axis_view("top")), 3, 0)
        grid.addWidget(mk("Ala", lambda: self.camera_axis_view("bottom")), 3, 1)

        v.addLayout(grid)

        gb2 = QtWidgets.QGroupBox("Kulmat")
        v2 = QtWidgets.QGridLayout(gb2)
        v2.setHorizontalSpacing(6)
        v2.setVerticalSpacing(6)

        v2.addWidget(mk("NW", lambda: self.camera_corner_view("NW")), 0, 0)
        v2.addWidget(mk("NE", lambda: self.camera_corner_view("NE")), 0, 1)
        v2.addWidget(mk("SW", lambda: self.camera_corner_view("SW")), 1, 0)
        v2.addWidget(mk("SE", lambda: self.camera_corner_view("SE")), 1, 1)

        v2.addWidget(mk("NW-D", lambda: self.camera_corner_view("NW-D")), 2, 0)
        v2.addWidget(mk("NE-D", lambda: self.camera_corner_view("NE-D")), 2, 1)
        v2.addWidget(mk("SW-D", lambda: self.camera_corner_view("SW-D")), 3, 0)
        v2.addWidget(mk("SE-D", lambda: self.camera_corner_view("SE-D")), 3, 1)

        v.addWidget(gb2)

        layout.addWidget(gb)
        return w

    # ---------------------------
    # shortcuts
    # ---------------------------
    def _bind_shortcuts(self):
        def sc(seq: str, fn):
            s = QtGui.QShortcut(QKeySequence(seq), self)
            s.setContext(QtCore.Qt.ApplicationShortcut)
            s.activated.connect(fn)
            return s

        self._s_undo = sc("Ctrl+Z", self.do_undo)
        self._s_redo = sc("Ctrl+Y", self.do_redo)
        self._s_open = sc("Ctrl+O", self.open_dialog)
        self._s_save = sc("Ctrl+S", self.save_as_dialog)

    def show_shortcuts(self):
        QtWidgets.QMessageBox.information(
            self,
            "Pikanäppäimet",
            "Ctrl+O  = Avaa\n"
            "Ctrl+S  = Tallenna nimellä\n"
            "Ctrl+Z  = Kumoa\n"
            "Ctrl+Y  = Tee uudelleen\n\n"
            "Orbit: vasen hiiri + vedä\n"
            "Pan: oikea hiiri + vedä\n"
            "Ankkuri: lyhyt klikkaus pintaan\n"
            "Muovailu: SHIFT + vedä"
        )

    # ---------------------------
    # status/hud
    # ---------------------------
    def _set_status(self, text: str):
        self.status_lbl.setText(text)
        self.statusBar().showMessage(text)

    def _update_hud(self, text: str, clear: bool = False):
        try:
            self.view.remove_actor(self.hud_actor_name, reset_camera=False)
        except Exception:
            pass
        if clear or not text:
            self.view.render()
            return
        self.view.add_text(text, position="lower_left", font_size=10, color=self.hud_color,
                           shadow=True, name=self.hud_actor_name)
        self.view.render()

    def _update_ruler_line(self, p0: np.ndarray | None, p1: np.ndarray | None):
        try:
            self.view.remove_actor(self.ruler_actor_name, reset_camera=False)
        except Exception:
            pass
        if p0 is None or p1 is None:
            self.view.render()
            return
        line = pv.Line(p0, p1, resolution=1)
        self.view.add_mesh(line, color=self.ruler_color, line_width=3, name=self.ruler_actor_name)
        self.view.render()

    # ---------------------------
    # axis controls enabled only for GRAB
    # ---------------------------
    def _sync_axis_controls_enabled(self):
        on = (self.brush == "TARTU")
        for b in (self.btn_ax_free, self.btn_ax_x, self.btn_ax_y, self.btn_ax_z):
            b.setEnabled(on)

    # ---------------------------
    # render + lighting
    # ---------------------------
    def set_display_mode(self, mode: str):
        self.display_mode = mode
        self.act_view_cad.setChecked(mode == "CAD")
        self.act_view_print.setChecked(mode == "PRINT")
        if mode == "CAD":
            if not self.rb_cad.isChecked():
                self.rb_cad.setChecked(True)
        else:
            if not self.rb_print.isChecked():
                self.rb_print.setChecked(True)
        self._apply_render_mode()
        self._set_status(f"Näyttötila: {'CAD' if mode=='CAD' else 'Tulostettu (matta)'}")

    def _apply_render_mode(self):
        if self.mesh is None:
            return
        keep_anchor = self.anchor is not None
        try:
            self.view.remove_actor("MESH", reset_camera=False)
        except Exception:
            pass

        if self.display_mode == "CAD":
            self.mesh_actor = self.view.add_mesh(
                self.mesh, color=self.mesh_color_cad,
                smooth_shading=True, show_edges=True,
                edge_color=self.edge_color_cad,
                name="MESH",
            )
        else:
            self.mesh_actor = self.view.add_mesh(
                self.mesh, color=self.mesh_color_print,
                smooth_shading=True, show_edges=False,
                name="MESH",
            )
        self._setup_studio_lighting()
        if keep_anchor:
            self._restore_anchor_actor()
        self.view.render()

    def _setup_studio_lighting(self):
        try:
            self.view.remove_all_lights()
        except Exception:
            pass
        try:
            key = pv.Light(light_type='scene light')
            key.position = (1, 1, 1)
            key.focal_point = (0, 0, 0)
            key.intensity = 1.0

            fill = pv.Light(light_type='scene light')
            fill.position = (-1, 0.7, 0.8)
            fill.focal_point = (0, 0, 0)
            fill.intensity = 0.60

            back = pv.Light(light_type='scene light')
            back.position = (0.3, -1.0, 0.6)
            back.focal_point = (0, 0, 0)
            back.intensity = 0.40

            self.view.add_light(key)
            self.view.add_light(fill)
            self.view.add_light(back)
        except Exception:
            try:
                self.view.enable_lightkit()
            except Exception:
                pass

    # ---------------------------
    # tool state
    # ---------------------------
    def set_brush(self, name: str):
        self.brush = name
        self._sync_axis_controls_enabled()
        self._set_status(f"Harja: {name}. Ankkuri: klikkaus pintaan. Muovaa: SHIFT+vedä.")

    def set_axis_mode(self, mode: str):
        self.axis_mode = mode
        self.axis = safe_unit_axis(mode) if mode in ("X", "Y", "Z") else None
        self._set_status(f"Tartu-akseli: {mode} (vain Tartu-harja)")

    def _on_radius(self, v: int):
        self.radius = float(v)
        self.radius_lbl.setText(f"Säde: {self.radius:.1f} mm")

    def _on_strength(self, v: int):
        self.strength = float(v) / 100.0
        self.str_lbl.setText(f"Voima: {self.strength:.2f}")

    def _on_smooth(self, v: int):
        self.smooth_amount = float(v) / 100.0
        self.sm_lbl.setText(f"Silotus: {self.smooth_amount:.2f}")

    def _on_decimate(self, v: int):
        self.dec_lbl.setText(f"Kevennys: {v} %")

    # ---------------------------
    # IO
    # ---------------------------
    def open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Avaa STL", "", "STL-tiedostot (*.stl)")
        if path:
            self.load_stl(path)

    def save_as_dialog(self):
        if self.mesh is None:
            QtWidgets.QMessageBox.warning(self, "Ei mallia", "Avaa ensin STL (Tiedosto → Avaa…).")
            return
        default = os.path.splitext(self.file_path or "muokattu.stl")[0] + "_muokattu.stl"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Tallenna nimellä", default, "STL-tiedostot (*.stl)")
        if not path:
            return
        if not path.lower().endswith(".stl"):
            path += ".stl"
        self.mesh.save(path)
        QtWidgets.QMessageBox.information(self, "Tallennettu", f"Tallennettu:\n{path}")

    def load_stl(self, path: str):
        self.file_path = path
        self.mesh = pv.read(path).triangulate().clean()
        self.mesh.compute_normals(inplace=True)
        self.points = self.mesh.points.copy()

        L = float(self.mesh.length) if self.mesh is not None else 100.0
        self._max_step_world = max(0.25, 0.004 * L)
        self._max_amount_world = max(0.20, 0.003 * L)

        self.undo.clear()
        self.redo.clear()
        self.anchor = None
        self.sculpting = False
        self.last_pick = None
        self._drag_consumed_by_sculpt = False
        self._cached_normals = np.asarray(self.mesh.point_normals) if self.mesh.point_normals is not None else None

        self._sculpt_total = 0.0
        self._axis_total_signed = 0.0
        self._update_hud("", clear=True)
        self._update_ruler_line(None, None)

        self._redraw_mesh()
        self.view.reset_camera()
        self._apply_render_mode()

        self._set_status(
            f"STL ladattu. Turvarajat: max askel {self._max_step_world:.2f}/tick. "
            "Ankkuri: klikkaus. Muovaa: SHIFT+vedä."
        )

    # ---------------------------
    # mesh redraw + anchor
    # ---------------------------
    def _redraw_mesh(self, keep_anchor: bool = False):
        self.view.clear()
        self.view.set_background(self.viewport_bg)
        self.mesh_actor = self.view.add_mesh(
            self.mesh, color=self.mesh_color_cad,
            smooth_shading=True, show_edges=True,
            edge_color=self.edge_color_cad,
            name="MESH",
        )
        if keep_anchor and self.anchor is not None:
            self._restore_anchor_actor()
        self._update_hud("", clear=True)
        self._update_ruler_line(None, None)
        self.view.render()

    def _restore_anchor_actor(self):
        if self.anchor is None or self.mesh is None:
            return
        r = max(0.5, float(self.mesh.length) * 0.01)
        try:
            self.view.remove_actor(self.anchor_actor_name, reset_camera=False)
        except Exception:
            pass
        self.view.add_mesh(pv.Sphere(radius=r, center=self.anchor), color=self.anchor_color, name=self.anchor_actor_name)

    # ---------------------------
    # picker + anchor
    # ---------------------------
    def pick_world_point(self, qpos: QtCore.QPoint) -> np.ndarray | None:
        if self.mesh_actor is None or self.mesh is None:
            return None
        x = int(qpos.x())
        y = int(self.view.interactor.height() - qpos.y())
        ren = self.view.renderer
        ok = self.picker.Pick(x, y, 0, ren)
        if not ok:
            return None
        p = np.array(self.picker.GetPickPosition(), dtype=float)
        if not np.isfinite(p).all():
            return None
        return p

    def set_anchor_at_mouse(self, qpos: QtCore.QPoint):
        p = self.pick_world_point(qpos)
        if p is None:
            self._set_status("Ei osumaa malliin (klikkaa suoraan pinnalle).")
            return
        self.anchor = p
        self._restore_anchor_actor()
        self.view.render()
        self._set_status("Ankkuri asetettu. Muovaa: SHIFT+vedä.")

    # ---------------------------
    # camera helpers
    # ---------------------------
    def camera_fit(self):
        if self.mesh is None:
            return
        self.view.reset_camera()
        self.view.render()

    def camera_iso(self):
        if self.mesh is None:
            return
        self.view.view_isometric()
        self.view.reset_camera()
        self.view.render()

    def camera_axis_view(self, which: str):
        if self.mesh is None:
            return
        which = which.lower()
        if which == "front":
            self.view.view_xy(negative=False)
        elif which == "back":
            self.view.view_xy(negative=True)
        elif which == "left":
            self.view.view_yz(negative=False)
        elif which == "right":
            self.view.view_yz(negative=True)
        elif which == "top":
            self.view.view_xz(negative=False)
        elif which == "bottom":
            self.view.view_xz(negative=True)
        self.view.reset_camera()
        self.view.render()

    def _camera_set_corner(self, sx: float, sy: float, sz: float):
        if self.mesh is None:
            return

        b = self.mesh.bounds  # (xmin,xmax, ymin,ymax, zmin,zmax)
        cx = (b[0] + b[1]) * 0.5
        cy = (b[2] + b[3]) * 0.5
        cz = (b[4] + b[5]) * 0.5
        center = np.array([cx, cy, cz], dtype=float)

        dx = (b[1] - b[0])
        dy = (b[3] - b[2])
        dz = (b[5] - b[4])
        diag = float(np.linalg.norm([dx, dy, dz]))
        if diag < 1e-9:
            diag = 1.0

        dirv = np.array([sx, sy, sz], dtype=float)
        dirv = dirv / (np.linalg.norm(dirv) + 1e-12)

        dist = diag * 1.6
        pos = center + dirv * dist

        cam = self.view.camera
        cam.SetFocalPoint(*center)
        cam.SetPosition(*pos)

        if abs(dirv[2]) > 0.95:
            cam.SetViewUp(0, 1, 0)
        else:
            cam.SetViewUp(0, 0, 1)

        self.view.reset_camera_clipping_range()
        self.view.render()

    def camera_corner_view(self, which: str):
        which = which.upper().strip()

        if which == "NE":
            return self._camera_set_corner(+1, +1, +1)
        if which == "NW":
            return self._camera_set_corner(-1, +1, +1)
        if which == "SE":
            return self._camera_set_corner(+1, -1, +1)
        if which == "SW":
            return self._camera_set_corner(-1, -1, +1)

        if which == "NE-D":
            return self._camera_set_corner(+1, +1, -1)
        if which == "NW-D":
            return self._camera_set_corner(-1, +1, -1)
        if which == "SE-D":
            return self._camera_set_corner(+1, -1, -1)
        if which == "SW-D":
            return self._camera_set_corner(-1, -1, -1)

    # ---------------------------
    # undo/redo
    # ---------------------------
    def push_undo(self):
        if self.points is None:
            return
        self.undo.append(self.points.copy())
        if len(self.undo) > self.stack_max:
            self.undo.pop(0)
        self.redo.clear()

    def do_undo(self):
        if self.mesh is None or not self.undo:
            return
        self.redo.append(self.points.copy())
        pts = self.undo.pop()
        self._set_points(pts)
        self._apply_render_mode()
        self._set_status("Kumottu.")

    def do_redo(self):
        if self.mesh is None or not self.redo:
            return
        self.undo.append(self.points.copy())
        pts = self.redo.pop()
        self._set_points(pts)
        self._apply_render_mode()
        self._set_status("Tehty uudelleen.")

    # ---------------------------
    # mesh points
    # ---------------------------
    def _set_points(self, pts: np.ndarray):
        self.points = pts
        self.mesh.points = self.points
        self.mesh.compute_normals(inplace=True)
        self._cached_normals = np.asarray(self.mesh.point_normals) if self.mesh.point_normals is not None else None
        self.view.render()

    # ---------------------------
    # mesh ops
    # ---------------------------
    def _post_mesh_op_fixup(self):
        if self.mesh is None:
            return
        self.mesh = self.mesh.extract_surface().triangulate().clean()
        self.mesh.compute_normals(inplace=True)
        self.points = self.mesh.points.copy()
        self._cached_normals = np.asarray(self.mesh.point_normals) if self.mesh.point_normals is not None else None
        self._redraw_mesh(keep_anchor=True)
        self._apply_render_mode()

    def subdivide_mesh(self):
        if self.mesh is None:
            return
        it = int(self.sub_iter.value())
        self.push_undo()
        try:
            try:
                self.mesh = self.mesh.subdivide(it, subfilter="loop")
            except Exception:
                self.mesh = self.mesh.subdivide(it)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Tihennys epäonnistui", f"{repr(e)}")
            self.do_undo()
            return
        self._post_mesh_op_fixup()

    def decimate_mesh(self):
        if self.mesh is None:
            return
        pct = int(self.dec_sl.value())
        target_reduction = pct / 100.0
        self.push_undo()
        try:
            self.mesh = self.mesh.decimate_pro(target_reduction, preserve_topology=True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Kevennys epäonnistui", f"{repr(e)}")
            self.do_undo()
            return
        self._post_mesh_op_fixup()

    # ---------------------------
    # sculpting
    # ---------------------------
    def _clamp_vec(self, v: np.ndarray, max_len: float) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n <= max_len or n < 1e-12:
            return v
        return v * (max_len / n)

    def _start_sculpting(self, start_pos: QtCore.QPoint | None = None):
        if self.mesh is None or self.anchor is None:
            return False
        if self.sculpting:
            return True
        self.sculpting = True
        self.push_undo()
        self._drag_consumed_by_sculpt = True
        self.last_pick = None
        self._sculpt_total = 0.0
        self._axis_total_signed = 0.0
        if start_pos is not None:
            p0 = self.pick_world_point(start_pos)
            if p0 is not None:
                self.last_pick = p0
        self._update_ruler_line(None, None)
        return True

    def sculpt_step(self, qpos: QtCore.QPoint):
        if self.mesh is None or self.points is None or self.anchor is None:
            return
        if self._sculpt_timer.elapsed() < self._sculpt_min_dt_ms:
            return
        self._sculpt_timer.restart()

        p = self.pick_world_point(qpos)
        if p is None:
            return
        if self.last_pick is None:
            self.last_pick = p
            return

        delta_raw = p - self.last_pick
        self.last_pick = p
        delta_raw = self._clamp_vec(delta_raw, self._max_step_world)
        if float(np.linalg.norm(delta_raw)) < 1e-10:
            return

        if self.brush == "TARTU" and self.axis is not None:
            signed_step = float(delta_raw @ self.axis)
            max_grab = min(self._max_step_world, max(0.25, 0.15 * self.radius))
            signed_step = float(np.clip(signed_step, -max_grab, +max_grab))
            delta = self.axis * signed_step

            self._axis_total_signed += signed_step
            self._sculpt_total += abs(signed_step)

            eff_strength = float(np.clip(self.strength, 0.0, 2.5))
            newp = grab_deform(self.points, self.anchor, delta, self.radius, eff_strength)
            self._set_points(newp)

            p0 = self.anchor
            p1 = self.anchor + (self.axis * self._axis_total_signed)
            self._update_ruler_line(p0, p1)
            self._update_hud(f"{self.axis_mode}-siirto: Δ {signed_step:.3f} mm   Σ {self._axis_total_signed:.3f} mm", clear=False)
            return

        if self.brush == "TARTU":
            delta = self._clamp_vec(delta_raw, min(self._max_step_world, max(0.25, 0.15 * self.radius)))
            eff_strength = float(np.clip(self.strength, 0.0, 2.5))
            newp = grab_deform(self.points, self.anchor, delta, self.radius, eff_strength)
            self._set_points(newp)
            self._update_ruler_line(self.anchor, p)
            d = float(np.linalg.norm(delta))
            dx, dy, dz = delta.tolist()
            self._sculpt_total += d
            ruler_len = float(np.linalg.norm(p - self.anchor))
            self._update_hud(
                f"Δ {d:.3f} mm (dx {dx:.3f}, dy {dy:.3f}, dz {dz:.3f})  Σ {self._sculpt_total:.3f} mm | Viivain {ruler_len:.3f} mm",
                clear=False
            )
            return

        if self._cached_normals is None or self._normals_timer.elapsed() > self._normals_min_dt_ms:
            self.mesh.compute_normals(inplace=True)
            self._cached_normals = np.asarray(self.mesh.point_normals) if self.mesh.point_normals is not None else None
            self._normals_timer.restart()
        if self._cached_normals is None:
            return
        normals = self._cached_normals
        drag = float(np.linalg.norm(delta_raw))

        if self.brush in ("PULLISTA", "PAINA"):
            sign = +1.0 if self.brush == "PULLISTA" else -1.0
            eff_strength = float(np.clip(self.strength, 0.0, 2.5))
            amount = sign * eff_strength * drag * 2.0
            max_amt = min(self._max_amount_world, max(0.20, 0.12 * self.radius))
            amount = float(np.clip(amount, -max_amt, +max_amt))
            newp = inflate_deflate(self.points, self.anchor, normals, self.radius, amount)
            self._set_points(newp)

        elif self.brush == "SILOTA":
            eff_strength = float(np.clip(self.strength, 0.0, 2.5))
            eff = float(eff_strength * max(self.smooth_amount, 0.05))
            newp = local_smooth(self.points, self.anchor, self.radius, eff, drag)
            self._set_points(newp)

        self._update_ruler_line(self.anchor, p)
        ruler_len = float(np.linalg.norm(p - self.anchor))
        self._update_hud(f"Viivain: {ruler_len:.3f} mm", clear=False)

    # ---------------------------
    # pan camera (RMB)
    # ---------------------------
    def _pan_camera(self, dx_px: float, dy_px: float):
        cam = self.view.camera
        if cam is None:
            return
        pos = np.array(cam.GetPosition(), dtype=float)
        fp = np.array(cam.GetFocalPoint(), dtype=float)
        up = np.array(cam.GetViewUp(), dtype=float)

        forward = fp - pos
        dist = float(np.linalg.norm(forward))
        if dist < 1e-9:
            return
        forward = forward / dist
        up = up / (np.linalg.norm(up) + 1e-12)
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-12)

        h = max(1, int(self.view.interactor.height()))
        fov = float(cam.GetViewAngle()) * np.pi / 180.0
        world_per_px = (2.0 * dist * np.tan(fov / 2.0)) / h

        shift = (-dx_px * world_per_px) * right + (dy_px * world_per_px) * up
        cam.SetPosition(*(pos + shift))
        cam.SetFocalPoint(*(fp + shift))
        self.view.render()

    # ---------------------------
    # Event filter (VTK interactor)
    # ---------------------------
    def eventFilter(self, obj, event):
        if obj is self.view.interactor:
            et = event.type()

            if et == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.RightButton:
                self._rmb_down = True
                self._rmb_last = event.position().toPoint()
                return True

            if et == QtCore.QEvent.MouseMove and self._rmb_down and (event.buttons() & QtCore.Qt.RightButton):
                cur = event.position().toPoint()
                if self._rmb_last is not None:
                    dx = cur.x() - self._rmb_last.x()
                    dy = cur.y() - self._rmb_last.y()
                    self._pan_camera(dx, dy)
                self._rmb_last = cur
                return True

            if et == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.RightButton:
                self._rmb_down = False
                self._rmb_last = None
                return True

            if et == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                if self.mesh is None:
                    return False
                self._lmb_down = True
                self._press_pos = event.position().toPoint()
                self._moved = False
                self._drag_consumed_by_sculpt = False

                if event.modifiers() & QtCore.Qt.ShiftModifier:
                    if self.anchor is None:
                        self._set_status("Aseta ensin ankkuri: lyhyt klikkaus pintaan.")
                        return True
                    self._start_sculpting(start_pos=event.position().toPoint())
                    return True
                return False

            if et == QtCore.QEvent.MouseMove:
                if self.mesh is None:
                    return False

                if self._lmb_down and (event.buttons() & QtCore.Qt.LeftButton):
                    if (not self.sculpting) and (event.modifiers() & QtCore.Qt.ShiftModifier):
                        if self.anchor is None:
                            self._set_status("Aseta ensin ankkuri: klikkaus pintaan.")
                            return True
                        self._start_sculpting(start_pos=event.position().toPoint())
                        return True

                    if self.sculpting:
                        self.sculpt_step(event.position().toPoint())
                        return True

                    if self._press_pos is not None:
                        cur = event.position().toPoint()
                        dx = cur.x() - self._press_pos.x()
                        dy = cur.y() - self._press_pos.y()
                        if (dx * dx + dy * dy) ** 0.5 >= self._click_move_threshold_px:
                            self._moved = True

                    if self._drag_consumed_by_sculpt:
                        return True
                    return False

                return False

            if et == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                if self.sculpting:
                    self.sculpting = False
                    self.last_pick = None
                    self._drag_consumed_by_sculpt = False
                    self._update_ruler_line(None, None)
                    self._set_status("Muovailu valmis. SHIFT+vedä jatkaaksesi.")
                    self._lmb_down = False
                    return True

                if self._lmb_down:
                    self._lmb_down = False
                    if (self.mesh is not None) and (not self._moved):
                        self.set_anchor_at_mouse(event.position().toPoint())
                        self._sculpt_total = 0.0
                        self._axis_total_signed = 0.0
                        self._update_hud("", clear=True)
                        self._update_ruler_line(None, None)
                        return True
                return False

        return super().eventFilter(obj, event)

    # ---------------------------
    # gizmo
    # ---------------------------
    def _install_axes_gizmo(self):
        try:
            self._orient_widget = vtkOrientationMarkerWidget()
            self._orient_widget.SetOrientationMarker(self._axes_actor)
            self._orient_widget.SetInteractor(self.view.interactor)
            self._orient_widget.SetViewport(0.83, 0.02, 0.98, 0.17)
            self._orient_widget.SetEnabled(1)
            self._orient_widget.InteractiveOff()
        except Exception:
            self._orient_widget = None


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = STLEditor()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
