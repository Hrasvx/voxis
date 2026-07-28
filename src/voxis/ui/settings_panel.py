"""Scrollable real-time controls covering audio, points, networks, and effects."""

from __future__ import annotations

from dataclasses import asdict

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import VisualizationSettings


SECTIONS = (
    (
        "AUDIO ANALYSIS",
        (
            ("audio_sensitivity", "Sensitivity", 0.05, 8.0, 0.05, 2),
            ("frequency_min", "Minimum frequency", 10.0, 18000.0, 10.0, 0),
            ("frequency_max", "Maximum frequency", 20.0, 24000.0, 100.0, 0),
            ("fft_size", "FFT size", 512, 8192, 512, 0),
            ("hop_size", "Hop size", 64, 2048, 64, 0),
            ("audio_smoothing", "Smoothing", 0.0, 0.98, 0.02, 2),
            ("transient_sensitivity", "Transient sensitivity", 0.0, 5.0, 0.05, 2),
            ("min_peaks_per_frame", "Minimum peaks / frame", 0, 20, 1, 0),
            ("max_peaks_per_frame", "Maximum peaks / frame", 1, 40, 1, 0),
            ("peak_prominence_db", "Peak prominence", 0.5, 30.0, 0.5, 1),
            ("spectral_noise_floor_db", "Spectral noise floor", 0.0, 40.0, 0.5, 1),
            ("silence_threshold_db", "Silence threshold", -100.0, -20.0, 1.0, 0),
            ("phrase_detection_sensitivity", "Phrase sensitivity", 0.1, 4.0, 0.05, 2),
            ("low_frequency_influence", "Low influence", 0.0, 3.0, 0.05, 2),
            ("mid_frequency_influence", "Mid influence", 0.0, 3.0, 0.05, 2),
            ("high_frequency_influence", "High influence", 0.0, 3.0, 0.05, 2),
        ),
    ),
    (
        "POINTS",
        (
            ("point_size", "Point size", 0.25, 3.0, 0.05, 2),
            ("point_core_size", "Core pixels", 1.0, 6.0, 0.1, 1),
            ("white_core_intensity", "White-core intensity", 0.0, 2.5, 0.05, 2),
            ("new_node_size_boost", "New-node size boost", 0.0, 2.0, 0.05, 2),
            ("new_node_glow_boost", "New-node glow boost", 0.0, 3.0, 0.05, 2),
            ("node_settle_duration", "Node settle time", 0.1, 5.0, 0.1, 2),
            ("visible_point_limit", "Visible point limit", 256, 25000, 128, 0),
            ("active_duration", "Active duration", 0.25, 8.0, 0.25, 2),
            ("historical_duration", "Historical duration", 1.0, 60.0, 0.5, 1),
            ("historical_opacity", "Historical opacity", 0.0, 0.8, 0.02, 2),
            ("maximum_retained_phrases", "Retained phrases", 1, 128, 1, 0),
            ("node_interval_s", "Node interval", 0.02, 1.0, 0.01, 2),
            ("fade_speed", "Fade speed", 0.1, 5.0, 0.1, 2),
            ("spatial_spread", "Spatial spread", 0.2, 4.0, 0.05, 2),
            ("horizontal_spacing", "Horizontal spacing", 0.25, 8.0, 0.05, 2),
            ("frequency_spacing", "Frequency spacing", 0.25, 8.0, 0.05, 2),
            ("point_brightness", "Brightness", 0.1, 3.0, 0.05, 2),
        ),
    ),
    (
        "CONNECTIONS",
        (
            ("connection_radius", "Connection radius", 0.0, 1.6, 0.05, 2),
            ("max_connections_per_point", "Max lines per point", 0, 10, 1, 0),
            ("max_temporal_connections", "Temporal maximum", 0, 1, 1, 0),
            ("max_intra_connections", "Intra-frame maximum", 0, 4, 1, 0),
            ("max_similarity_connections", "Similarity maximum", 0, 4, 1, 0),
            ("visible_line_limit", "Visible line limit", 256, 100000, 256, 0),
            ("edge_lifetime", "Edge lifetime", 0.25, 60.0, 0.5, 1),
            ("line_thickness", "Line thickness", 0.5, 3.0, 0.1, 2),
            ("line_brightness", "Line brightness", 0.0, 2.0, 0.05, 2),
            ("connection_opacity", "Connection opacity", 0.0, 1.0, 0.02, 2),
            ("temporal_edge_strength", "Temporal strength", 0.0, 1.0, 0.02, 2),
            ("intra_frame_edge_strength", "Intra-frame strength", 0.0, 1.0, 0.02, 2),
            ("similarity_edge_strength", "Similarity strength", 0.0, 1.0, 0.02, 2),
            (
                "frequency_similarity_influence",
                "Frequency similarity",
                0.0,
                1.0,
                0.02,
                2,
            ),
        ),
    ),
    (
        "MOTION & CAMERA",
        (
            ("rotation_speed_x", "X rotation", -1.5, 1.5, 0.01, 2),
            ("rotation_speed_y", "Y rotation", -1.5, 1.5, 0.01, 2),
            ("rotation_speed_z", "Z rotation", -1.5, 1.5, 0.01, 2),
            ("camera_distance", "Camera distance", 3.0, 30.0, 0.25, 2),
            ("camera_min_distance", "Minimum distance", 1.5, 20.0, 0.25, 2),
            ("camera_max_distance", "Maximum distance", 3.0, 60.0, 0.5, 1),
            ("field_of_view", "Field of view", 25.0, 90.0, 1.0, 0),
            ("camera_drift", "Camera drift", 0.0, 1.0, 0.02, 2),
            ("fog_intensity", "Fog strength", 0.0, 1.0, 0.02, 2),
            ("viewport_occupancy", "Viewport occupancy", 0.50, 1.60, 0.01, 2),
            ("camera_smoothing", "Camera smoothing", 0.01, 1.0, 0.01, 2),
            ("camera_follow_strength", "New-node follow strength", 0.0, 1.0, 0.02, 2),
            ("camera_follow_smoothing", "Follow responsiveness", 0.01, 1.0, 0.01, 2),
            ("camera_follow_max_speed", "Maximum follow speed", 0.1, 30.0, 0.25, 2),
            ("camera_focus_hold_duration", "Focus hold time", 0.0, 5.0, 0.05, 2),
            ("camera_return_duration", "Return-to-center time", 0.1, 10.0, 0.1, 2),
            ("grid_spacing", "Grid spacing", 0.25, 5.0, 0.25, 2),
            ("grid_opacity", "Grid opacity", 0.0, 0.30, 0.01, 2),
        ),
    ),
    (
        "ANALOG EFFECTS",
        (
            ("glow_intensity", "Glow", 0.0, 2.5, 0.05, 2),
            ("bloom_intensity", "Bloom", 0.0, 1.5, 0.05, 2),
            ("grain_intensity", "Film grain", 0.0, 0.5, 0.01, 2),
            ("flicker_intensity", "Flicker", 0.0, 0.3, 0.01, 2),
            ("trail_persistence", "Trail persistence", 0.0, 0.96, 0.02, 2),
            ("scanline_intensity", "Scanlines", 0.0, 0.3, 0.01, 2),
            ("chromatic_aberration", "Chromatic separation", 0.0, 1.0, 0.02, 2),
            ("background_brightness", "Background brightness", 0.0, 1.5, 0.05, 2),
        ),
    ),
    (
        "OVERLAYS & QUALITY",
        (
            ("label_percentage", "Strong-node labels", 0.0, 1.0, 0.01, 2),
            ("label_max_count", "Maximum labels", 1, 2000, 10, 0),
            ("label_box_size", "Label box size", 2.0, 24.0, 0.5, 1),
            ("label_text_size", "Label text size", 6, 24, 1, 0),
            ("label_min_opacity", "Label opacity floor", 0.05, 1.0, 0.05, 2),
        ),
    ),
    (
        "SYSTEM",
        (
            ("random_seed", "Random seed", 0, 2147483647, 1, 0),
            ("performance_limit_fps", "Preview FPS limit", 24, 144, 1, 0),
            ("analysis_lookahead_ms", "Analysis look-ahead", 50, 500, 10, 0),
        ),
    ),
)

INTEGER_FIELDS = {
    "hop_size",
    "min_peaks_per_frame",
    "max_peaks_per_frame",
    "max_active_points",
    "visible_point_limit",
    "maximum_retained_phrases",
    "max_connections_per_point",
    "max_temporal_connections",
    "max_intra_connections",
    "max_similarity_connections",
    "visible_line_limit",
    "label_max_count",
    "label_text_size",
    "random_seed",
    "performance_limit_fps",
    "analysis_lookahead_ms",
}

COMBO_FIELDS = {
    "palette": (
        "Full Spectral Rainbow",
        "Birdsong Spectral",
        "Scientific Neon",
        "Archival Gold",
        "Monochrome Laboratory",
        "Deep Space",
    ),
    "preview_quality": ("Medium", "High"),
    "export_quality": ("Medium", "High"),
}

BOOLEAN_FIELDS = (
    ("spiderweb", "Spiderweb"),
    ("automatic_rotation", "Automatic rotation"),
    ("auto_fit_camera", "Auto fit geometry"),
    ("camera_follow_new_nodes", "Follow new nodes"),
    ("grid_enabled", "3D acoustic grid"),
    ("persistent_history", "Persistent history"),
    ("reverse_palette", "Reverse frequency palette"),
    ("scientific_labels", "Scientific labels in preview"),
    ("scientific_labels_export", "Scientific labels in export"),
    ("frequency_legend_preview", "Frequency legend in preview"),
    ("frequency_legend_export", "Frequency legend in export"),
)


class FloatSliderControl(QWidget):
    valueChanged = Signal(float)

    def __init__(
        self,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
        value: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._step = float(step)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(
            round(float(minimum) / self._step),
            round(float(maximum) / self._step),
        )
        self.spin = QDoubleSpinBox()
        self.spin.setRange(float(minimum), float(maximum))
        self.spin.setSingleStep(self._step)
        self.spin.setDecimals(int(decimals))
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(72)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)
        self.setValue(value)

    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, value: float) -> None:
        bounded = max(
            self.spin.minimum(),
            min(self.spin.maximum(), float(value)),
        )
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(round(bounded / self._step))
        self.spin.setValue(bounded)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)

    def _slider_changed(self, value: int) -> None:
        selected = value * self._step
        self.spin.blockSignals(True)
        self.spin.setValue(selected)
        self.spin.blockSignals(False)
        self.valueChanged.emit(selected)

    def _spin_changed(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(round(value / self._step))
        self.slider.blockSignals(False)
        self.valueChanged.emit(float(value))


class ColorButton(QPushButton):
    colorChanged = Signal(str)

    def __init__(self, value: str, parent=None) -> None:
        super().__init__(parent)
        self._color = "#000000"
        self.clicked.connect(self._choose)
        self.setColor(value)

    def color(self) -> str:
        return self._color

    def setColor(self, value: str) -> None:
        color = QColor(str(value))
        if not color.isValid():
            color = QColor("#000000")
        self._color = color.name().upper()
        lightness = color.lightnessF()
        foreground = "#101114" if lightness > 0.58 else "#F4F5F8"
        self.setText(self._color)
        self.setStyleSheet(
            f"background-color: {self._color}; color: {foreground};"
        )

    def _choose(self) -> None:
        selected = QColorDialog.getColor(
            QColor(self._color),
            self,
            "Select visualization background",
        )
        if selected.isValid():
            self.setColor(selected.name())
            self.colorChanged.emit(self._color)


class SettingsPanel(QFrame):
    setting_changed = Signal(str, object)

    def __init__(self, settings: VisualizationSettings, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("settingsPanel")
        self.setMinimumWidth(292)
        self.setMaximumWidth(370)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        title = QLabel("LIVE PARAMETERS")
        title.setObjectName("sectionTitle")
        title.setContentsMargins(16, 14, 16, 6)
        outer.addWidget(title)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 6, 14, 18)
        layout.setSpacing(10)
        self.controls: dict[str, QWidget] = {}

        for section_name, specs in SECTIONS:
            heading = QLabel(section_name)
            heading.setObjectName("parameterHeading")
            layout.addWidget(heading)
            form = QFormLayout()
            form.setSpacing(7)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
            for name, label, minimum, maximum, step, decimals in specs:
                if name in {"horizontal_spacing", "frequency_spacing"}:
                    control = FloatSliderControl(
                        minimum,
                        maximum,
                        step,
                        decimals,
                        float(getattr(settings, name)),
                    )
                    axis_name = (
                        "horizontal brightness"
                        if name == "horizontal_spacing"
                        else "vertical logarithmic frequency"
                    )
                    control.setToolTip(
                        f"Stretches only the {axis_name} axis. "
                        "Existing points move immediately."
                    )
                    control.valueChanged.connect(
                        lambda value, key=name: self.setting_changed.emit(
                            key,
                            value,
                        )
                    )
                elif name == "fft_size":
                    control = QComboBox()
                    control.addItems(["512", "1024", "2048", "4096", "8192"])
                    control.setCurrentText(str(settings.fft_size))
                    control.currentTextChanged.connect(
                        lambda value, key=name: self.setting_changed.emit(
                            key, int(value)
                        )
                    )
                elif name in INTEGER_FIELDS:
                    control = QSpinBox()
                    control.setRange(int(minimum), int(maximum))
                    control.setSingleStep(int(step))
                    control.setValue(int(getattr(settings, name)))
                    control.valueChanged.connect(
                        lambda value, key=name: self.setting_changed.emit(key, value)
                    )
                else:
                    control = QDoubleSpinBox()
                    control.setRange(float(minimum), float(maximum))
                    control.setSingleStep(float(step))
                    control.setDecimals(int(decimals))
                    control.setKeyboardTracking(False)
                    control.setValue(float(getattr(settings, name)))
                    control.valueChanged.connect(
                        lambda value, key=name: self.setting_changed.emit(key, value)
                    )
                self.controls[name] = control
                form.addRow(label, control)
            layout.addLayout(form)

        heading = QLabel("PALETTE & MODES")
        heading.setObjectName("parameterHeading")
        layout.addWidget(heading)
        combo_form = QFormLayout()
        for name, options in COMBO_FIELDS.items():
            control = QComboBox()
            control.addItems(options)
            control.setCurrentText(str(getattr(settings, name)))
            control.currentTextChanged.connect(
                lambda value, key=name: self.setting_changed.emit(key, value)
            )
            self.controls[name] = control
            combo_form.addRow(name.replace("_", " ").title(), control)
        layout.addLayout(combo_form)

        color_form = QFormLayout()
        background_control = ColorButton(settings.background_color)
        background_control.setToolTip(
            "Select any background color; Background brightness controls "
            "its intensity."
        )
        background_control.colorChanged.connect(
            lambda value: self.setting_changed.emit("background_color", value)
        )
        self.controls["background_color"] = background_control
        color_form.addRow("Background color", background_control)
        layout.addLayout(color_form)

        for name, label in BOOLEAN_FIELDS:
            control = QCheckBox(label)
            if name == "spiderweb":
                control.setToolTip(
                    "Connects every visible pair when possible, bounded by "
                    "the Visible line limit for GPU safety."
                )
            control.setChecked(getattr(settings, name))
            control.toggled.connect(
                lambda value, key=name: self.setting_changed.emit(key, value)
            )
            self.controls[name] = control
            layout.addWidget(control)
        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def set_values(self, settings: VisualizationSettings) -> None:
        for name, value in asdict(settings).items():
            control = self.controls.get(name)
            if control is None:
                continue
            control.blockSignals(True)
            if isinstance(control, QCheckBox):
                control.setChecked(bool(value))
            elif isinstance(control, QComboBox):
                control.setCurrentText(str(value))
            elif isinstance(control, ColorButton):
                control.setColor(str(value))
            else:
                control.setValue(value)
            control.blockSignals(False)
