"""Export-path, resolution, frame-rate, and bitrate selection."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..export.ffmpeg_encoder import ExportOptions


class ExportDialog(QDialog):
    def __init__(self, source_path: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export visualization MP4")
        self.setModal(True)
        self.setMinimumWidth(520)
        root = QVBoxLayout(self)
        form = QFormLayout()

        self.path_edit = QLineEdit()
        source = Path(source_path)
        movies = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.MoviesLocation
        )
        movies_path = Path(movies) if movies else None
        default_dir = (
            movies_path
            if movies_path is not None and movies_path.is_dir()
            else source.parent
        )
        self.path_edit.setText(str(default_dir / f"{source.stem}_visualization.mp4"))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        form.addRow("Output path", path_row)

        self.width = QSpinBox()
        self.width.setRange(320, 7680)
        self.width.setValue(1920)
        self.height = QSpinBox()
        self.height.setRange(180, 4320)
        self.height.setValue(1080)
        size_row = QHBoxLayout()
        size_row.addWidget(self.width)
        size_row.addWidget(self.height)
        form.addRow("Resolution", size_row)

        self.fps = QComboBox()
        self.fps.addItems(["24", "30", "60"])
        self.fps.setCurrentText("60")
        form.addRow("Frames per second", self.fps)

        self.bitrate = QSpinBox()
        self.bitrate.setRange(500, 100000)
        self.bitrate.setValue(16000)
        self.bitrate.setSuffix(" kbps")
        form.addRow("Video bitrate", self.bitrate)
        root.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Save
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def options(self) -> ExportOptions:
        return ExportOptions(
            output_path=self.path_edit.text().strip(),
            width=self.width.value(),
            height=self.height.value(),
            fps=int(self.fps.currentText()),
            video_bitrate_kbps=self.bitrate.value(),
        ).validated()

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export visualization",
            self.path_edit.text(),
            "MP4 video (*.mp4)",
        )
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.path_edit.setText(path)
