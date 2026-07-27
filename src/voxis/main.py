"""Application entry point."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from .config import load_settings
from .logging_config import configure_logging
from .ui.main_window import MainWindow


def main() -> int:
    configure_logging()
    surface = QSurfaceFormat()
    surface.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    surface.setVersion(3, 3)
    surface.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface.setDepthBufferSize(24)
    surface.setStencilBufferSize(8)
    surface.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(surface)

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setApplicationName("Voxis")
    app.setOrganizationName("Voxis")
    window = MainWindow(load_settings())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
