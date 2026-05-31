# -*- coding: utf-8 -*-
"""
/***************************************************************************
 population_model
                                 A QGIS plugin
 The plugin offers diverse options of loss function and deep learning models
 to be applied on LandScan population rasters of custom AOI and time frame.
 ***************************************************************************/
"""
import os
import sys
import subprocess

from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox

# Initialize Qt resources from file resources.py
from . import resources  # noqa: F401

REQUIRED_PACKAGES = ["numpy", "requests"]


def _get_python_exe() -> str:
    """Find the actual Python executable, escaping the qgis-bin.exe trap."""
    if "python" in os.path.basename(sys.executable).lower():
        return sys.executable
    if sys.platform == "win32":
        return os.path.join(sys.prefix, "python.exe")
    return os.path.join(sys.prefix, "bin", "python3")


def _check_numpy_version() -> bool:
    """
    Return True if numpy is healthy (importable and < 2.0).
    If numpy 2.x is detected, warn the user and offer to downgrade automatically.
    numpy 2.0 removed numpy.core which breaks QGIS's bundled GDAL/osgeo bindings.
    """
    try:
        import numpy as np
        major = int(np.__version__.split(".")[0])
        if major < 2:
            return True  # all good
        # numpy 2.x detected — offer to downgrade
        reply = QMessageBox.warning(
            None,
            "numpy version conflict",
            f"numpy {np.__version__} is installed, but QGIS requires numpy < 2.0.\n\n"
            "numpy 2.0 removed numpy.core, which breaks GDAL and this plugin.\n\n"
            "Click OK to downgrade numpy automatically, or Cancel to abort.",
            QMessageBox.Ok | QMessageBox.Cancel,
        )
        if reply != QMessageBox.Ok:
            return False
        python_exe = _get_python_exe()
        creationflags = 0x08000000 if sys.platform == "win32" else 0
        try:
            subprocess.check_call(
                [python_exe, "-m", "pip", "install", "numpy<2.0",
                 "--force-reinstall"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
            )
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(
                errors="replace") if e.stderr else "Unknown error."
            QMessageBox.critical(
                None, "Downgrade failed",
                f"Could not downgrade numpy automatically:\n\n{err}\n\n"
                "Please run this manually in the OSGeo4W Shell:\n"
                '    python -m pip install "numpy<2.0" --force-reinstall\n\n'
                "Then restart QGIS.",
            )
            return False
        QMessageBox.information(
            None, "numpy downgraded",
            "numpy has been downgraded to a compatible version.\n"
            "Please restart QGIS for the change to take effect.",
        )
        return False  # restart required — don't continue loading
    except ImportError:
        return True  # not installed yet; handled by _ensure_dependencies


def _ensure_dependencies(plugin_dir: str) -> bool:
    """
    Check that every package in REQUIRED_PACKAGES can be imported.
    Any that are missing are installed into QGIS's own Python via pip.
    """
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return True

    missing_str = "\n".join(f"  - {p}" for p in missing)
    msg = (
        "The LandScan Population Modelling plugin requires the following "
        f"Python packages:\n\n{missing_str}\n\n"
        "Click OK to install them automatically."
    )
    reply = QMessageBox.question(
        None, "Install missing dependencies?", msg,
        QMessageBox.Ok | QMessageBox.Cancel)
    if reply != QMessageBox.Ok:
        return False

    python_exe = _get_python_exe()
    req_file = os.path.join(plugin_dir, "requirements.txt")

    try:
        creationflags = 0x08000000 if sys.platform == "win32" else 0
        subprocess.check_call(
            [python_exe, "-m", "pip", "install", "-r", req_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags
        )
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode(
            errors="replace") if e.stderr else "Unknown error."
        QMessageBox.critical(
            None,
            "Installation failed",
            f"Could not install dependencies:\n\n{err_msg}\n\n"
            f"Please run the following command manually in OSGeo4W Shell:\n"
            f'"{python_exe}" -m pip install -r "{req_file}"'
        )
        return False

    # Final verification
    still_missing = []
    for pkg in missing:
        try:
            __import__(pkg)
        except ImportError:
            still_missing.append(pkg)

    if still_missing:
        still_missing_str = "\n".join(f"  - {p}" for p in still_missing)
        QMessageBox.critical(
            None,
            "Import still failing",
            f"Installed but still cannot import:\n\n{still_missing_str}\n\n"
            "Try restarting QGIS, or install manually via OSGeo4W Shell."
        )
        return False

    QMessageBox.information(
        None, "Dependencies installed",
        "All required packages were installed successfully.\n"
        "The plugin is ready to use.")
    return True


class population_model:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor."""
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)

        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir, 'i18n',
            'population_model_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        self.actions = []
        self.menu = self.tr(u'&LandScan Population Modelling')
        self.first_start = None

    def tr(self, message):
        return QCoreApplication.translate('population_model', message)

    def add_action(self, icon_path, text, callback, enabled_flag=True,
                   add_to_menu=True, add_to_toolbar=True,
                   status_tip=None, whats_this=None, parent=None):
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)
        if whats_this is not None:
            action.setWhatsThis(whats_this)
        if add_to_toolbar:
            self.iface.addToolBarIcon(action)
        if add_to_menu:
            self.iface.addPluginToMenu(self.menu, action)

        self.actions.append(action)
        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        icon_path = ':/plugins/population_model/icon.svg'
        self.add_action(
            icon_path,
            text=self.tr(u'LandScan Population Modelling using DL'),
            callback=self.run,
            parent=self.iface.mainWindow())
        self.first_start = True

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&LandScan Population Modelling'), action)
            self.iface.removeToolBarIcon(action)

    def run(self):
        """Run method that performs all the real work."""
        if self.first_start:
            self.first_start = False

            # Check numpy version FIRST — numpy 2.x breaks GDAL/osgeo bindings
            if not _check_numpy_version():
                self.iface.messageBar().pushWarning(
                    "LandScan",
                    "numpy version conflict detected. Please restart QGIS after the fix.")
                self.first_start = True
                return

            # Install missing packages before importing the dialog
            if not _ensure_dependencies(self.plugin_dir):
                self.iface.messageBar().pushWarning(
                    "LandScan", "Plugin not loaded — missing dependencies.")
                self.first_start = True
                return

            # Safe to import dialog now
            from .population_model_dialog import population_modelDialog
            self.dlg = population_modelDialog(iface=self.iface)

        self.dlg.show()
        self.dlg.exec_()
