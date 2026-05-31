# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Optional

# Guard against numpy 2.x / numpy.core removal conflict with QGIS's bundled numpy.
try:
    import numpy as np
    # Trigger the submodule that commonly fails on version mismatch
    import numpy.core.multiarray  # noqa: F401
except (ImportError, AttributeError) as _np_err:
    raise ImportError(
        "numpy could not be imported correctly. This is almost always caused by a "
        "version conflict: a newer numpy (2.x) was pip-installed alongside QGIS's "
        "bundled numpy (1.x), and numpy.core was removed in 2.0.\n\n"
        "Fix — open the OSGeo4W Shell and run:\n"
        '    python -m pip install "numpy<2.0" --force-reinstall\n'
        "Then restart QGIS.\n\n"
        f"Original error: {_np_err}"
    ) from _np_err

import requests

from qgis.PyQt import uic, QtWidgets
from qgis.PyQt.QtCore import QThread, pyqtSignal
from qgis.PyQt.QtWidgets import QMessageBox, QApplication
from osgeo import gdal, osr

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'population_model_dialog_base.ui'))

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HF_BASE_URL = (
    "/vsicurl/https://huggingface.co/datasets/Daksh17440/"
    "global_population_data/resolve/main/landscan-global-{}.tif"
)

# ─────────────────────────────────────────────────────────────────────────────
# Page-index constants  (must match the .ui stacked-widget order exactly)
# ─────────────────────────────────────────────────────────────────────────────
PAGE_MODE = 0   # Application selector         (shared entry point)
PAGE_ROI = 1   # Region of Interest            (shared)
PAGE_MODEL = 2   # Model selection               (Train & Run only)
PAGE_WEIGHTS = 3   # Weights upload                (Run Existing only)
# Hyperparameters               (shared; locked in Run Existing)
PAGE_HYPERPARAM = 4
PAGE_LOSS = 5   # Loss function                 (Train & Run only)
PAGE_TIMEFRAME = 6   # Time frame                    (shared)
PAGE_OUTPUT = 7   # Output directory              (shared)

# Ordered page sequences for each mode
FLOW_TRAIN_RUN = [PAGE_MODE, PAGE_ROI, PAGE_MODEL,
                  PAGE_HYPERPARAM, PAGE_LOSS, PAGE_TIMEFRAME, PAGE_OUTPUT]
FLOW_RUN_EXISTING = [PAGE_MODE, PAGE_ROI, PAGE_WEIGHTS,
                     PAGE_HYPERPARAM, PAGE_TIMEFRAME, PAGE_OUTPUT]

# Default hyperparameter values (single source of truth)
HYPERPARAM_DEFAULTS = {
    "patch_size": 64,
    "stride": 32,
    "batch_size": 8,
    "timesteps": 4,
    "epochs": 10,
}


# ════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ════════════════════════════════════════════════════════════════════════════

def _save_checkpoint(model, hyperparams: dict, architecture: str, path: str) -> None:
    """
    Save a training checkpoint that carries architecture identity and
    all hyperparameters alongside the model weights.

    Checkpoint schema
    -----------------
    {
        "architecture": str,          # e.g. "ConvLSTM"
        "hyperparams":  dict,         # patch_size, stride, batch_size, timesteps, epochs
        "state_dict":   OrderedDict,  # model.state_dict()
    }
    """
    import torch
    torch.save(
        {
            "architecture": architecture,
            "hyperparams": hyperparams,
            "state_dict": model.state_dict(),
        },
        path,
    )


def _load_checkpoint(path: str) -> dict:
    """
    Load a checkpoint saved by _save_checkpoint and return the raw dict.
    Raises ValueError with a user-friendly message if the file is not a
    valid LandScan plugin checkpoint.
    """
    import torch
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Could not read weights file:\n{exc}") from exc

    required = {"architecture", "hyperparams", "state_dict"}
    missing = required - set(ckpt.keys())
    if missing:
        raise ValueError(
            "The selected file does not appear to be a LandScan plugin checkpoint.\n"
            f"Missing keys: {', '.join(sorted(missing))}\n\n"
            "Only .pth files produced by this plugin's 'Train and Run New Model' "
            "workflow are supported."
        )
    return ckpt


# ════════════════════════════════════════════════════════════════════════════
# Background worker thread
# ════════════════════════════════════════════════════════════════════════════
class ProjectionWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)   # emits a dict of output file paths
    error = pyqtSignal(str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self.out_gt = None
        self.historical_paths = []
        self.actual_path = None

    def run(self):
        try:
            data = self._load_population_data()
            result_paths = self._run_model(data)
            self.finished.emit(result_paths)
        except Exception as exc:
            self.error.emit(str(exc))

    # ── GeoTIFF helper ───────────────────────────────────────────────────────

    def _save_tiff(self, arr: np.ndarray, path: str):
        """Save a 2-D float32 numpy array as a GeoTIFF with NaN as nodata."""
        drv = gdal.GetDriverByName("GTiff")
        H, W = arr.shape
        ds = drv.Create(path, W, H, 1, gdal.GDT_Float32)
        ds.SetGeoTransform(self.out_gt)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetProjection(srs.ExportToWkt())
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(float("nan"))
        band.WriteArray(arr)
        ds.FlushCache()
        ds = None

    # ── data loading ─────────────────────────────────────────────────────────

    def _load_population_data(self) -> dict:
        gdal.UseExceptions()
        cfg = self.config

        roi = cfg["roi"]
        if roi["type"] == "bbox":
            lat_min, lat_max = roi["lat_min"], roi["lat_max"]
            lon_min, lon_max = roi["lon_min"], roi["lon_max"]
        else:
            lat_min, lat_max, lon_min, lon_max = roi["bbox"]

        start_yr = cfg["start_year"]
        end_yr = cfg["end_year"]
        target_year = end_yr + 1

        years_to_fetch = list(range(start_yr, end_yr + 1))
        if target_year <= 2023:
            years_to_fetch.append(target_year)

        pop_slices = []

        for yr in years_to_fetch:
            self.progress.emit(f"Streaming LandScan {yr} via HuggingFace…")
            ds = gdal.Open(HF_BASE_URL.format(yr))
            if not ds:
                raise ValueError(f"Failed to open remote raster for {yr}.")

            gt = ds.GetGeoTransform()
            col_min = max(0, int((lon_min - gt[0]) / gt[1]))
            col_max = min(ds.RasterXSize, int((lon_max - gt[0]) / gt[1]))
            row_min = max(0, int((lat_max - gt[3]) / gt[5]))
            row_max = min(ds.RasterYSize, int((lat_min - gt[3]) / gt[5]))
            width = col_max - col_min
            height = row_max - row_min

            if width <= 0 or height <= 0:
                raise ValueError(
                    "Selected ROI does not intersect the global raster extent.")

            arr = ds.GetRasterBand(1).ReadAsArray(
                col_min, row_min, width, height)

            # Replace LandScan's large-negative nodata sentinel with NaN
            arr = arr.astype(np.float32)
            arr[arr < 0] = np.nan

            if self.out_gt is None:
                self.out_gt = (
                    gt[0] + col_min * gt[1], gt[1], 0,
                    gt[3] + row_min * gt[5], 0, gt[5],
                )

            if yr <= end_yr:
                path = os.path.join(
                    cfg["out_dir"], f"LandScan_{yr}_Historical.tif")
                self.historical_paths.append(path)
                pop_slices.append(arr)
            else:
                path = os.path.join(
                    cfg["out_dir"], f"LandScan_{yr}_Actual.tif")
                self.actual_path = path

            self._save_tiff(arr, path)
            ds = None

        self.progress.emit("Historical data extracted successfully.")
        return {
            "population": np.stack(pop_slices, axis=0),
            "years": np.array(range(start_yr, end_yr + 1)),
        }

    # ── model dispatch ───────────────────────────────────────────────────────

    def _run_model(self, data: dict) -> dict:
        # ── numpy sanity check ────────────────────────────────────────────────
        # numpy.core was removed in numpy 2.0. If QGIS's bundled gdal/osgeo
        # pulled in numpy 1.x but pip later installed numpy 2.x (or vice-versa),
        # this import will fail with a cryptic C-extension error.
        try:
            import numpy.core.multiarray  # noqa: F401
        except (ImportError, AttributeError):
            raise RuntimeError(
                "numpy.core.multiarray failed to import.\n\n"
                "This is caused by a numpy version conflict between QGIS's "
                "bundled numpy and a newer version installed via pip.\n\n"
                "Fix — open the OSGeo4W Shell and run:\n"
                '    python -m pip install "numpy<2.0" --force-reinstall\n\n'
                "Then restart QGIS and try again."
            )

        # ── torch check ───────────────────────────────────────────────────────
        try:
            import torch  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "This model requires PyTorch, which is not installed.\n\n"
                "Because PyTorch is a very large Deep Learning library (~2.5 GB), "
                "it must be installed manually.\n\n"
                "Please open the OSGeo4W Shell and run:\n"
                "    python -m pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cpu"
            )

        cfg = self.config
        target_year = cfg["end_year"] + 1
        mode = cfg["mode"]

        if mode == "run_existing":
            pred_arr = self._run_inference_only(
                data, cfg["weights_path"], cfg["architecture"])
            weights_saved = None
        else:
            name = cfg["model"]
            if name == "ConvLSTM":
                model, pred_arr = self._run_convlstm(data)
            elif name == "Vision Transformers (ViT)":
                model, pred_arr = self._run_vit(data)
            else:
                raise ValueError(f"Unknown model: {name}")

            # Save checkpoint with architecture + hyperparams embedded
            weights_saved = os.path.join(
                cfg["out_dir"], f"{name.replace(' ', '_')}_Weights.pth"
            )
            _save_checkpoint(
                model,
                hyperparams={k: cfg[k] for k in HYPERPARAM_DEFAULTS},
                architecture=name,
                path=weights_saved,
            )

        proj_path = os.path.join(
            cfg["out_dir"], f"LandScan_{target_year}_Projected.tif")
        self._save_tiff(pred_arr, proj_path)

        return {
            "historical": self.historical_paths,
            "actual": self.actual_path,
            "projected": proj_path,
            "weights": weights_saved,
        }

    # ── model stubs ──────────────────────────────────────────────────────────

    def _run_inference_only(self, data: dict, weights_path: str, architecture: str) -> np.ndarray:
        """Load checkpoint and run inference using the embedded architecture."""
        self.progress.emit(
            f"Loading {architecture} weights from: {weights_path}")
        _load_checkpoint(weights_path)
        # TODO: instantiate model from ckpt["architecture"] + ckpt["hyperparams"],
        #       call model.load_state_dict(ckpt["state_dict"]), then run forward pass.
        self.progress.emit(f"Running {architecture} inference…")
        # dummy – replace with real forward pass
        return data["population"][-1]

    def _run_convlstm(self, data: dict):
        """Train ConvLSTM and return (model, prediction_array)."""
        self.progress.emit("ConvLSTM training + inference running…")
        # TODO: real ConvLSTM training logic here

        class _DummyModel:
            def state_dict(self):
                return {}
        return _DummyModel(), data["population"][-1]

    def _run_vit(self, data: dict):
        """Train ViT and return (model, prediction_array)."""
        self.progress.emit("ViT training + inference running…")
        # TODO: real ViT training logic here

        class _DummyModel:
            def state_dict(self):
                return {}
        return _DummyModel(), data["population"][-1]


# ════════════════════════════════════════════════════════════════════════════
# Nominatim helper
# ════════════════════════════════════════════════════════════════════════════
def _nominatim_search(query: str) -> list:
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": query, "format": "json",
                "limit": 10, "addressdetails": 0},
        headers={"User-Agent": "QGIS-LandScanPlugin/1.0"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# ════════════════════════════════════════════════════════════════════════════
# Dialog
# ════════════════════════════════════════════════════════════════════════════
class population_modelDialog(QtWidgets.QDialog, FORM_CLASS):
    """
    Wizard-style dialog with two independent page flows:

      Run Existing  → Mode → ROI → Weights → Hyperparams (locked) → Time Frame → Output
      Train & Run   → Mode → ROI → Model   → Hyperparams (free)   → Loss → Time Frame → Output

    When the user lands on PAGE_WEIGHTS and picks a file, the checkpoint is
    parsed immediately and its hyperparameters are written into the spinboxes
    on PAGE_HYPERPARAM, which are then locked to prevent accidental mismatch.
    Pressing Back from PAGE_HYPERPARAM unlocks them again.
    """

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.iface = iface

        self._worker: Optional[ProjectionWorker] = None
        self._nom_results: list = []
        # parsed when weights file is chosen
        self._checkpoint: Optional[dict] = None

        self._flow: list[int] = FLOW_TRAIN_RUN
        self._flow_pos: int = 0

        self._setup_pages()
        self._connect_signals()
        self._go_to_flow_pos(0)

    # ── initial setup ────────────────────────────────────────────────────────

    def _setup_pages(self):
        self.radio_train_run.setChecked(True)

        self.radio_nominatim.setChecked(True)
        self._toggle_roi_sections()

        downloads_path = os.path.expanduser("~/Downloads")
        self.dir_output.setFilePath(downloads_path)

        for sb in (self.box_min_y, self.box_max_y):
            sb.setRange(-90.0, 90.0)
            sb.setDecimals(6)
        for sb in (self.box_min_x, self.box_max_x):
            sb.setRange(-180.0, 180.0)
            sb.setDecimals(6)

        # Warning banner hidden by default; shown only in Run Existing flow
        self.frame_hyperparam_warning.setVisible(False)

        # Info panel starts empty
        self._reset_weights_info_panel()

    def _connect_signals(self):
        self.btn_next.clicked.connect(self._on_next)
        self.btn_back.clicked.connect(self._on_back)
        self.btn_run.clicked.connect(self._on_run)

        self.radio_nominatim.toggled.connect(self._toggle_roi_sections)
        self.radio_bbox.toggled.connect(self._toggle_roi_sections)
        self.radio_upload.toggled.connect(self._toggle_roi_sections)

        self.btn_search.clicked.connect(self._on_search)
        self.input_search.returnPressed.connect(self._on_search)
        self.combo_suggestions.currentIndexChanged.connect(
            self._on_suggestion_selected)

        self.radio_train_run.toggled.connect(self._on_mode_changed)
        self.radio_run_existing.toggled.connect(self._on_mode_changed)

        # Parse checkpoint as soon as user selects a weights file
        self.file_weights.fileChanged.connect(self._on_weights_file_changed)

    # ── flow / navigation ────────────────────────────────────────────────────

    def _active_flow(self) -> list[int]:
        if self.radio_run_existing.isChecked():
            return FLOW_RUN_EXISTING
        return FLOW_TRAIN_RUN

    def _on_mode_changed(self):
        self._flow = self._active_flow()
        self._refresh_buttons()

    def _go_to_flow_pos(self, pos: int):
        self._flow = self._active_flow()
        self._flow_pos = pos
        page_index = self._flow[pos]

        # Side-effects when arriving at certain pages
        if page_index == PAGE_HYPERPARAM:
            self._apply_hyperparam_mode()

        self.stackedWidget.setCurrentIndex(page_index)
        self._update_step_labels(page_index)
        self._refresh_buttons()

    def _refresh_buttons(self):
        pos = self._flow_pos
        last = len(self._flow) - 1
        self.btn_back.setEnabled(pos > 0)
        self.btn_next.setEnabled(pos < last)
        self.btn_run.setEnabled(pos == last)

    def _update_step_labels(self, page_index: int):
        """Stamp the correct step number on pages whose label depends on the flow."""
        flow = self._flow
        if page_index not in flow:
            return
        step = flow.index(page_index)

        if page_index == PAGE_HYPERPARAM:
            self.lbl_hyperparam_step.setText(f"Step {step}: Hyperparameters:")
        if page_index == PAGE_OUTPUT:
            self.lbl_output_step.setText(
                f"Step {step}: Output Directory Setup:")
        if page_index == PAGE_TIMEFRAME:
            self.lbl_timeframe_step.setText(f"Step {step}: Time Frame:")
        if page_index == PAGE_WEIGHTS:
            self.lbl_weights_step.setText(
                f"Step {step}: Pre-Trained Model Weights:")

    def _on_next(self):
        if self._validate_current_page():
            self._go_to_flow_pos(self._flow_pos + 1)

    def _on_back(self):
        self._go_to_flow_pos(self._flow_pos - 1)

    # ── hyperparameter lock / unlock ─────────────────────────────────────────

    def _apply_hyperparam_mode(self):
        """
        Called when navigating TO PAGE_HYPERPARAM.
        • Run Existing: fill spinboxes from checkpoint, lock them, show warning.
        • Train & Run:  reset to defaults, unlock, hide warning.
        """
        if self.radio_run_existing.isChecked() and self._checkpoint:
            hp = self._checkpoint.get("hyperparams", {})
            self.spin_patch_size.setValue(
                hp.get("patch_size", HYPERPARAM_DEFAULTS["patch_size"]))
            self.spin_stride.setValue(
                hp.get("stride", HYPERPARAM_DEFAULTS["stride"]))
            self.spin_batch_size.setValue(
                hp.get("batch_size", HYPERPARAM_DEFAULTS["batch_size"]))
            self.spin_timesteps.setValue(
                hp.get("timesteps", HYPERPARAM_DEFAULTS["timesteps"]))
            self.spin_epochs.setValue(
                hp.get("epochs", HYPERPARAM_DEFAULTS["epochs"]))
            self._set_hyperparam_locked(True)
            self.frame_hyperparam_warning.setVisible(True)
        else:
            # Train & Run or no checkpoint yet: editable defaults
            self.spin_patch_size.setValue(HYPERPARAM_DEFAULTS["patch_size"])
            self.spin_stride.setValue(HYPERPARAM_DEFAULTS["stride"])
            self.spin_batch_size.setValue(HYPERPARAM_DEFAULTS["batch_size"])
            self.spin_timesteps.setValue(HYPERPARAM_DEFAULTS["timesteps"])
            self.spin_epochs.setValue(HYPERPARAM_DEFAULTS["epochs"])
            self._set_hyperparam_locked(False)
            self.frame_hyperparam_warning.setVisible(False)

    def _set_hyperparam_locked(self, locked: bool):
        """Enable or disable all hyperparameter spinboxes."""
        for spin in (self.spin_patch_size, self.spin_stride,
                     self.spin_batch_size, self.spin_timesteps,
                     self.spin_epochs):
            spin.setEnabled(not locked)

    # ── weights file handling ────────────────────────────────────────────────

    def _on_weights_file_changed(self, path: str):
        """Parse the checkpoint the moment a file is selected and update the info panel."""
        self._checkpoint = None
        self._reset_weights_info_panel()

        if not path or not os.path.isfile(path):
            return

        try:
            __import__("torch")
        except ImportError:
            self.lbl_detected_status.setText(
                "⚠  PyTorch not installed — cannot parse file yet.")
            return

        try:
            ckpt = _load_checkpoint(path)
        except ValueError as exc:
            self.lbl_detected_status.setText(f"✗  {exc}")
            return

        self._checkpoint = ckpt
        hp = ckpt.get("hyperparams", {})

        self.lbl_detected_arch.setText(
            f"Architecture:  {ckpt.get('architecture', '—')}"
        )
        self.lbl_detected_patch.setText(
            f"Patch size:  {hp.get('patch_size', '—')}   "
            f"Stride:  {hp.get('stride', '—')}   "
            f"Batch size:  {hp.get('batch_size', '—')}"
        )
        self.lbl_detected_time.setText(
            f"Timesteps:  {hp.get('timesteps', '—')}   "
            f"Epochs trained:  {hp.get('epochs', '—')}"
        )
        self.lbl_detected_status.setText(
            "✓  Valid LandScan checkpoint — hyperparameters will be loaded automatically."
        )
        self.lbl_detected_status.setStyleSheet(
            "color: green; font-style: italic;")

    def _reset_weights_info_panel(self):
        self.lbl_detected_arch.setText("Architecture:  —")
        self.lbl_detected_patch.setText(
            "Patch size:  —     Stride:  —     Batch size:  —")
        self.lbl_detected_time.setText("Timesteps:  —     Epochs trained:  —")
        self.lbl_detected_status.setText("")
        self.lbl_detected_status.setStyleSheet(
            "color: gray; font-style: italic;")

    # ── per-page validation ──────────────────────────────────────────────────

    def _validate_current_page(self) -> bool:
        page = self.stackedWidget.currentIndex()

        if page == PAGE_MODE:
            return True

        if page == PAGE_ROI:
            if self.radio_nominatim.isChecked():
                idx = self.combo_suggestions.currentIndex()
                if idx < 0 or not self._nom_results \
                        or "_parsed_bbox" not in self._nom_results[idx]:
                    QMessageBox.warning(self, "No region",
                                        "Please search for and select a valid region.")
                    return False
            elif self.radio_bbox.isChecked():
                if self.box_min_y.value() >= self.box_max_y.value() or \
                        self.box_min_x.value() >= self.box_max_x.value():
                    QMessageBox.warning(self, "Invalid bounds",
                                        "Min value must be less than max value.")
                    return False
            else:
                if not os.path.isfile(self.frame.filePath()):
                    QMessageBox.warning(self, "No file",
                                        "Please select a valid shapefile.")
                    return False

        elif page == PAGE_WEIGHTS:
            path = self.file_weights.filePath()
            if not path or not os.path.isfile(path):
                QMessageBox.warning(self, "No weights file",
                                    "Please select a valid .pth / .pt weights file.")
                return False
            if self._checkpoint is None:
                QMessageBox.warning(
                    self, "Invalid checkpoint",
                    "The selected file could not be parsed as a LandScan checkpoint.\n"
                    "Only .pth files produced by this plugin's training workflow are supported."
                )
                return False

        elif page == PAGE_TIMEFRAME:
            start = self.spin_start_year.value()
            end = self.spin_end_year.value()
            if end <= start:
                QMessageBox.warning(self, "Invalid years",
                                    "End year must be greater than start year.")
                return False
            # For Train & Run: only block if there are FEWER years than timesteps.
            # Having more years than timesteps is fine — the model uses a sliding
            # window of size `timesteps` across all available years.
            if self.radio_train_run.isChecked():
                n_years = end - start   # number of input rasters
                timesteps = self.spin_timesteps.value()
                if n_years < timesteps:
                    QMessageBox.warning(
                        self,
                        "Not enough years for Timesteps",
                        f"You selected {n_years} historical year(s) "
                        f"(Start={start} → End={end}),\n"
                        f"but Timesteps is set to {timesteps}.\n\n"
                        f"You need at least {timesteps} years of data "
                        f"(i.e. End Year ≥ {start + timesteps}) "
                        f"or reduce Timesteps to ≤ {n_years}.",
                    )
                    return False

        elif page == PAGE_OUTPUT:
            out_dir = self.dir_output.filePath()
            if not out_dir or not os.path.isdir(out_dir):
                QMessageBox.warning(self, "Invalid directory",
                                    "Please select a valid output directory.")
                return False

        return True

    # ── ROI helpers ──────────────────────────────────────────────────────────

    def _toggle_roi_sections(self):
        nom = self.radio_nominatim.isChecked()
        bbox = self.radio_bbox.isChecked()
        shp = self.radio_upload.isChecked()
        self.input_search.setEnabled(nom)
        self.btn_search.setEnabled(nom)
        self.combo_suggestions.setEnabled(nom)
        self.gridLayoutWidget.setEnabled(bbox)
        self.frame.setEnabled(shp)

    def _on_search(self):
        query = self.input_search.text().strip()
        if not query:
            return
        self.btn_search.setEnabled(False)
        self.combo_suggestions.clear()
        QApplication.processEvents()
        try:
            self._nom_results = _nominatim_search(query)
            if not self._nom_results:
                self.combo_suggestions.addItem("No results found")
            else:
                for r in self._nom_results:
                    self.combo_suggestions.addItem(
                        r.get("display_name", "Unknown"))
        except Exception as exc:
            QMessageBox.warning(self, "Search Error", str(exc))
        finally:
            self.btn_search.setEnabled(True)

    def _on_suggestion_selected(self, index: int):
        if 0 <= index < len(self._nom_results):
            bb = self._nom_results[index].get("boundingbox")
            if bb and len(bb) == 4:
                self._nom_results[index]["_parsed_bbox"] = tuple(
                    float(v) for v in bb)

    # ── config building ──────────────────────────────────────────────────────

    def _build_roi(self) -> dict:
        if self.radio_nominatim.isChecked():
            r = self._nom_results[self.combo_suggestions.currentIndex()]
            return {"type": "nominatim", "bbox": r["_parsed_bbox"]}
        elif self.radio_bbox.isChecked():
            return {
                "type": "bbox",
                "lat_min": self.box_min_y.value(),
                "lat_max": self.box_max_y.value(),
                "lon_min": self.box_min_x.value(),
                "lon_max": self.box_max_x.value(),
            }
        else:
            from qgis.core import QgsVectorLayer
            lyr = QgsVectorLayer(self.frame.filePath(), "roi", "ogr")
            ext = lyr.extent()
            return {
                "type": "shapefile",
                "bbox": (ext.yMinimum(), ext.yMaximum(),
                         ext.xMinimum(), ext.xMaximum()),
            }

    def _build_config(self) -> dict:
        mode = "run_existing" if self.radio_run_existing.isChecked() else "train_run"

        # Hyperparameters are read from the (possibly locked) spinboxes —
        # in Run Existing mode these were already filled from the checkpoint.
        cfg = {
            "mode": mode,
            "roi": self._build_roi(),
            "out_dir": self.dir_output.filePath(),
            "patch_size": self.spin_patch_size.value(),
            "stride": self.spin_stride.value(),
            "batch_size": self.spin_batch_size.value(),
            "timesteps": self.spin_timesteps.value(),
            "epochs": self.spin_epochs.value(),
            # ── Time frame: shared by both flows ──────────────────────────
            "start_year": self.spin_start_year.value(),
            "end_year": self.spin_end_year.value(),
        }

        if mode == "run_existing":
            cfg["weights_path"] = self.file_weights.filePath()
            cfg["architecture"] = self._checkpoint["architecture"]
            # No model-selection combo in this flow; architecture comes from checkpoint
            cfg["model"] = cfg["architecture"]
        else:
            cfg["model"] = self.combo_model.currentText()

        return cfg

    # ── run / thread ─────────────────────────────────────────────────────────

    def _on_run(self):
        if not self._validate_current_page():
            return
        try:
            config = self._build_config()
        except Exception as exc:
            QMessageBox.critical(self, "Configuration Error", str(exc))
            return

        self.btn_run.setEnabled(False)
        self.btn_back.setEnabled(False)

        self._worker = ProjectionWorker(config, parent=self)
        self._worker.progress.connect(
            lambda msg: self.iface.messageBar().pushMessage("LandScan", msg, duration=0)
        )
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, result: dict):
        from qgis.core import QgsRasterLayer, QgsProject

        for p in result["historical"]:
            lyr = QgsRasterLayer(p, os.path.basename(p))
            if lyr.isValid():
                QgsProject.instance().addMapLayer(lyr)

        if result["actual"]:
            lyr = QgsRasterLayer(
                result["actual"], os.path.basename(result["actual"]))
            if lyr.isValid():
                QgsProject.instance().addMapLayer(lyr)

        proj_lyr = QgsRasterLayer(
            result["projected"], os.path.basename(result["projected"]))
        if proj_lyr.isValid():
            QgsProject.instance().addMapLayer(proj_lyr)

        self.btn_run.setEnabled(True)
        self.btn_back.setEnabled(True)
        self.iface.messageBar().clearWidgets()

        out_dir = self._worker.config["out_dir"]
        if result["weights"]:
            self.iface.messageBar().pushSuccess(
                "LandScan",
                f"Training complete! Projection + weights saved to {out_dir}"
            )
        else:
            self.iface.messageBar().pushSuccess(
                "LandScan",
                f"Inference complete! Projection saved to {out_dir}"
            )

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(False)
        self.btn_back.setEnabled(True)
        QMessageBox.critical(self, "Projection Error", msg)
