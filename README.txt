# LandScan Population Modelling (QGIS Plugin)

A QGIS plugin that utilizes Deep Learning architectures to model and project population dynamics. The plugin streams historical LandScan global population rasters for custom Areas of Interest (AOIs) and applies spatial-temporal neural networks to forecast future population distributions.

## ✨ Features

* **Automated Data Streaming:** Fetches historical LandScan population rasters dynamically via Hugging Face.
* **Custom ROI Selection:** Define your study area using Nominatim (OpenStreetMap) text search, manual Bounding Box coordinates, or by uploading a custom Vector Shapefile.
* **Deep Learning Architectures:** Supports **ConvLSTM** and **Vision Transformers (ViT)** for spatial-temporal time-series forecasting.
* **Dual Execution Modes:**
  * **Train & Run:** Train a new model from scratch on the selected timeframe and hyperparameters, then run inferences. 
  * **Run Existing:** Load pre-trained `.pth` weights to run immediate projections (hyperparameters are automatically parsed and locked from the checkpoint).

## 🛠️ Prerequisites & Dependencies

The plugin handles most dependencies (like `requests` and `numpy < 2.0`) automatically by installing them into your QGIS Python environment. 

**Manual Requirement:** PyTorch is required for the Deep Learning models. Because of its large size, it must be installed manually via the OSGeo4W Shell:

```bash
python -m pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
```
*(Note: If you have a dedicated GPU configured with QGIS, you can install the CUDA version of PyTorch instead).*

## 🚀 Installation

1. Download the latest `.zip` release of this plugin.
2. Open QGIS.
3. Go to **Plugins** > **Manage and Install Plugins...** > **Install from ZIP**.
4. Select the downloaded `.zip` file and click **Install Plugin**.
5. Restart QGIS if prompted. The plugin icon will appear in your toolbar.

## 💻 Usage 

1. **Launch:** Click the LandScan plugin icon in the QGIS toolbar.
2. **Select Mode:** Choose to train a new model or run an existing one.
3. **Define ROI:** Search for a location, input a bounding box, or upload a shapefile.
4. **Configure Model:** * If training: Select the architecture (ConvLSTM/ViT) and define hyperparameters (Patch Size, Stride, Batch Size, Timesteps, Epochs).
    * If running existing: Upload your `.pth` checkpoint.
5. **Set Timeframe:** Define the historical start and end years.
6. **Execute:** Choose an output directory and click Run. 

The plugin will run in a background thread to prevent freezing the QGIS GUI. Once finished, the historical rasters and the projected raster will automatically load into your QGIS map canvas.

## 📂 Checkpoint Architecture

When training a new model, the plugin saves a `.pth` file containing the weights, architecture identity, and hyperparameter dictionary. This ensures zero configuration mismatch when sharing models for future inference.

## 👤 Author

**Daksh17440**
