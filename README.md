# DeepLabCut GUI - Markerless Rodent Pose Estimation

A comprehensive PySide6 GUI application for the complete DeepLabCut workflow, designed for markerless pose estimation in rodents and other animals.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![DeepLabCut](https://img.shields.io/badge/DeepLabCut-3.0.0rc13-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

### Complete Workflow Coverage
- **Video Preprocessing** - Clean and re-encode videos to fix corruption issues
- **Project Management** - Create and manage DeepLabCut projects with ease
- **Frame Extraction** - Extract frames using K-means clustering (FAISS-GPU) or uniform sampling
- **Frame Labeling** - Intuitive interface for labeling keypoints and building skeletons
- **Training Dataset Creation** - Configure network architecture and augmentation
- **Model Training** - GPU-accelerated training with PyTorch backend
- **Video Analysis** - Analyze videos and create labeled visualizations
- **Outlier Refinement** - Extract and label outlier frames for model improvement
- **System Diagnostics** - Monitor GPU, CUDA, and dependency status

### Key Advantages
- **GPU Acceleration** - FAISS-GPU for fast K-means clustering, CUDA for training
- **Transfer Learning** - SuperAnimal TopViewMouse weights for better accuracy
- **Modern UI** - Clean, responsive PySide6 interface with dark theme
- **Batch Processing** - Handle multiple videos efficiently
- **Real-time Monitoring** - Track training progress and system resources

## Screenshots

### Main Interface
The application features 9 tabs covering the complete workflow:

1. **Clean Videos** - Video preprocessing and integrity checking
2. **Project Manager** - Project creation with multi-animal support
3. **Extract Frames** - K-means or uniform frame extraction
4. **Label Frames** - Keypoint management and skeleton building
5. **Create Training Dataset** - Network and augmentation configuration
6. **Train Network** - GPU-accelerated model training
7. **Analyze Videos** - Inference and labeled video creation
8. **Extract Outliers** - Iterative model refinement
9. **System Info** - Hardware and software diagnostics

## Installation

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended)
- 16GB+ RAM recommended
- Linux, macOS, or Windows

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Gianlz/markerless-rodent-pose.git
cd markerless-rodent-pose
```

2. **Install uv (Python package manager)**
```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

3. **Install dependencies**
```bash
uv sync
```

4. **Run the application**
```bash
uv run python main.py
```

### GPU Support (Recommended)

For CUDA support (NVIDIA GPUs):
```bash
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv add faiss-gpu
```

For CPU-only (not recommended for training):
```bash
uv add torch torchvision
uv add faiss-cpu
```

## Usage

### Basic Workflow

1. **Prepare Videos**
   - Add your rodent behavior videos
   - Optionally clean/re-encode problematic videos

2. **Create Project**
   - Set project name and experimenter
   - Add videos to the project
   - Choose single or multi-animal tracking

3. **Extract Frames**
   - Use K-means clustering for diverse frames
   - Or uniform sampling for evenly spaced frames
   - Typically 20-50 frames per video

4. **Label Frames**
   - Define keypoints (e.g., nose, ears, body, tail)
   - Build skeleton connections
   - Label all extracted frames

5. **Create Training Dataset**
   - Select network architecture (ResNet-50 recommended)
   - Choose weight initialization (SuperAnimal for best results)
   - Configure data augmentation

6. **Train Network**
   - Set training parameters (200k epochs typical)
   - Monitor progress in terminal
   - Training takes several hours with GPU

7. **Analyze Videos**
   - Run inference on new videos
   - Create labeled videos with pose overlay
   - Export predictions to CSV/H5

8. **Refine Model** (Optional)
   - Extract outlier frames
   - Label problematic frames
   - Retrain for improved accuracy

### Example: Mouse Tracking

```python
# Typical keypoints for top-view mouse tracking
keypoints = [
    'nose',
    'left_ear',
    'right_ear',
    'neck',
    'body_center',
    'tail_base',
    'tail_mid',
    'tail_tip'
]

# Skeleton connections
skeleton = [
    ['nose', 'neck'],
    ['neck', 'left_ear'],
    ['neck', 'right_ear'],
    ['neck', 'body_center'],
    ['body_center', 'tail_base'],
    ['tail_base', 'tail_mid'],
    ['tail_mid', 'tail_tip']
]
```

## Configuration

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- GPU: Not required (CPU training possible but slow)
- Storage: 10GB free space

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA RTX 3060 or better (8GB+ VRAM)
- Storage: 50GB+ free space (for videos and models)

### Network Architectures

**Single Animal:**
- ResNet-50/101/152 (standard, good accuracy)
- MobileNet (lightweight, faster inference)
- EfficientNet (balanced performance)

**Multi-Animal:**
- DLCRNet-MS5 (recommended)
- EfficientNet variants

### Weight Initialization

- **SuperAnimal TopViewMouse** - Best for rodents (transfer learning)
- **ImageNet** - General transfer learning
- **Random** - Train from scratch (requires more data)

## Project Structure

```
markerless-rodent-pose/
├── src/
│   ├── ui/                    # PySide6 GUI components
│   │   ├── tabs/              # 9 workflow tabs
│   │   └── styles/            # Theme and stylesheets
│   ├── core/                  # DeepLabCut wrappers
│   │   ├── frame_extractor.py
│   │   ├── project_manager.py
│   │   ├── label_manager.py
│   │   ├── training_manager.py
│   │   ├── train_manager.py
│   │   └── inference_manager.py
│   └── utils/                 # Utilities
│       ├── validators.py
│       └── video_utils.py
├── assets/                    # UI assets
│   └── styles/
├── tests/                     # E2E tests
│   └── e2e/
├── main.py                    # Application entry point
├── pyproject.toml             # Dependencies
└── README.md
```

## Testing

Run the end-to-end test suite:

```bash
# All tests
uv run pytest tests/e2e -v

# Specific test
uv run pytest tests/e2e/test_complete_workflow.py -v

# With coverage
uv run pytest tests/e2e --cov=src --cov-report=html
```

## Troubleshooting

### CUDA Not Available

Check the System Info tab (Tab 9) to verify GPU detection. If CUDA is not available:

1. Install NVIDIA drivers
2. Install CUDA toolkit
3. Reinstall PyTorch with CUDA support:
```bash
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory Errors

- Reduce batch size in training configuration
- Use a smaller network (MobileNet instead of ResNet)
- Close other GPU applications
- Use gradient accumulation

### Slow Frame Extraction

- Ensure FAISS-GPU is installed
- Check GPU is being used (System Info tab)
- Reduce cluster_resize_width parameter
- Use uniform extraction instead of K-means

### Video Codec Issues

Use the Clean Videos tab to re-encode problematic videos:
- Codec: libx264 (most compatible)
- CRF: 18 (high quality)
- Preset: medium (balanced speed/compression)

## Performance Tips

1. **Use GPU acceleration** - 10-100x faster training
2. **Transfer learning** - SuperAnimal weights reduce training time
3. **K-means extraction** - More diverse frames = better model
4. **Data augmentation** - Improves generalization
5. **Iterative refinement** - Extract and label outliers

## Dependencies

### Core
- Python 3.10+
- DeepLabCut >=3.0.0rc13
- PySide6 >=6.10.0
- PyTorch >=2.9.1
- TorchVision >=0.24.1

### GPU Acceleration
- FAISS-GPU >=1.7.2
- CUDA 12.1+
- cuDNN

### Utilities
- OpenCV >=4.8.0
- NumPy, Pandas, PyYAML
- PyTables 3.8.0

See `pyproject.toml` for complete dependency list.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this software in your research, please cite:

```bibtex
@software{deeplabcut_gui,
  title = {DeepLabCut GUI - Markerless Rodent Pose Estimation},
  author = {Zugno, Gianluca},
  year = {2024},
  url = {https://github.com/Gianlz/markerless-rodent-pose}
}
```

And cite the original DeepLabCut paper:

```bibtex
@article{Mathis2018,
  title={DeepLabCut: markerless pose estimation of user-defined body parts with deep learning},
  author={Mathis, Alexander and Mamidanna, Pranav and Cury, Kevin M and Abe, Taiga and Murthy, Venkatesh N and Mathis, Mackenzie Weygandt and Bethge, Matthias},
  journal={Nature neuroscience},
  volume={21},
  number={9},
  pages={1281--1289},
  year={2018},
  publisher={Nature Publishing Group}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **DeepLabCut Team** - For the excellent pose estimation framework
- **FAISS Team** - For fast similarity search and clustering
- **PyTorch Team** - For the deep learning framework
- **Qt/PySide6** - For the GUI framework

## Support

- **Issues**: [GitHub Issues](https://github.com/Gianlz/markerless-rodent-pose/issues)
- **Documentation**: See `docs/` folder
- **DeepLabCut Docs**: [deeplabcut.github.io](https://deeplabcut.github.io)


## Authors

- **Gianlz** - Initial work - [GitHub](https://github.com/Gianlz)

---

**Note**: This is a GUI wrapper around DeepLabCut. For the core DeepLabCut library, see [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut).
