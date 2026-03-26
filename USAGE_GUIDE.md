# Point-E Environment Usage Guide

## Quick Reference Commands

### Activate Virtual Environment
```powershell
# Windows PowerShell
.\.venv\Scripts\activate.ps1

# After activation, prompt will show (point-e-enhanced-cline-model) prefix
# Or use: .venv\Scripts\activate.bat for Command Prompt
```

### Verify Installation
```powershell
# Run comprehensive tests
python test_setup.py

# Quick import test
python -c "import torch; import point_e; print('✓ Setup OK')"

# Check specific package versions
pip list | Select-String "torch|numpy|jupyter"
```

### Launch Development Environment
```powershell
# Start JupyterLab
jupyter lab

# Start classic Jupyter
jupyter notebook

# Start IPython interactive shell
ipython
```

### Run Examples
```powershell
# Opens JupyterLab where you can select notebooks
jupyter lab

# Direct notebook execution (generates output in same directory)
jupyter nbconvert --to notebook --execute point_e/examples/text2pointcloud.ipynb
```

---

## Environment Management

### Create Fresh Virtual Environment (if needed)
```powershell
# Remove old environment
Remove-Item -Recurse -Force .venv

# Create new environment
python -m venv .venv

# Activate and install
.\.venv\Scripts\activate.ps1
pip install -r requirements.txt
pip install -e .
```

### Update Dependencies
```powershell
# Upgrade specific package
pip install --upgrade torch

# Upgrade all packages
pip install --upgrade pip setuptools wheel
pip install --upgrade -r requirements.txt

# Check for outdated packages
pip list --outdated
```

### Export Environment
```powershell
# Create frozen requirements with exact versions
pip freeze > requirements-frozen.txt

# This can be used to reproduce exact environment elsewhere
```

---

## Development Workflow

### Using the Point-E API

#### Basic Usage Pattern
```python
import torch
from point_e.models.download import load_checkpoint
from point_e.models.configs import model_from_config, MODEL_CONFIGS
from point_e.diffusion.configs import diffusion_from_config, DIFFUSION_CONFIGS
from point_e.diffusion.sampler import PointCloudSampler

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = model_from_config(MODEL_CONFIGS['base40M-textvec'], device)
model.eval()
model.load_state_dict(load_checkpoint('base40M-textvec', device))

# Create sampler
sampler = PointCloudSampler(
    device=device,
    models=[model],
    diffusions=[diffusion_from_config(DIFFUSION_CONFIGS['base40M-textvec'])],
)

# Generate point cloud
samples = next(sampler.sample_batch_progressive(
    batch_size=1, 
    model_kwargs=dict(texts=['a red motorcycle'])
))
```

#### Enhancement Functions
```python
from point_e.enhancements.applyenhancements import enhance_point_cloud
from point_e.util.plotting import plot_point_cloud

# Enhance point cloud
enhanced_pc = enhance_point_cloud(pc)

# Visualize
fig = plot_point_cloud(enhanced_pc, grid_size=3)
```

#### Point Cloud Operations
```python
from point_e.util.point_cloud import PointCloud
import numpy as np

# Create point cloud
coords = np.random.randn(1000, 3).astype(np.float32)
channels = {'R': np.ones(1000), 'G': np.zeros(1000), 'B': np.zeros(1000)}
pc = PointCloud(coords, channels)

# Access properties
print(f"Points: {len(pc.coords)}")
print(f"Channels: {list(pc.channels.keys())}")

# Save/Load (see ply_util module)
```

---

## Troubleshooting & Common Issues

### Issue: "ModuleNotFoundError: No module named 'point_e'"

**Cause**: Virtual environment not activated or package not installed in editable mode

**Solution**:
```powershell
# Activate environment
.\.venv\Scripts\activate.ps1

# Reinstall in editable mode
pip install -e .

# Verify installation
python -c "import point_e; print(point_e.__file__)"
```

### Issue: "CUDA out of memory" (if using GPU)

**Solution**: Reduce batch size or switch to CPU
```python
# Use CPU instead
device = torch.device('cpu')

# Or reduce batch size
batch_size = 1  # instead of 8
```

### Issue: "Connection timeout" downloading models

**Solution**: Models are cached after first download
```powershell
# Check cache location
python -c "from point_e.models.download import default_cache_dir; print(default_cache_dir())"

# Manual download retry
python -c "from point_e.models.download import load_checkpoint; load_checkpoint('base40M-textvec', device)"
```

### Issue: Jupyter kernel not found

**Solution**: Reinstall kernel
```powershell
python -m ipykernel install --user --name point_e --display-name "Point-E"
```

### Issue: JupyterLab extensions not loading

**Solution**: Rebuild JupyterLab
```powershell
jupyter lab build --dev-build=False --minimize=True
```

---

## Performance Tips

### CPU Mode Optimization
```python
# Use half precision (not always supported on CPU)
import torch
model = model.half()  # Use float16

# Reduce point cloud size
num_points = 1024  # instead of 4096

# Single-threaded if experiencing slowdowns
torch.set_num_threads(1)
```

### GPU Mode Optimization
```python
# Enable mixed precision training
from torch.cuda.amp import autocast

with autocast():
    output = model(input)

# Increase batch size if memory allows
batch_size = 16  # experiment with this

# Enable TFLoat32 for faster computation
torch.backends.cuda.matmul.allow_tf32 = True
```

### Memory Management
```python
# Clear CUDA cache
torch.cuda.empty_cache()

# Monitor memory usage
import torch
print(torch.cuda.memory_allocated())  # bytes
print(torch.cuda.memory_reserved())   # bytes

# Set memory growth (TensorFlow style)
# Not directly available in PyTorch, but can use:
torch.cuda.set_per_process_memory_fraction(0.5)  # 50% of GPU memory
```

---

## File Organization

### Where to put your code
```
Point-E-Enhanced-Cline-Model/
├── my_scripts/              # Create for your custom code
│   ├── my_inference.py
│   ├── my_batch_process.py
│   └── my_evaluation.py
│
├── notebooks/               # Create for custom notebooks
│   ├── my_experiments.ipynb
│   └── my_analysis.ipynb
│
└── data/                    # Create for your data
    ├── input_images/
    ├── output_clouds/
    └── results.json
```

---

## Reproducibility

### Environment Snapshot
```powershell
# Save current versions
pip freeze > environment_snapshot.txt

# Later: Create identical environment
python -m venv new_env
new_env\Scripts\activate.ps1
pip install -r environment_snapshot.txt
```

### Code Reproducibility
```python
# Set random seeds for reproducibility
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
```

---

## Documentation References

### Point-E API
- Model configs: `point_e/models/configs.py`
- Diffusion configs: `point_e/diffusion/configs.py`
- Sampler: `point_e/diffusion/sampler.py`
- Utilities: `point_e/util/` (point_cloud.py, mesh.py, plotting.py)

### External Documentation
- PyTorch: https://pytorch.org/docs/stable/
- CLIP: https://github.com/openai/CLIP
- NumPy: https://numpy.org/doc/
- SciPy: https://docs.scipy.org/
- Open3D: http://www.open3d.org/docs/

---

## Getting Help

### Debug Information
```powershell
# Complete environment info
pip show point-e
python -c "import sys; print(sys.version); import torch; print(torch.version)"
python test_setup.py  # Run full validation

# Check specific package
python -c "import point_e.models.configs; print(point_e.models.configs.MODEL_CONFIGS.keys())"
```

### Common Documentation Files
- `README.md` - Project overview
- `model-card.md` - Model details and usage
- `SETUP_SUMMARY.md` - Environment configuration
- `requirements.txt` - Dependencies list

---

**Last Updated**: March 26, 2026  
**Environment Status**: ✓ Ready for Use
