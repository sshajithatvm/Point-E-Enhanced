# Point·E

![Animation of four 3D point clouds rotating](point_e/examples/paper_banner.gif)

This is the official code and model release for [Point-E: A System for Generating 3D Point Clouds from Complex Prompts](https://arxiv.org/abs/2212.08751).

# Usage

Install with `pip install -e .`.

To get started with examples, see the following notebooks:

 * [image2pointcloud.ipynb](point_e/examples/image2pointcloud.ipynb) - sample a point cloud, conditioned on some example synthetic view images.
 * [text2pointcloud.ipynb](point_e/examples/text2pointcloud.ipynb) - use our small, worse quality pure text-to-3D model to produce 3D point clouds directly from text descriptions. This model's capabilities are limited, but it does understand some simple categories and colors.
 * [pointcloud2mesh.ipynb](point_e/examples/pointcloud2mesh.ipynb) - try our SDF regression model for producing meshes from point clouds.

For our P-FID and P-IS evaluation scripts, see:

 * [evaluate_pfid.py](point_e/evals/scripts/evaluate_pfid.py)
 * [evaluate_pis.py](point_e/evals/scripts/evaluate_pis.py)

For our Blender rendering code, see [blender_script.py](point_e/evals/scripts/blender_script.py)

# Samples

You can download the seed images and point clouds corresponding to the paper banner images [here](https://openaipublic.azureedge.net/main/point-e/banner_pcs.zip).

You can download the seed images used for COCO CLIP R-Precision evaluations [here](https://openaipublic.azureedge.net/main/point-e/coco_images.zip).


# Project Structure

This project has been organized into a clean, production-ready structure:

```
├── docs/                    # Documentation and guides
├── examples/               # Demo scripts and Jupyter notebooks
├── models/                 # Model checkpoints and cached data
├── point_e/               # Main Point-E package
├── results/               # Output directories and results
├── scripts/               # Utility scripts and pipelines
├── tests/                 # Test files and validation scripts
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

## Directory Contents

- **docs/**: Documentation, guides, and reports
- **examples/**: Demo scripts, notebooks, and example usage
- **models/**: Pre-trained model checkpoints and cached data
- **results/**: Generated point clouds, logs, and comparison outputs
- **scripts/**: Production scripts and optimized pipelines
- **tests/**: Unit tests, integration tests, and validation scripts

# How to Run
You can run the code exactly as you would with the original Point-E version. No changes are required in the execution commands or workflow.

Or follow below steps
1. Open Visual Studio Code and select the project folder(Point-e).
2. Open the terminal by going to View > Terminal, then create and activate a virtual environment.
3. Build and launch JupyterLab by running the following command in the terminal, which will open the interface in your browser.
   jupyter lab build
   jupyter lab
4. Select either text2pointcloud.ipynb or image2pointcloud.ipynb and run it using the JupyterLab interface to see the results directly.



