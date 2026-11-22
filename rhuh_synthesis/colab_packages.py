"""Packages the Colab notebooks install before they import ``rhuh_synthesis``.

One list for every notebook, so that none misses a package the code imports. A notebook installs ``DATA`` for CPU
work and ``DATA + MODEL`` for work with the MedSegFactory model:

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *DATA], check=True)

The packages in ``PROVIDED`` are imported by the code but not installed: past Colab runs imported them without
installing them. The others are installed even where Colab may already have them, so that each run gets the pinned
or bounded version. A local test checks that every third-party import of the code is in one of the lists.

This module imports nothing, and neither does the package's ``__init__``, so a notebook can read the lists before
anything is installed.
"""

# CPU work: data, geometry, registration and manifests. ``manifest`` imports pydicom through ``rhuh_qc.dicom``.
DATA = ("nibabel>=5.3,<6", "SimpleITK==2.5.6", "pydicom>=3,<4")

# GPU work with the MedSegFactory model, pinned as in the Stage 5 notebooks.
MODEL = ("diffusers==0.40.0", "transformers==5.17.0", "accelerate==1.15.0", "scikit-image>=0.25")

# Imported by the code and provided by Colab (import names, not package names).
PROVIDED = ("numpy", "scipy", "pandas", "matplotlib", "PIL", "torch", "googleapiclient", "google")
