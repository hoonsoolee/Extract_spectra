"""
data_loader.py
--------------
Hyperspectral file loader supporting ENVI, GeoTIFF, HDF5 formats.
Sources: local folder or GitHub repository.
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Type alias
HsiData = Tuple[np.ndarray, Dict[str, Any]]   # (data H×W×B, metadata)
# ------------------------------------------------------------------ #


class HyperspectralLoader:
    """Load hyperspectral files from a local folder or GitHub repository."""

    SUPPORTED_EXTS = {".hdr", ".tif", ".tiff", ".h5", ".hdf5", ".mat"}

    def __init__(self, config: dict):
        self.config = config
        self.cache_dir = Path(config.get("cache_dir", "./cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Public: file discovery
    # ============================================================

    def list_local_files(self, folder: str) -> List[Path]:
        """Return sorted list of supported hyperspectral files in *folder*."""
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Local folder not found: {folder}")

        files: List[Path] = []
        for ext in self.SUPPORTED_EXTS:
            # Collect header/main files; skip companion .raw/.bil/.bip/.bsq etc.
            found = sorted(folder.rglob(f"*{ext}"))
            files.extend(found)

        # For ENVI: only keep .hdr files (companion binary is loaded automatically)
        hdr_stems = {f.stem for f in files if f.suffix.lower() == ".hdr"}
        files = [
            f for f in files
            if not (f.suffix.lower() in {".raw", ".bil", ".bip", ".bsq"}
                    and f.stem in hdr_stems)
        ]
        return sorted(set(files))

    def list_github_files(
        self,
        repo: str,
        folder: str = "",
        token: Optional[str] = None,
    ) -> List[str]:
        """Return list of supported file paths inside a GitHub repository folder."""
        try:
            from github import Github, GithubException
        except ImportError:
            raise ImportError("PyGithub is required for GitHub access: pip install PyGithub")

        gh_token = token or os.environ.get("GITHUB_TOKEN")
        g = Github(gh_token) if gh_token else Github()

        try:
            repository = g.get_repo(repo)
        except Exception as e:
            raise RuntimeError(f"Cannot access GitHub repo '{repo}': {e}")

        paths: List[str] = []
        queue = [folder] if folder else [""]
        while queue:
            current = queue.pop(0)
            try:
                items = repository.get_contents(current)
                if not isinstance(items, list):
                    items = [items]
                for item in items:
                    if item.type == "dir":
                        queue.append(item.path)
                    elif any(item.name.lower().endswith(ext) for ext in self.SUPPORTED_EXTS):
                        paths.append(item.path)
            except Exception as e:
                logger.warning(f"Cannot list GitHub folder '{current}': {e}")

        return sorted(paths)

    # ============================================================
    # Public: loading
    # ============================================================

    def load_local(self, path: str | Path) -> HsiData:
        """Load a hyperspectral file from a local path."""
        path = Path(path)
        ext = path.suffix.lower()
        logger.info(f"Loading local file: {path.name}")

        if ext == ".hdr":
            return self._load_envi(path)
        elif ext in {".tif", ".tiff"}:
            return self._load_tiff(path)
        elif ext in {".h5", ".hdf5"}:
            return self._load_hdf5(path)
        elif ext == ".mat":
            return self._load_mat(path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

    def load_github(
        self,
        repo: str,
        file_path: str,
        token: Optional[str] = None,
    ) -> HsiData:
        """Download a file from GitHub (cached locally) and load it."""
        local_path = self._download_github_file(repo, file_path, token)
        return self.load_local(local_path)

    # ============================================================
    # Private: format-specific loaders
    # ============================================================

    def _load_envi(self, hdr_path: Path) -> HsiData:
        try:
            import spectral
        except ImportError:
            raise ImportError("spectral (SPy) required: pip install spectral")

        img = spectral.open_image(str(hdr_path))
        data = np.array(img.load(), dtype=np.float32)  # (H, W, B)

        wavelengths = None
        if img.bands and img.bands.centers:
            wavelengths = list(img.bands.centers)

        metadata = {
            "format": "ENVI",
            "filename": hdr_path.name,
            "shape": data.shape,
            "wavelengths": wavelengths,
            "n_bands": data.shape[2],
        }
        logger.info(f"  ENVI loaded: {data.shape}, wavelengths={'yes' if wavelengths else 'no'}")
        return data, metadata

    def _load_tiff(self, path: Path) -> HsiData:
        try:
            import tifffile
            img = tifffile.imread(str(path))
        except ImportError:
            from PIL import Image
            img = np.array(Image.open(path))

        # Ensure (H, W, B) shape
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        elif img.ndim == 3 and img.shape[0] < min(img.shape[1], img.shape[2]):
            img = img.transpose(1, 2, 0)  # (B, H, W) → (H, W, B)

        data = img.astype(np.float32)
        metadata = {
            "format": "GeoTIFF",
            "filename": path.name,
            "shape": data.shape,
            "wavelengths": None,
            "n_bands": data.shape[2],
        }
        logger.info(f"  TIFF loaded: {data.shape}")
        return data, metadata

    def _load_hdf5(self, path: Path) -> HsiData:
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required: pip install h5py")

        with h5py.File(path, "r") as f:
            # Try common dataset names
            data_key = None
            for candidate in ["reflectance", "radiance", "data", "hypercube", "cube"]:
                if candidate in f:
                    data_key = candidate
                    break
            if data_key is None:
                data_key = list(f.keys())[0]
                logger.warning(f"  HDF5: using first dataset '{data_key}'")

            data = f[data_key][:]

            # Wavelengths
            wl = None
            for wl_key in ["wavelengths", "wavelength", "wl", "bands"]:
                if wl_key in f:
                    wl = list(f[wl_key][:])
                    break

        # Ensure (H, W, B)
        if data.ndim == 3 and data.shape[0] < data.shape[1]:
            data = data.transpose(1, 2, 0)

        data = data.astype(np.float32)
        metadata = {
            "format": "HDF5",
            "filename": path.name,
            "shape": data.shape,
            "wavelengths": wl,
            "n_bands": data.shape[2],
        }
        logger.info(f"  HDF5 loaded: {data.shape}, wavelengths={'yes' if wl else 'no'}")
        return data, metadata

    def _load_mat(self, path: Path) -> HsiData:
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy required: pip install scipy")

        mat = loadmat(str(path))
        # Filter out MATLAB metadata keys
        data_keys = [k for k in mat if not k.startswith("_")]
        if not data_keys:
            raise ValueError(f"No data arrays found in .mat file: {path.name}")

        # Pick the largest array
        data_key = max(data_keys, key=lambda k: mat[k].size)
        data = mat[data_key].astype(np.float32)

        # Ensure (H, W, B)
        if data.ndim == 3 and data.shape[0] < data.shape[1]:
            data = data.transpose(1, 2, 0)

        wl = mat.get("wavelengths", mat.get("wl", None))
        if wl is not None:
            wl = wl.flatten().tolist()

        metadata = {
            "format": "MATLAB",
            "filename": path.name,
            "shape": data.shape,
            "wavelengths": wl,
            "n_bands": data.shape[2],
        }
        logger.info(f"  MAT loaded: {data.shape}")
        return data, metadata

    # ============================================================
    # Private: GitHub download
    # ============================================================

    def _download_github_file(
        self,
        repo: str,
        file_path: str,
        token: Optional[str] = None,
    ) -> Path:
        """Download a file from GitHub to the local cache directory."""
        import requests as req
        try:
            from github import Github
        except ImportError:
            raise ImportError("PyGithub required: pip install PyGithub")

        cache_path = self.cache_dir / Path(file_path).name
        if cache_path.exists():
            logger.info(f"  Using cached: {cache_path.name}")
            return cache_path

        gh_token = token or os.environ.get("GITHUB_TOKEN")
        g = Github(gh_token) if gh_token else Github()
        repository = g.get_repo(repo)
        contents = repository.get_contents(file_path)

        logger.info(f"  Downloading from GitHub: {file_path}")
        response = req.get(contents.download_url, stream=True)
        response.raise_for_status()

        with open(cache_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"  Saved to: {cache_path}")
        return cache_path
