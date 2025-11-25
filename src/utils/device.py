"""Device detection and management for PyTorch and FAISS"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "mps", "cpu"]


def get_torch_device() -> str:
    """Get the best available PyTorch device (cuda > mps > cpu)"""
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("Using CUDA device")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("Using MPS device (Apple Silicon)")
            return "mps"
        else:
            logger.info("Using CPU device")
            return "cpu"
    except ImportError:
        logger.warning("PyTorch not available, defaulting to CPU")
        return "cpu"


def get_device_type() -> DeviceType:
    """Get the device type as a typed literal"""
    return get_torch_device()  # type: ignore


def is_gpu_available() -> bool:
    """Check if any GPU acceleration is available (CUDA or MPS)"""
    try:
        import torch

        return torch.cuda.is_available() or torch.backends.mps.is_available()
    except ImportError:
        return False


def is_cuda_available() -> bool:
    """Check if CUDA is available"""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available"""
    try:
        import torch

        return torch.backends.mps.is_available()
    except ImportError:
        return False


def get_faiss_index(dimension: int, use_gpu: bool = True):
    """
    Create a FAISS IndexFlatL2 with GPU support if available.
    
    For MPS: FAISS doesn't support MPS directly, so we use CPU.
    For CUDA: Uses GPU resources if faiss-gpu is installed.
    """
    import faiss

    index = faiss.IndexFlatL2(dimension)

    if not use_gpu:
        return index

    # Try CUDA GPU first
    if is_cuda_available():
        try:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("Using FAISS with CUDA GPU")
            return gpu_index
        except Exception as e:
            logger.warning(f"FAISS GPU not available: {e}")

    # MPS note: FAISS doesn't support MPS, use CPU
    if is_mps_available():
        logger.info("FAISS using CPU (MPS not supported by FAISS)")

    return index


def get_faiss_clustering(dimension: int, n_clusters: int, use_gpu: bool = True):
    """
    Create FAISS clustering with GPU support if available.
    
    Returns: (clustering, index) tuple
    """
    import faiss

    quantizer = faiss.IndexFlatL2(dimension)
    clustering = faiss.Clustering(dimension, n_clusters)
    clustering.niter = 20
    clustering.verbose = True

    # Try CUDA GPU
    if use_gpu and is_cuda_available():
        try:
            res = faiss.StandardGpuResources()
            gpu_quantizer = faiss.index_cpu_to_gpu(res, 0, quantizer)
            logger.info("Using FAISS clustering with CUDA GPU")
            return clustering, gpu_quantizer
        except Exception as e:
            logger.warning(f"FAISS GPU clustering not available: {e}")

    if is_mps_available():
        logger.info("FAISS clustering using CPU (MPS not supported)")

    return clustering, quantizer


def get_device_info() -> dict:
    """Get detailed device information"""
    info = {
        "device": get_torch_device(),
        "cuda_available": is_cuda_available(),
        "mps_available": is_mps_available(),
        "gpu_available": is_gpu_available(),
    }

    try:
        import torch

        info["torch_version"] = torch.__version__

        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    return info
