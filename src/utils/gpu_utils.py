"""src/utils/gpu_utils.py — device selection and memory helpers."""

import logging
logger = logging.getLogger(__name__)


def get_device(prefer_gpu: bool = True):
    """Return the best available torch device."""
    try:
        import torch
        if prefer_gpu:
            if torch.cuda.is_available():
                dev = torch.device("cuda")
                name = torch.cuda.get_device_name(0)
                mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"Using GPU: {name}  ({mem:.1f} GB VRAM)")
                return dev
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Using Apple MPS backend")
                return torch.device("mps")
        logger.info("Using CPU")
        return torch.device("cpu")
    except ImportError:
        logger.warning("PyTorch not installed — returning 'cpu' string")
        return "cpu"


def gpu_memory_summary() -> str:
    """Return a human-readable GPU memory summary string."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "No CUDA device"
        alloc   = torch.cuda.memory_allocated()  / 1024**2
        reserved = torch.cuda.memory_reserved()  / 1024**2
        total   = torch.cuda.get_device_properties(0).total_memory / 1024**2
        return (f"GPU memory — allocated: {alloc:.0f} MB  "
                f"reserved: {reserved:.0f} MB  total: {total:.0f} MB")
    except Exception as e:
        return f"GPU info unavailable: {e}"


def clear_gpu_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    except ImportError:
        pass
