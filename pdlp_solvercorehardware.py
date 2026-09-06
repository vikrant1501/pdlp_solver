import cupy as cp

class HardwareManager:
    """Monitors and manages CUDA GPU resources for CuPy execution."""
    @staticmethod
    def get_device_info():
        dev = cp.cuda.Device()
        mem_info = dev.mem_info
        free_mb = mem_info[0] / (1024 ** 2)
        total_mb = mem_info[1] / (1024 ** 2)
        return {"device_id": dev.id, "free_vram_mb": free_mb, "total_vram_mb": total_mb}

    @staticmethod
    def synchronize():
        cp.cuda.Stream.null.synchronize()