from ..common import DeviceOpOverrides

class CUDADeviceOpOverrides(DeviceOpOverrides):
    @classmethod
    def import_get_raw_stream_as(self, name):
        return f"from torch._C import _cuda_getCurrentRawStream as {name}"

    @classmethod
    def set_device(self, device_idx):
        return f"torch.cuda.set_device({device_idx})"

    @classmethod
    def synchronize(self):
        return "torch.cuda.synchronize()"

    @classmethod
    def DeviceGuard(self, device_idx):
        return f"torch.cuda._DeviceGuard({device_idx})"
