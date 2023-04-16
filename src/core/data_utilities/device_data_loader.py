from torch.utils.data import DataLoader


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, data_loader: DataLoader, device: str):
        self.dl = data_loader
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield self._to_device(b)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

    def _to_device(self, data):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list, tuple)):
            return [self._to_device(x) for x in data]
        return data.to(self.device, non_blocking=True)
