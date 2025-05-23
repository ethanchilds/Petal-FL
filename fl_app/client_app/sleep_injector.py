import time

class SleepInjector:
    def __init__(self, dataloader, sleep_sec):
        self.dataloader = dataloader
        self.sleep_sec = sleep_sec
        self.iter_loader = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iter_loader)
        except StopIteration:
            time.sleep(self.sleep_sec)
            self.iter_loader = iter(self.dataloader)
            raise StopIteration
