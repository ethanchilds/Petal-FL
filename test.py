import time


class SleepyDataLoaderWrapper:
    def __init__(self, dataloader, sleep_sec):
        self.dataloader = dataloader
        self.sleep_sec = sleep_sec

    def __iter__(self):
        # Return a new iterator for each epoch loop
        return _SleepyIterator(self.dataloader, self.sleep_sec)

class _SleepyIterator:
    def __init__(self, dataloader, sleep_sec):
        self.iter_loader = iter(dataloader)
        self.sleep_sec = sleep_sec
        self._slept = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iter_loader)
        except StopIteration:
            if not self._slept:
                print("sleeping")
                time.sleep(self.sleep_sec)  # Sleep only once per epoch
                self._slept = True
            raise StopIteration
