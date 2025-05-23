import time

class SleepInjector:
    def __init__(self, dataloader, sleep_sec):
        self.dataloader = dataloader
        self.sleep_sec = sleep_sec
    
    def __iter__(self):
        time.sleep(self.sleep_sec)
        return iter(self.dataloader)

