import pytorch_lightning as pl
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)  # or DEBUG for more output

# --------------------- LIGHTENING MODULES  ---------------------

# call back for setting epoch for dataloader
class SetEpochCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.pl_module = pl_module
        dataloader = trainer.train_dataloader
        dataloader.batch_sampler.set_epoch(trainer.current_epoch) 
        # I am still unsure about this bit, lets see how it unrols on testing /KACPER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #if isinstance(dataloader, dict):
        #    for dl in dataloader.values():
        #        if hasattr(dl.batch_sampler, "set_epoch"):
        #            dl.batch_sampler.set_epoch(trainer.current_epoch)
        #else:
        #    if hasattr(dataloader.batch_sampler, "set_epoch"):
                

