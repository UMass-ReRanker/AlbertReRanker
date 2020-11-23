import pytorch_lightning as pl
from AlbertReRanker import AlbertReRanker
import sys

model = AlbertReRanker.load_from_checkpoint(sys.argv[1])
trainer = pl.Trainer()
trainer.test(model)
    
