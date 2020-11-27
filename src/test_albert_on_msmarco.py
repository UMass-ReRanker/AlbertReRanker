import pytorch_lightning as pl
from AlbertReRanker import AlbertReRanker
import sys
import argparse
import torch

def main(hparams):
    model = AlbertReRanker(hparams)
    checkpoint = torch.load(hparams.checkpoint_path, map_location=lambda storage, loc: storage)
    if 'model.albert.embeddings.position_ids' in checkpoint['state_dict']:
        del checkpoint['state_dict']['model.albert.embeddings.position_ids']
    model.load_state_dict(checkpoint['state_dict'])    
    trainer = pl.Trainer()
    trainer.test(model)
    
if __name__=='__main__':

    
    parser = argparse.ArgumentParser(description='Albert-MARCO')
    # MODEL SPECIFIC
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum number of wordpieces of the sequence")
    parser.add_argument("--lr", type=float, default=3e-6,
                        help="Learning rate")
    parser.add_argument("--num_warmup_steps", type=int, default=2500)
    parser.add_argument("--num_training_steps", type=int, default=120000)
    parser.add_argument("--val_check_interval", type=int, default=20000,
                        help='Run through dev set every N steps')
    parser.add_argument("--clf_dropout", type=float, default=-1.0,
                        help='Dropout for classifier. Set negative to use transformer dropout')
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Num subprocesses for DataLoader")

    # EXPERIMENT SPECIFIC
    parser.add_argument("--data_dir", type=str, default='../data/',)
    parser.add_argument("--dataset", type=str, default='document',
                        help="`passage` or `document` re-ranking on MS MARCO")
    # effective batch size will be: 
    # trainer_batch_size * data_loader_bs
    parser.add_argument("--trainer_batch_size", type=int, default=5,
                        help='Batch size for Trainer. Accumulates grads every k batches')
    parser.add_argument("--data_loader_bs", type=int, default=1,
                        help='Batch size for DataLoader object')
    parser.add_argument("--val_data_loader_bs", type=int, default=0,
                        help='Batch size for validation data loader. If not specified,\
                        --data_loader_bs is used.')
    parser.add_argument("--use_10_percent_of_dev", type=int, default=1,
                        help='0 to use the full dev dataset, else to use 10% only')
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--slurm_job_id", type=int, default=1)

    # Distributed training
    parser.add_argument("--gpus", type=int, default=1, help="Num of GPUs per node")
    parser.add_argument("--num_nodes", type=int, default=1, help="Num nodes allocated by SLURM")
    parser.add_argument("--distributed_backend", type=str, default='ddp',
                        help="Use distributed backend: dp/ddp/ddp2")
    parser.add_argument("--checkpoint_path", type=str, default='checkpoints.ckpt',
                        help="Absolute path to checkpoint of trained model")
    hparams = parser.parse_args()
    if hparams.val_data_loader_bs <= 0:
        hparams.val_data_loader_bs = hparams.data_loader_bs
    print(hparams)
    main(hparams)

