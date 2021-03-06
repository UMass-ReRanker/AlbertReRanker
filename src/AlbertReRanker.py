import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import get_linear_schedule_with_warmup, AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from MarcoDataset import MarcoDataset
from IPython import embed


class AlbertReRanker(pl.LightningModule):
    """
    The Model. Impelements a few functions needed by PytorchLightning to do it's magic
    Important parts: 
       __init__: initialize the model and all of it's parts
       forward: normal forward of the network
       configure_optimizers: configure optimizers
    """

    def __init__(self, hparams):
        # super().__init__()
        super(AlbertReRanker, self).__init__()
        self.hparams = hparams
        self.tokenizer = AutoTokenizer.from_pretrained(f'albert-base-v2')
        self.model = AutoModelForSequenceClassification.from_pretrained(f'albert-base-v2')

        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None
        if self.hparams.dataset == 'document':  # document re-ranking
            self.DatasetClass = MarcoDataset
        elif self.hparams.dataset == 'passage':  # passage re-ranking
            self.DatasetClass = MarcoPassages
        else:
            raise ValueError(
                'hparams.dataset must be one of: passage|document')

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs[0]

        return logits

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr,
                          betas=(0.9, 0.999), weight_decay=0.01, correct_bias=False)
        scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                                  self.hparams.num_warmup_steps,
                                                                  self.hparams.num_training_steps),
                     'interval': 'step',
                     'name': 'linear_with_warmup'}
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = self.DatasetClass(data_dir=self.hparams.data_dir,
                                    mode='train',
                                    tokenizer=self.tokenizer,
                                    max_seq_len=self.hparams.max_seq_len,
                                    args=self.hparams,
                                    )

        self.train_dataloader_object = DataLoader(
            dataset, batch_size=self.hparams.data_loader_bs,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=AlbertReRanker.collate_fn
        )
        return self.train_dataloader_object

    def val_dataloader(self):
        dataset = self.DatasetClass(data_dir=self.hparams.data_dir,
                                    mode='dev',
                                    tokenizer=self.tokenizer,
                                    max_seq_len=self.hparams.max_seq_len,
                                    args=self.hparams,
                                    )

        self.val_dataloader_object = DataLoader(
            dataset, batch_size=self.hparams.val_data_loader_bs,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=AlbertReRanker.collate_fn
        )
        return self.val_dataloader_object

    def test_dataloader(self):
        dataset = self.DatasetClass(data_dir=self.hparams.data_dir,
                                    mode='test',
                                    tokenizer=self.tokenizer,
                                    max_seq_len=self.hparams.max_seq_len,
                                    args=self.hparams,
                                    )

        self.test_dataloader_object = DataLoader(
            dataset, batch_size=self.hparams.val_data_loader_bs,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=AlbertReRanker.collate_fn
        )
        return self.test_dataloader_object

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, labels, idxs = batch
        output = self.forward(input_ids, attention_mask, token_type_ids)

        loss = F.cross_entropy(output, labels.squeeze(1))
        if self.logger:
            self.logger.log_metrics({'train_loss': loss.item()})

        # return {'loss': loss}
        return {'out': output, 'labels': labels}

    def training_step_end(self, outputs):
        out = outputs['out']
        labels = outputs['labels'].squeeze(1)
        loss = F.cross_entropy(out, labels)
        if self.logger:
            self.logger.log_metrics({'train_loss': loss.item()})

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, labels, idxs = batch
        output = self.forward(input_ids, attention_mask, token_type_ids)

        loss = F.cross_entropy(output, labels.squeeze(1))
        if self.logger:
            self.logger.log_metrics({'val_loss': loss.item()})
        return {'out': output, 'idxs': idxs, 'labels': labels}
        # return {'loss': loss, 'probs': F.softmax(output, dim=1)[:,1], 'idxs': idxs}

    def validation_step_end(self, outputs):
        """
        outputs: dict of outputs of all batches in `dp` or `ddp2` mode
        """
        out = outputs['out']
        labels = outputs['labels']
        idxs = outputs['idxs']

        loss = F.cross_entropy(out, labels.squeeze(1))
        if self.logger:
            self.logger.log_metrics({'val_loss': loss.item()})
        return {'loss': loss, 'probs': F.softmax(out, dim=1)[:, 1], 'idxs': idxs}

    def validation_epoch_end(self, outputs):
        """ 
        outputs: dict of outputs of validation_step (or validation_step_end in dp/ddp2)
        outputs['loss'] --> losses of all the batches
        outputs['probs'] --> scores for each example
        outputs['idxs'] --> indexes in Dataset to connect with scores
        """

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        mrr = self._get_mrr_score(outputs)
        mrr10 = self._get_mrr_score(outputs, k=10)

        if self.logger:
            self.logger.log_metrics({'val_epoch_loss': avg_loss,
                                     'mrr': mrr,
                                     'mrr10': mrr10})

        print(f"\nDEV:: avg-LOSS: {avg_loss} || MRR: {mrr} || MRR@10: {mrr10}")

        # ,
        return {'val_epoch_loss': avg_loss, 'mrr10': torch.tensor(mrr10),  'progress_bar': {'val_epoch_loss': avg_loss}}
        # 'log': {'val_epoch_loss': avg_loss, 'mrr': mrr, 'mrr10': mrr10}}

    def _get_mrr_score(self, outputs, k=None, mode='dev'):
        """ Calculates MRR@k (Mean Reciprocal Rank)."""
        if mode == 'dev':
            ds = self.val_dataloader_object.dataset
        elif mode == 'test':
            ds = self.test_dataloader_object.dataset

        probs, idxs = [], []
        qids, dids, labels = [], [], []
        for x in outputs:
            probs += x['probs'].tolist()
            idxs += x['idxs'].tolist()

            top100_qids = ds.top100.iloc[x['idxs'].cpu()].qid.values.tolist()
            top100_dids = ds.top100.iloc[x['idxs'].cpu()].did.values.tolist()
            for qid, did in zip(top100_qids, top100_dids):
                qids.append(qid)
                dids.append(did)
                labels.append(0 if ds.relations[(ds.relations['qid'] == qid) & (
                    ds.relations['did'] == did)].empty else 1)

        df = pd.DataFrame({'prob': probs, 'idx': idxs,
                           'qid': qids, 'did': dids, 'label': labels})
        mrr = 0.0
        for qid in df.qid.unique():
            tmp = df[df['qid'] == qid].sort_values(
                'prob', ascending=False).reset_index()
            if k:
                tmp = tmp.head(k)
            trues = tmp.index[tmp['label'] == 1].tolist()
            # if there is no relevant docs for this query
            if not trues:
                # add to total number of qids or not?
                pass
            else:
                first_relevant = trues[0] + 1  # pandas zero-indexing
                mrr += 1.0/first_relevant

        mrr /= len(df.qid.unique())
        return mrr

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([x['input_ids'] for x in batch])
        token_type_ids = torch.stack(
            [torch.tensor(x['token_type_ids']) for x in batch])
        attention_mask = torch.stack([x['attention_mask'] for x in batch])
        label = torch.stack([x['label'] for x in batch])
        idx = torch.stack([x['idx'] for x in batch])

        return (input_ids, attention_mask, token_type_ids, label, idx)

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, labels, idxs = batch
        output = self.forward(input_ids, attention_mask, token_type_ids)

        return {'probs': F.softmax(output, dim=1)[:, 1], 'idxs': idxs}

    def test_epoch_end(self, outputs):
        """ 
        outputs: dict of outputs of test_step (or test_step_end in dp/ddp2)
        outputs['loss'] --> losses of all the batches
        outputs['probs'] --> scores for each example
        outputs['idxs'] --> indexes in Dataset to connect with scores
        """

        self._store_trec_output(outputs)
        return

    def _store_trec_output(self, outputs):

        ds = self.test_dataloader_object.dataset

        probs, idxs = [], []
        qids, dids = [], []
        for x in outputs:
            probs += x['probs'].tolist()
            idxs += x['idxs'].tolist()

            top1000_qids = ds.top100.iloc[x['idxs'].cpu()].qid.values.tolist()
            top1000_dids = ds.top100.iloc[x['idxs'].cpu()].did.values.tolist()
            for qid, did in zip(top1000_qids, top1000_dids):
                qids.append(qid)
                dids.append(did)

        df = pd.DataFrame(
            {'prob': probs, 'idx': idxs, 'qid': qids, 'did': dids})
        df['Q0'] = 'Q0'
        df['run_name'] = 'albert-reranker'
        df['prank'] = df.groupby('qid')['prob'].rank(ascending=0)
        df.prank = df.prank.astype(int)
        df = df.rename(columns={"prank" : "rank"})
        df = df[['qid', 'Q0', 'did', 'rank', 'prob', 'run_name']]
        df.to_csv(f'msmarco-test-qrels.tsv',
                  sep=' ', header=False, index=False)
        return
