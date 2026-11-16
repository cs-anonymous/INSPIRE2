"""Fixed cuts, compact BEAT vocabulary, feedback and partial-update gradients."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import TrainingArguments

from src.data_process.dinr_beat_dataset import DINRBeatCollator, fixed_event_spans
from src.evaluate.evaluate_dinr_beat import event_chunks, predict_many, predict, token_metrics
from src.model.dinr_beat import DINRBeatModel
from src.model.pianoformer import PianoT5GemmaConfig
from src.train.train_dinr_beat import BeatTrainer
from script.run_exp1116_beat_queue import Supervisor1116


class Simulated1116(Supervisor1116):
    def stage(self, settings):
        self.add(settings)
        for job in self.manifest['jobs']:
            if job['status']!='completed':
                job.update(status='completed',selection_score=float(job['seed'])/100)
        self.save()

    def report(self):
        pass


class Tests1116(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, vocab=2):
        torch.manual_seed(42)
        config=PianoT5GemmaConfig(encoder_layers_num=1,decoder_layers_num=1,hidden_size=32,
            intermediate_size=64,num_attention_heads=2,num_key_value_heads=1,head_dim=16)
        return DINRBeatModel(config,decoder_mask_probability=0,use_beat_type_embedding=False,
                             measure_start_vocab_size=vocab)

    def test_arbitrary_cuts_and_pad_overflow(self):
        self.assertEqual(fixed_event_spans(1030),[(0,512),(512,1024),(1024,1030)])
        events=torch.tensor([[0,-1,500,-1,-1,1]]+[[1,60,250,.2,64,-1]]*7,dtype=torch.float32)
        self.assertEqual(event_chunks(events,max_events=3),[(0,3),(3,6),(6,8)])
        self.assertEqual(int(events[3,0]),1)  # The second chunk starts mid-beat.
        example=dict(encoder_tokens=torch.zeros(3,6,dtype=torch.long),
            decoder_tokens=torch.zeros(3,6,dtype=torch.long),event_types=torch.ones(3,dtype=torch.long),
            labels=torch.zeros(3,4,dtype=torch.long))
        batch=DINRBeatCollator(512)([example])
        self.assertEqual(batch['encoder_tokens'].shape,(1,512,6))
        self.assertEqual(int(batch['attention_mask'].sum()),3)
        self.assertTrue((batch['labels'][:,3:]==-100).all())
        with self.assertRaises(ValueError): DINRBeatCollator(2)([example])

    def test_beat_embedding_is_the_only_beat_identity(self):
        for size in (1,2):
            model=self.model(size)
            tokens=torch.tensor([[[0,0,100,0,0,0],[0,0,100,0,0,1]]])
            result=model._sum_events(tokens)
            ids=torch.tensor([[0,min(1,size-1)]])
            expected=model.duration_embedding(tokens[...,2])+model.beat_embedding(ids)
            torch.testing.assert_close(result,expected)
            self.assertEqual(model.beat_embedding.num_embeddings,size)
            self.assertFalse(hasattr(model,'measure_start_embedding'))
            if size==1: torch.testing.assert_close(result[:,0],result[:,1])
            else: self.assertFalse(torch.equal(result[:,0],result[:,1]))

    def test_note_first_generation_and_missing_event_metrics(self):
        events=torch.tensor([[0,-1,500,-1,-1,1],[1,60,250,0,64,-1],
            [1,64,250,.2,64,-1],[1,67,250,.3,64,-1],[0,-1,500,-1,-1,0],
            [1,70,250,.5,64,-1],[1,72,250,.7,64,-1]],dtype=torch.float32)
        targets=torch.tensor([[500,-1,-1],[200,0,72],[250,.2,68],[200,.3,70],
                              [550,-1,-1],[200,.5,80],[250,.7,70]],dtype=torch.float32)
        run=dict(duration_target='absolute_log',beat_duration_target='absolute_log',
            note_duration_target='absolute_log',duration_input_tables='shared',measure_start=True,
            window_events=3,fixed_event_length=8)
        model=self.model().eval()
        first=predict_many(model,events,targets,run,torch.device('cpu'),[123,456])
        changed=predict_many(model,events,torch.ones_like(targets)*42,run,torch.device('cpu'),[123,456])
        for a,b in zip(first,changed):
            self.assertEqual(a.shape,(7,3));self.assertTrue(torch.equal(a,b))
        tf=predict(model,events,targets,run,torch.device('cpu'),123,teacher_forced=True)
        self.assertEqual(tf.shape,(7,3))
        # A note-only tail has no meaningful beat metric, but note metrics
        # remain finite; no made-up beat anchor or JS is reported.
        note_metrics=token_metrics(tf[1:3],tf[1:3],events[1:3],targets[1:3].numpy(),targets[1:3].numpy())
        self.assertNotIn('beat_duration_log_mae',note_metrics)
        self.assertNotIn('beat_anchor_mae_ms',note_metrics)
        self.assertEqual(note_metrics['onset_mae'],0)
        self.assertTrue(all(np.isfinite(v) for v in note_metrics.values()))

    def test_partial_accumulation_matches_actual_microbatch_average(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(2.))
            def forward(self,x):
                return {'loss':((self.weight*x)**2).mean()}
        with tempfile.TemporaryDirectory() as directory:
            model=Toy()
            args=TrainingArguments(output_dir=directory,use_cpu=True,gradient_accumulation_steps=2,
                report_to='none',disable_tqdm=True)
            trainer=BeatTrainer(model=model,args=args)
            trainer.normalize_partial_accumulation=True
            for inputs in ([{'x':torch.tensor([1.,2.])},{'x':torch.tensor([3.,4.])}],
                           [{'x':torch.tensor([1.,2.])}]):
                model.zero_grad();model.train()
                batches,items=trainer.get_batch_samples(iter(inputs),2,torch.device('cpu'))
                logged=sum(float(trainer.training_step(model,b,items)) for b in batches)
                expected_loss=sum(float(((2*b['x'])**2).mean()) for b in batches)/len(batches)
                expected_grad=sum(float((4*b['x']**2).mean()) for b in batches)/len(batches)
                self.assertAlmostEqual(logged,expected_loss,places=5)
                self.assertAlmostEqual(float(model.weight.grad),expected_grad,places=5)

    def test_ten_single_training_jobs_without_C(self):
        with tempfile.TemporaryDirectory() as directory:
            manager=Simulated1116(Path(directory),[0,1,2]);manager.campaign()
            self.assertEqual(len(manager.manifest['jobs']),10)
            self.assertTrue(all(len(s['seeds'])==1 for s in manager.manifest['settings'].values()))
            self.assertFalse(any(j['setting'].startswith('C') for j in manager.manifest['jobs']))
            self.assertEqual(manager.manifest['status'],'completed')
            self.assertEqual(manager.manifest['settings']['D4']['duration_output_head'],'shared_conditioned')
            before=manager.manifest['decisions']
            resumed=Simulated1116(Path(directory),[0,1,2]);resumed.campaign()
            self.assertEqual(len(resumed.manifest['jobs']),10)
            self.assertEqual(before,resumed.manifest['decisions'])


if __name__=='__main__':unittest.main()
