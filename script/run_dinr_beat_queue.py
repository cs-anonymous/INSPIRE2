#!/usr/bin/env python3
"""Persistent three-GPU beat experiment supervisor with validation-only dependencies.

No A0 training. 15 first-stage runs, six output-head runs, and 15 interaction/
confirmation runs = 36 trainings. Restarting this script resumes ready artifacts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/'data/ASAP_processed_dinr_beat_v1'
RUN=ROOT.parent/'results/exp1115_beat'
SEEDS=(42,43,44)


def now():
    return datetime.now().astimezone().isoformat()


def atomic(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def spec(identifier,target='absolute_log',tables='shared',head='separate',measure=True,
         beat=None,note=None,seeds=SEEDS,source=None,onset_target='ratio'):
    return dict(id=identifier,duration_target=target,beat_duration_target=beat or target,
                note_duration_target=note or target,duration_input_tables=tables,
                duration_output_head=head,measure_start=measure,seeds=list(seeds),source=source,
                onset_target=onset_target)


BASE=[spec('A1'),spec('A2',target='log_dev'),spec('A3',measure=False),
      spec('B1',tables='separate'),spec('B2',target='log_dev',tables='separate')]


class Supervisor:
    def __init__(self,run_dir,gpus,*,data_dir=DATA,sidecar_tag='b512_o50',batch_size=8,
                 accumulation=4,fixed_event_length=None,normalize_partial=False,
                 beat_type_embedding=True,experiment='exp1115'):
        self.run_dir,self.gpus=run_dir,gpus
        self.data_dir,self.sidecar_tag=data_dir,sidecar_tag
        self.batch_size,self.accumulation=batch_size,accumulation
        self.fixed_event_length=fixed_event_length
        self.normalize_partial=normalize_partial
        self.beat_type_embedding=beat_type_embedding
        self.experiment=experiment
        self.lock=threading.RLock()
        self.manifest_path=run_dir/'manifest.json'
        self.manifest=(json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else
                       dict(created_at=now(),status='running',jobs=[],settings={},decisions={}))
        self.manifest.update(supervisor_pid=os.getpid(),gpus=gpus,status='running',updated_at=now())
        self.report_lock=threading.Lock()
        self.last_report=0.0
        self.save()

    def save(self):
        with self.lock:
            self.manifest['updated_at']=now()
            atomic(self.manifest_path,self.manifest)

    def report(self):
        if self.experiment == 'exp1117':
            return
        if time.monotonic()-self.last_report < 60:
            return
        if not self.report_lock.acquire(blocking=False):
            return
        try:
            self.last_report=time.monotonic()
            with (self.run_dir/'report.log').open('a') as log:
                report_command=[sys.executable,str(ROOT/'script/report_dinr_beat.py'),
                                '--campaign',str(self.run_dir),'--experiment',self.experiment]
                if self.experiment == 'exp1116':
                    report_command.extend(['--doc-path','docs/exp1115_BEAT.md'])
                result=subprocess.run(report_command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    print('Report generation failed; see report.log',flush=True)
        finally:
            self.report_lock.release()

    def add(self,settings):
        self.manifest['settings'].update({s['id']:s for s in settings})
        # Alternate settings before seeds, so all three GPUs give early evidence
        # about different configurations while preserving identical seed cohorts.
        max_seeds=max(len(s['seeds']) for s in settings)
        existing={j['id'] for j in self.manifest['jobs']}
        for seed_index in range(max_seeds):
            for s in settings:
                if seed_index>=len(s['seeds']):
                    continue
                seed=s['seeds'][seed_index]
                identifier=f"{s['id']}_seed{seed}"
                if identifier not in existing:
                    self.manifest['jobs'].append(dict(id=identifier,setting=s['id'],seed=seed,
                         run_dir=str(self.run_dir/identifier),status='queued',config=deepcopy(s)))
        self.save()

    def execute(self,job,gpu):
        directory=Path(job['run_dir'])
        directory.mkdir(parents=True,exist_ok=True)
        job.pop('error', None)
        job.pop('traceback', None)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
             'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
        s=job['config']
        def command_run(command,phase,logname,attempt=0):
            with self.lock:
                job.update(status=phase,gpu=gpu,updated_at=now(),command=command)
                self.save()
            with (directory/logname).open('a') as log:
                log.write(f'\n{now()} attempt={attempt} command={command!r}\n');log.flush()
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                with self.lock:
                    job.update(child_pid=process.pid)
                    self.save()
                while process.poll() is None:
                    time.sleep(15)
                    with self.lock:
                        job['heartbeat']=now()
                        self.save()
                    self.report()
                if process.returncode:
                    raise RuntimeError(f'{phase} exit={process.returncode}; see {directory/logname}')
        try:
            summary_path=directory/'training_summary.json'
            if not summary_path.exists():
                batch,accumulation=self.batch_size,self.accumulation
                for attempt in range(2):
                    command=[sys.executable,'-u','-m','src.train.train_dinr_beat','--output-dir',str(directory),
                             '--train-sidecar',str(self.data_dir/f'dinr_beat_train_{self.sidecar_tag}.pt'),
                             '--valid-sidecar',str(self.data_dir/f'dinr_beat_valid_{self.sidecar_tag}.pt'),
                             '--epochs','16','--learning-rate','3e-4','--batch-size',str(batch),
                             '--gradient-accumulation-steps',str(accumulation),'--seed',str(job['seed']),
                             '--early-stopping-patience','4','--duration-target',s['duration_target'],
                             '--beat-duration-target',s['beat_duration_target'],
                             '--note-duration-target',s['note_duration_target'],
                             '--duration-input-tables',s['duration_input_tables'],
                             '--duration-output-head',s['duration_output_head'],
                             '--onset-target',s.get('onset_target','ratio'),
                             '--measure-start' if s['measure_start'] else '--no-measure-start']
                    if self.fixed_event_length:
                        command.extend(['--fixed-event-length',str(self.fixed_event_length)])
                    if self.normalize_partial:
                        command.append('--normalize-partial-accumulation')
                    if not self.beat_type_embedding:
                        command.append('--no-beat-type-embedding')
                    try:
                        command_run(command,'training','train.log',attempt)
                        break
                    except RuntimeError:
                        log=(directory/'train.log').read_text(errors='replace')
                        if self.experiment == 'exp1115' and attempt==0 and ('out of memory' in log.lower()):
                            batch,accumulation=4,8
                            job['oom_retry']='batch 4 x accumulation 8; effective batch remains 32'
                            continue
                        raise
            summary=json.loads(summary_path.read_text())
            if not (directory/'best/pytorch_model.bin').exists():
                raise RuntimeError('Training summary exists but best checkpoint is missing')
            with self.lock:
                job.update(training=summary)
                self.save()
            for split in ('valid','test'):
                output=directory/('validation' if split=='valid' else 'test')
                ready=output/'metrics.json'
                if ready.exists():
                    continue
                command=[sys.executable,'-u','-m','src.evaluate.evaluate_dinr_beat',
                         '--checkpoint',str(directory/'best'),'--sidecar',str(self.data_dir/f'dinr_beat_{split}_{self.sidecar_tag}.pt'),
                         '--output-dir',str(output),'--split',split,'--samples','3']
                for attempt in range(2):
                    try:
                        command_run(command,'validating' if split=='valid' else 'testing',split+'.log',attempt)
                        break
                    except RuntimeError:
                        if attempt:
                            raise
                result=json.loads(ready.read_text())
                if split=='test' and (result['works']!=19 or result['samples']!=3):
                    raise RuntimeError('Test work/draw coverage is incomplete')
            valid=json.loads((directory/'validation/metrics.json').read_text())
            with self.lock:
                job.update(status='completed',finished_at=now(),selection_score=valid['metrics']['selection_score'])
                job.pop('error',None)
                self.save()
            print(f'{now()} GPU{gpu} completed {job["id"]}',flush=True)
        except Exception as error:
            with self.lock:
                job.update(status='failed',error=str(error),traceback=traceback.format_exc(),failed_at=now())
                self.save()
            print(f'{now()} GPU{gpu} FAILED {job["id"]}: {error}',flush=True)
        self.report()

    def stage(self,settings):
        self.add(settings)
        setting_ids={s['id'] for s in settings}
        pending=[j for j in self.manifest['jobs'] if j['setting'] in setting_ids and j['status']!='completed']
        for job in pending:
            pid=job.get('child_pid')
            if pid:
                try:
                    os.kill(pid,0)
                except ProcessLookupError:
                    pass
                else:
                    # Guard PID reuse using Linux process command lines.
                    cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                    if job['run_dir'] in cmdline:
                        raise RuntimeError(f'Previous child {pid} is still active for {job["id"]}; refusing duplicate run')
        task_lock=threading.Lock()
        def worker(gpu):
            while True:
                with task_lock:
                    if not pending:
                        return
                    job=pending.pop(0)
                self.execute(job,gpu)
        with ThreadPoolExecutor(max_workers=len(self.gpus)) as pool:
            futures=[pool.submit(worker,gpu) for gpu in self.gpus]
            for future in as_completed(futures):
                future.result()
        bad=[j['id'] for j in self.manifest['jobs'] if j['setting'] in setting_ids and j['status']!='completed']
        if bad:
            raise RuntimeError(f'Stage incomplete; dependent stages cannot select configurations: {bad}')

    def best(self,identifiers,decision):
        if decision in self.manifest['decisions']:
            return deepcopy(self.manifest['decisions'][decision]['config'])
        scores={identifier:sum(j['selection_score'] for j in self.manifest['jobs'] if j['setting']==identifier)/
                sum(j['setting']==identifier for j in self.manifest['jobs']) for identifier in identifiers}
        chosen=min(scores,key=scores.get)
        result=deepcopy(self.manifest['settings'][chosen])
        self.manifest['decisions'][decision]=dict(chosen=chosen,scores=scores,config=result,
             criterion='valid greedy physical-error score, mean across three independent training seeds; no test metrics',time=now())
        self.save()
        return result

    def campaign(self):
        self.stage(BASE)
        source=self.best(['A1','A2'],'C_target')
        c2=deepcopy(source);c2.update(id='C2',duration_output_head='shared_conditioned',source=source['id'])
        c3=deepcopy(source);c3.update(id='C3',duration_output_head='shared',source=source['id'])
        self.stage([c2,c3])
        # Mixed duration targets always have independent output heads.
        self.stage([spec('D1',beat='absolute_log',note='log_dev'),
                    spec('D2',beat='log_dev',note='absolute_log')])
        pair=self.best(['A1','A2','D1','D2'],'D_duration_pair')
        d3=deepcopy(pair);d3.update(id='D3',seeds=[45,46,47],source=pair['id'])
        d4=deepcopy(pair);d4.update(id='D4',duration_input_tables='separate',
                                  duration_output_head='shared_conditioned',seeds=list(SEEDS),source=pair['id'])
        self.stage([d3,d4])
        best=self.best(['A1','A2','A3','B1','B2','C2','C3','D1','D2','D3','D4'],'D_measure_start')
        d5=deepcopy(best);d5.update(id='D5',measure_start=not best['measure_start'],
                                  seeds=[45,46,47],source=best['id'])
        self.stage([d5])
        self.manifest.update(status='completed',finished_at=now())
        self.save();self.report()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=RUN)
    parser.add_argument('--gpus',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=True)
    with (args.run_dir/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        audit=json.loads((ROOT/'analysis/design1115_beat_data_audit.json').read_text())
        if audit['status']!='passed' or audit['splits']['test']['works']!=19:
            raise RuntimeError('Data audit must pass with all 19 test works')
        for split,row in audit['splits'].items():
            if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:
                raise RuntimeError(f'{split} sidecar changed since audit')
        manager=Supervisor(args.run_dir.resolve(),args.gpus)
        atomic(args.run_dir/'data_audit.json',audit)
        if args.prepare_only:
            manager.add(BASE);manager.manifest['status']='prepared';manager.save();manager.report();return
        try:
            manager.campaign()
        except Exception:
            manager.manifest.update(status='failed',error=traceback.format_exc())
            manager.save();manager.report();raise


if __name__=='__main__':
    main()
