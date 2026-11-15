"""Verify the complete adaptive campaign without using GPUs or real results."""
import tempfile
import unittest
from pathlib import Path

from script.run_dinr_beat_queue import Supervisor


class SimulatedSupervisor(Supervisor):
    def stage(self, settings):
        self.add(settings)
        for job in self.manifest['jobs']:
            if job['status'] != 'completed':
                job.update(status='completed',selection_score=float(job['seed'])/100)
        self.save()

    def report(self):
        pass


class QueueTests(unittest.TestCase):
    def test_all_36_trainings_and_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            supervisor=SimulatedSupervisor(Path(directory),[0,1,2])
            supervisor.campaign()
            jobs=supervisor.manifest['jobs']
            self.assertEqual(len(jobs),36)
            self.assertEqual(len({j['id'] for j in jobs}),36)
            self.assertFalse(any(j['setting']=='A0' for j in jobs))
            self.assertEqual(supervisor.manifest['status'],'completed')
            settings=supervisor.manifest['settings']
            self.assertEqual(len(settings),12)
            self.assertEqual(settings['D1']['beat_duration_target'],'absolute_log')
            self.assertEqual(settings['D1']['note_duration_target'],'log_dev')
            self.assertEqual(settings['D2']['beat_duration_target'],'log_dev')
            self.assertEqual(settings['D2']['note_duration_target'],'absolute_log')
            self.assertEqual(settings['C2']['duration_output_head'],'shared_conditioned')
            self.assertEqual(settings['C3']['duration_output_head'],'shared')
            self.assertEqual(settings['D3']['seeds'],[45,46,47])
            self.assertEqual(settings['D4']['duration_input_tables'],'separate')
            source=settings[settings['D5']['source']]
            self.assertNotEqual(settings['D5']['measure_start'],source['measure_start'])
            for decision in supervisor.manifest['decisions'].values():
                self.assertIn('no test',decision['criterion'])
            # Resuming a completed manifest reuses decisions and run IDs.
            resumed=SimulatedSupervisor(Path(directory),[0,1,2])
            resumed.campaign()
            self.assertEqual(len(resumed.manifest['jobs']),36)
            self.assertEqual(resumed.manifest['decisions'],supervisor.manifest['decisions'])


if __name__=='__main__':unittest.main()
