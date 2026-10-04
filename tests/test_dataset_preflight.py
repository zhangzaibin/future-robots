"""Ensure invalid merge requests cannot remove or partially replace a dataset."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('merger', ROOT / 'scripts/set_atom/macg/merge_multitask_datasets.py')
merger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merger)


class DatasetPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = [self.root / 'a', self.root / 'b']
        self.output = self.root / 'combined'
        for source in self.sources:
            (source / 'meta').mkdir(parents=True)
            (source / 'data/chunk-000').mkdir(parents=True)
            info = {'features': {'state': {'dtype': 'float32', 'shape': [16]}},
                    'fps': 30, 'robot_type': 'dual_arm', 'total_episodes': 1, 'total_chunks': 1}
            (source / 'meta/info.json').write_text(json.dumps(info))
            (source / 'meta/tasks.jsonl').write_text('{"task_index": 0, "task": "stack"}\n')
            (source / 'meta/episodes.jsonl').write_text('{"episode_index": 0, "length": 2}\n')
            (source / 'data/chunk-000/episode_000000.parquet').touch()

    def validate(self, **kwargs):
        return merger.validate_sources([str(p) for p in self.sources], str(self.output), **kwargs)

    def change_info(self, **kwargs):
        path = self.sources[1] / 'meta/info.json'
        info = json.loads(path.read_text())
        info.update(kwargs)
        path.write_text(json.dumps(info))

    def test_compatible_sources_validate_without_creating_output(self):
        self.assertEqual(self.validate(), [p.resolve() for p in self.sources])
        self.assertFalse(self.output.exists())

    def test_existing_output_requires_explicit_overwrite(self):
        self.output.mkdir()
        (self.output / 'keep').write_text('keep')
        with self.assertRaises(FileExistsError):
            self.validate()
        self.assertEqual((self.output / 'keep').read_text(), 'keep')
        self.assertEqual(len(self.validate(overwrite=True)), 2)

    def test_overlapping_directories_are_rejected_even_with_overwrite(self):
        for output in [self.root, self.sources[0], self.sources[0] / 'nested']:
            self.output = output
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'overlap'):
                self.validate(overwrite=True)

    def test_incompatible_fps_is_rejected_before_overwrite(self):
        self.output.mkdir()
        (self.output / 'keep').write_text('keep')
        self.change_info(fps=60)
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            self.validate(overwrite=True)
        self.assertTrue((self.output / 'keep').exists())

    def test_video_input_is_rejected(self):
        for source in self.sources:
            path = source / 'meta/info.json'
            info = json.loads(path.read_text())
            info['features']['camera'] = {'dtype': 'video'}
            path.write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError, 'External video'):
            self.validate()

    def test_missing_episode_is_not_silently_skipped(self):
        (self.sources[1] / 'data/chunk-000/episode_000000.parquet').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Episode parquet'):
            self.validate()
        self.assertFalse(self.output.exists())

    def test_duplicate_source_is_rejected(self):
        self.sources = [self.sources[0], self.sources[0]]
        with self.assertRaisesRegex(ValueError, 'duplicate datasets'):
            self.validate()

    def test_multi_chunk_input_is_rejected_instead_of_dropping_episodes(self):
        self.change_info(total_chunks=2)
        with self.assertRaisesRegex(ValueError, 'chunk-000'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
