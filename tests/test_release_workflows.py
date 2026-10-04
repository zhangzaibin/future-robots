"""Portable behavior checks; no model downloads or GPU dependencies."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_public_release import export_snapshot


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config_utils = load_module("config_utils", ROOT / "src/openpi/training/config_utils.py")
paths = load_module("training_paths", ROOT / "src/openpi/training/paths.py")


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.log = self.directory / "calls.jsonl"
        self.uv = self.directory / "mock uv"
        self.uv.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "with Path(os.environ['CALL_LOG']).open('a') as f:\n"
            " f.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), 'env': dict(os.environ)}) + '\\n')\n"
            "if os.environ.get('FAIL_NORM') and 'scripts/compute_norm_stats.py' in sys.argv: sys.exit(7)\n"
        )
        self.uv.chmod(0o755)
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(('FUTURE_', 'DATASET_', 'NORM_', 'EXP_', 'BATCH_', 'FSDP_', 'NUM_TRAIN_'))}
        self.env.update(UV_BIN=str(self.uv), CALL_LOG=str(self.log))

    def run_recipe(self, *args, script="scripts/macg_new/train_mavla.sh", **env):
        return subprocess.run(["bash", str(ROOT / script), *args], cwd=self.directory,
                              env={**self.env, **env}, capture_output=True, text=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_default_from_other_directory_computes_stats_before_training(self):
        result = self.run_recipe()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        self.assertIn('scripts/compute_norm_stats.py', calls[0]['args'])
        self.assertIn('pi05_two_arms_multi_task_mavla_shuffle06_mask04', calls[1]['args'])
        self.assertEqual(calls[1]['cwd'], str(ROOT))
        self.assertNotIn('--overwrite', calls[1]['args'])

    def test_norm_failure_stops_training(self):
        result = self.run_recipe(FAIL_NORM='1')
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(self.calls()), 1)

    def test_resume_keeps_existing_statistics(self):
        result = self.run_recipe('--resume', NUM_TRAIN_STEPS='60000')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 1)
        self.assertIn('--resume', self.calls()[0]['args'])
        self.assertIn('60000', self.calls()[0]['args'])

    def test_conflicting_modes_fail_before_any_process(self):
        for options in [('--resume', '--overwrite'), ('--norm-only', '--skip-norm')]:
            with self.subTest(options=options):
                self.assertEqual(self.run_recipe(*options).returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_dry_run_launches_nothing(self):
        result = self.run_recipe('--dry-run', UV_BIN='/missing/uv')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('scripts/train.py', result.stdout)
        self.assertEqual(self.calls(), [])

    def test_spaces_and_environment_overrides_survive(self):
        result = self.run_recipe('--seed', '17', EXP_NAME='trial one', DATASET_DIR='/tmp/data with spaces',
                                 NORM_ASSETS_DIR='/tmp/stats one', BATCH_SIZE='8', FSDP_DEVICES='1',
                                 PALIGEMMA_TOKENIZER_PATH='/tmp/tokenizer.model')
        self.assertEqual(result.returncode, 0, result.stderr)
        for call in self.calls():
            self.assertEqual(call['env']['FUTURE_DATASET_DIR'], '/tmp/data with spaces')
            self.assertEqual(call['env']['FUTURE_ASSETS_DIR'], '/tmp/stats one')
            self.assertEqual(call['env']['PALIGEMMA_TOKENIZER_PATH'], '/tmp/tokenizer.model')
        self.assertIn('trial one', self.calls()[1]['args'])
        self.assertEqual(self.calls()[1]['args'][-2:], ['--seed', '17'])

    def test_arm_recipes_select_distinct_configs(self):
        for arm in ['left', 'right']:
            result = self.run_recipe('--norm-only',
                script=f'scripts/macg_new/per-arm-policy/train_pi05_single_arm_{arm}_multi_task.sh')
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('pi05_single_arm_left_multi_task', self.calls()[0]['args'])
        self.assertIn('pi05_single_arm_right_multi_task', self.calls()[1]['args'])

    def test_explicit_overwrite_is_forwarded(self):
        result = self.run_recipe('--overwrite', '--skip-norm')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 1)
        self.assertIn('--overwrite', self.calls()[0]['args'])

    def test_serving_requires_checkpoint(self):
        result = self.run_recipe(script='scripts/macg_new/per-arm-policy/deploy_left_arm0.sh')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_serving_preserves_checkpoint_path_and_uses_localhost(self):
        result = self.run_recipe(script='scripts/macg_new/per-arm-policy/deploy_right_arm1.sh',
                                 CHECKPOINT_DIR='/tmp/step with spaces/29999')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.calls()[0]['args']
        self.assertIn('/tmp/step with spaces/29999', args)
        self.assertIn('127.0.0.1', args)
        self.assertIn('20022', args)


@dataclasses.dataclass(frozen=True)
class Transform:
    shuffle_prob: float = 0.6
    image_mask_prob: float = 0.4
    image_noise_prob: float = 0.2
    wait_augment_prob: float = 0.3
    single_arm_id: int = 1
    use_delta_joint_actions: bool = True


@dataclasses.dataclass(frozen=True)
class Assets:
    assets_dir: str | None = '/tmp/statistics'


@dataclasses.dataclass(frozen=True)
class Data:
    data_transforms: object = dataclasses.field(default_factory=Transform)
    assets: Assets = dataclasses.field(default_factory=Assets)
    repo_id: str = '/tmp/shared_dataset'
    asset_id: str = 'right_arm'


@dataclasses.dataclass(frozen=True)
class Config:
    data: Data = dataclasses.field(default_factory=Data)
    assets_dirs: Path = Path('/tmp/default_assets')


class ConfigurationTests(unittest.TestCase):
    def test_statistics_and_inference_remove_only_stochastic_augmentation(self):
        original = Config()
        clean = config_utils.without_augmentation(original)
        self.assertEqual(clean.data.data_transforms, Transform(0, 0, 0, 0))
        self.assertEqual(original.data.data_transforms.shuffle_prob, 0.6)
        self.assertEqual(clean.data.data_transforms.single_arm_id, 1)
        self.assertTrue(clean.data.data_transforms.use_delta_joint_actions)

    def test_callable_transform_remains_supported(self):
        original = Config(data=Data(data_transforms=lambda x: x))
        self.assertIs(config_utils.without_augmentation(original), original)

    def test_statistics_use_asset_id_instead_of_shared_repo_id(self):
        config = Config()
        self.assertEqual(config_utils.norm_stats_directory(config, config.data), Path('/tmp/statistics/right_arm'))

    def test_statistics_fall_back_to_config_assets(self):
        config = Config(data=Data(assets=Assets(None)))
        self.assertEqual(config_utils.norm_stats_directory(config, config.data), Path('/tmp/default_assets/right_arm'))

    def test_default_paths_are_portable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(paths.dataset_path('combined'), str(ROOT / 'dataset/combined'))
            self.assertEqual(paths.pretrained_params('pi05'), 'gs://openpi-assets/checkpoints/pi05_base/params')

    def test_explicit_paths_take_precedence(self):
        with patch.dict(os.environ, {'FUTURE_DATA_ROOT': '/tmp/data', 'FUTURE_CHECKPOINT_ROOT': '/tmp/runs',
                                     'FUTURE_PI05_PARAMS': '/tmp/weights/params'}):
            self.assertEqual(paths.dataset_path('combined'), '/tmp/data/combined')
            self.assertEqual(paths.checkpoint_path('trial'), '/tmp/runs/trial')
            self.assertEqual(paths.pretrained_params('pi05'), '/tmp/weights/params')


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.output = self.root / 'public'
        self.git('init', '-q')
        (self.source / 'README.md').write_text('original\n')
        (self.source / '.gitignore').write_text('dataset/\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Release Test', '-c', 'user.email=test@example.invalid',
                 '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture')

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.source), *args], stderr=subprocess.STDOUT)

    def test_export_includes_edits_and_excludes_history_local_settings_and_data(self):
        (self.source / 'README.md').write_text('updated\n')
        (self.source / 'new.py').write_text('value = 1\n')
        (self.source / '.claude').mkdir()
        (self.source / '.claude/local.json').write_text('{}')
        (self.source / 'dataset').mkdir()
        (self.source / 'dataset/data.bin').write_bytes(b'data')
        (self.source / '.env').write_text('LOCAL=1')
        self.git('add', '.env', '.claude')
        manifest = export_snapshot(self.source, self.output)
        self.assertEqual((self.output / 'README.md').read_text(), 'updated\n')
        self.assertTrue((self.output / 'new.py').exists())
        for excluded in ['.git', '.claude', '.env', 'dataset']:
            self.assertFalse((self.output / excluded).exists())
        self.assertFalse(manifest['source_git_history_included'])

    def test_secret_is_rejected_without_echoing_value(self):
        marker = '-----BEGIN ' + 'OPENSSH ' + 'PRIVATE KEY-----'
        (self.source / 'bad.sh').write_text(marker + '\nnot-a-real-key\n')
        with self.assertRaisesRegex(ValueError, 'Potential credential in: bad.sh') as raised:
            export_snapshot(self.source, self.output)
        self.assertNotIn(marker, str(raised.exception))
        self.assertFalse(self.output.exists())

    def test_nonempty_destination_is_preserved(self):
        self.output.mkdir()
        (self.output / 'keep').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'new or empty'):
            export_snapshot(self.source, self.output)
        self.assertEqual((self.output / 'keep').read_text(), 'keep')

    def test_source_nested_destination_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside the source'):
            export_snapshot(self.source, self.source / 'export')

    def test_symlinks_require_review(self):
        (self.source / 'outside').symlink_to('/etc/hosts')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            export_snapshot(self.source, self.output)

    def test_only_registered_submodules_are_exported(self):
        (self.source / '.gitmodules').write_text(
            '[submodule "third_party/example"]\n\tpath = third_party/example\n\turl = https://example.invalid/repo\n')
        commit = self.git('rev-parse', 'HEAD').decode().strip()
        for path in ['third_party/example', 'unregistered']:
            self.git('update-index', '--add', '--cacheinfo', '160000', commit, path)
        manifest = export_snapshot(self.source, self.output)
        self.assertEqual(manifest['submodules'], [{'path': 'third_party/example', 'commit': commit}])


if __name__ == '__main__':
    unittest.main()
