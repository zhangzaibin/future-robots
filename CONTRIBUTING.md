# Contributing to Future

Future is a compositional robot learning project from [Evolving Machines Lab](https://zhangzaibin.github.io/evolving-machines/), built on OpenPI. Start with the [documentation map](docs/README.md) and [installation instructions](docs/installation.md).

For a code change, explain the problem, the final behavior, and how it was validated. For a research change, also record the config, dataset/splits, seed, checkpoint, and any changes to observations, action representation, augmentation, or evaluation protocol. Preserve existing config names when checkpoints depend on them.

Run the portable checks before opening a pull request:

```bash
python3 scripts/check_release.py
python3 -m unittest discover -s tests -p 'test_*.py'
```

Changes to models, training, or robot transforms also need relevant tests in the Linux GPU environment and a representative training/inference smoke run. State clearly when GPU checks were not run. Public CI runs the portable checks; it does not verify numerical training or benchmark success.

Keep datasets, model weights, tokens, SSH keys, machine-specific settings, and generated outputs out of commits. Use environment variables for paths; retain third-party license and attribution notices. See [publishing](docs/publishing.md) when moving an experimental checkout into the public repository.
