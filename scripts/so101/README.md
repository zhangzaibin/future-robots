# SO101 / Bi-SO100 experiments

The maintained end-to-end instructions are in [Real-world training](../../docs/real_world.md):

- [Convert LeRobot v3 recordings](../../docs/real_world.md#convert-lerobot-v3-recordings)
- [Phase-label and train multiple tasks](../../docs/real_world.md#so101-multi-task)
- [Train towel folding](../../docs/real_world.md#towel-folding)
- [Append DAgger experiences and resume](../../docs/real_world.md#dagger-and-resuming)
- [Deploy the resulting policy](../../docs/deployment.md)

## State and arm convention

The physical state is `[left_j1..j5, left_gripper, right_j1..j5, right_gripper]`: 12 dimensions. Grippers occupy indices 5 and 11. Calibrate units and open/closed values for your own recordings; thresholds in the older phase detectors came from specific recorded trajectories.

## Phase labeling tools

| Script | Use |
| --- | --- |
| [`multitask/set_phases_so101_v2.py`](multitask/set_phases_so101_v2.py) | Current multi-task phase detection; use `--dry-run --visualize` first |
| [`multitask/run_phase_and_merge.sh`](multitask/run_phase_and_merge.sh) | Label three tasks, merge, and compute training-compatible statistics |
| [`stack_two_bowls/set_phases_stackbowls.py`](stack_two_bowls/set_phases_stackbowls.py) | Earlier two-phase bowl detector based on gripper changes and motion |
| [`stack_two_bowls/repair_dataset_phase.py`](stack_two_bowls/repair_dataset_phase.py) | Earlier bowl-dataset repair utility; inspect its configured paths before use |
| [`stack_two_bowls/visualize_phases_stackbowls.py`](stack_two_bowls/visualize_phases_stackbowls.py) | Inspect the earlier bowl annotations |
| [`place_two_cubes/set_phases_placecubes.py`](place_two_cubes/set_phases_placecubes.py) | Earlier sequential two-arm cube placement detector |

The older two-phase detectors assume the right arm moves before the left. The multi-task detector uses a richer pick/action/recover schema. Confirm labels visually and do not mix these schemas without remapping task instructions. The merged data's task metadata and arm order must match the chosen training config.
