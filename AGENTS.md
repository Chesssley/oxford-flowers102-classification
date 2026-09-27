# Oxford Flowers102 Classification

## Project overview

This repository is a university deep-learning experiment for image classification
using the Oxford 102 Flowers dataset.

Primary goal:

- classify all 102 Oxford Flowers102 categories
- use PyTorch and torchvision
- maintain a reproducible and understandable experimental workflow
- produce training results suitable for analysis in the experiment report

Prefer a straightforward implementation appropriate for a university deep-learning
project rather than unnecessary production infrastructure.

## Project root

The repository root is:

D:\Dev\Projects\deep-learning-labs\oxford-flowers102-classification

Treat this directory as the project boundary.

Do not modify sibling projects under:

D:\Dev\Projects\deep-learning-labs\

unless explicitly authorized.

## Environment

Use a project-local Python virtual environment:

.venv

Do not reuse another project's environment.

Do not install project dependencies globally.

The virtual environment is disposable and must be reproducible from project
dependency metadata.

Do not commit .venv.

Before installing PyTorch, CUDA-related packages, or other version-sensitive
dependencies, check the current environment and use officially supported compatible
versions.

Do not modify the NVIDIA driver, system CUDA installation, global Python, system
PATH, or registry merely to make this project work.

## Technology

Primary stack:

- Python
- PyTorch
- torchvision

Use additional libraries only when they provide a clear project need.

Likely supporting libraries may include packages for:

- numerical computation
- image processing
- metrics
- plotting
- testing

Do not add dependencies unnecessarily.

## Dataset

Use the official Oxford Flowers102 dataset through torchvision when practical.

The primary experiment targets all 102 classes.

Preserve the official train / validation / test distinction.

Do not silently mix validation or test data into training.

Downloaded dataset files are project data, not source code.

Large reproducible dataset files should not be committed to Git.

## Preferred repository structure

Use the following structure when each directory is actually needed:

.
├── AGENTS.md
├── .gitignore
├── README.md
├── pyproject.toml and/or the chosen dependency metadata
├── .venv/
├── src/
│   └── flowers102/
├── data/
│   ├── raw/
│   └── processed/
├── configs/
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   └── predictions/
├── tests/
├── notebooks/
├── scripts/
└── reports/

Do not create empty directories merely to match this example.

Add notebooks, scripts, reports, processed data, or other directories only when the
project actually needs them.

## Source layout

Reusable project logic should live under:

src\flowers102\

Keep responsibilities reasonably separated, such as:

- dataset and transforms
- model definitions
- training
- evaluation
- metrics
- shared utilities

Avoid a single large script when the code has clearly separable responsibilities.

At the same time, do not over-engineer the project with unnecessary abstractions.

## Configuration

Training parameters that users are expected to change repeatedly should be kept in a
clear configuration mechanism rather than scattered through source files.

Examples include:

- batch size
- learning rate
- epochs
- image size
- model selection
- random seed
- output paths

Use the simplest configuration approach appropriate to the project's size.

## Outputs

Generated experiment artifacts belong under:

outputs\

Examples:

- model checkpoints
- training logs
- loss and accuracy curves
- confusion matrices
- prediction results

Do not scatter generated files throughout src or the repository root.

Large generated checkpoints should normally remain untracked by Git.

## Reproducibility

Where practical:

- use explicit random seeds
- record important training configuration
- preserve train/validation/test separation
- record dependency versions
- keep paths relative to the project root
- avoid machine-specific absolute paths in source code

The project should be rebuildable from a fresh checkout plus the documented dataset
and dependency setup.

## Git

The repository should track durable project assets such as:

- source code
- configuration
- tests
- documentation
- dependency metadata
- small project-owned assets

The repository should normally ignore:

- .venv/
- Python caches
- IDE temporary files
- downloaded raw dataset files
- logs
- disposable generated outputs
- large model checkpoints

Use .gitignore rather than relying on manually avoiding these files.

## Change discipline

Before making changes:

1. inspect the repository
2. understand the existing structure
3. inspect current dependency metadata
4. inspect relevant code before creating new files

Make the smallest coherent change required by the task.

Do not restructure the entire project for a small feature or bug fix.

Do not change dependency versions or the environment unless the task requires it.

## Validation

Use the project-local .venv for all Python validation.

For environment-related work, verify at minimum:

- the Python executable belongs to this project's .venv
- pip belongs to this project's .venv
- torch imports successfully
- torchvision imports successfully

When GPU execution is expected, verify CUDA availability through PyTorch rather than
assuming it works.

For code changes, run relevant:

- tests
- import checks
- lightweight smoke tests
- training/evaluation smoke runs when appropriate

Avoid launching an expensive full training run solely for validation unless the task
requires it.

## Completion report

At the end of a Codex task, report concisely:

- what changed
- which files changed
- dependencies added or changed
- validation commands actually run
- validation results
- anything that remains unverified or unresolved