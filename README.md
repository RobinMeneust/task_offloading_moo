# moo_project

## Install

### Requirements

#### Without uv

`pip install -r requirements.txt` and then `pip install -e .`

#### With uv

- Install uv: https://docs.astral.sh/uv/getting-started/installation/
- `uv venv --python 3.12.0`
- Then activate it using the command displayed
- `uv pip install -r requirements.txt`
- `uv pip install -e .`

### Pre-commit

`pre-commit install`

## Usage

With conda use `conda activate moo_project` to activate the environment.

## Variables

### Machines

- cpu_rate: MIPS (1e6)
- cpu_usage_cost: €/s
- ram_usage_cost: €/GB/s
- bandwidth_usage_cost: €/GB
- distance_edge: m
- bandwidth: GB/s
- ram_limit: GB

### Tasks

- num_instructions: millions of instructions (1e6)
- ram_required: GB
- in_traffic: GB
- out_traffic: GB


### Assumptions and notations

- IPC (instructions per cycle) is assumed to be 1, but it varies actually
- CPS (cycles per second) is clock frequency
- cpu_rate = IPS (instructions per second) is calculated as $IPC \times CPS = 2 CPS$ 