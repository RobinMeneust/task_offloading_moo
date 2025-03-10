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

### Pre-commit (optional)

To enable checks before commits, use: `pre-commit install`

## Usage

With conda use `conda activate moo_project` to activate the environment. With uv, or any otheer virtual env, use the associated command or set your interpreter to the one inside the virtual env Scripts folder.

Then, simply use the notebooks to learn how to use the package, and refer to the documentation if needed.

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

## Documentation

### Requirements (NOT in requirements.txt)

- sphinx
- myst_parser
- rst2pdf
- sphinx-mdinclude
- sphinx_rtd_theme


`uv pip install sphinx myst_parser rst2pdf sphinx-mdinclude sphinx_rtd_theme`

### Generation

#### Setup

1. Generate the rst files from the code: `sphinx-apidoc -o docs/source/ src/task_offloading_moo`
2. Generate the documentation from the rst files 
   - In HTML: `docs/make.bat html`
   - In PDF: `docs/make.bat pdf`
3. Open the documentation:
   - HTML: Open `docs/_build/html/index.html`
   - PDF: Open `docs/_build/pdf/MoFGBMLPy.pdf`
