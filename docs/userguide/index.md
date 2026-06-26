<!-- markdownlint-disable MD014 -->

(userguide)=

# User Guide

Welcome to the ALCHEMI Toolkit user guide: this side of the documentation
is to provide a high-level and conceptual understanding of the philosophy
and supported features in `nvalchemi`.

## Quick Start

The quickest way to install ALCHEMI Toolkit:

```bash
$ pip install nvalchemi-toolkit-ops
```

Make sure it is importable:

```bash
$ python -c "import nvalchemi; print(nvalchemi.__version__)"
```

## About

- [Install](about/install)
- [Introduction](about/intro)

## Core Components

- [AtomicData and Batch](data)
- [Data Loading Pipeline](datapipes)
- {doc}`Models: Wrapping ML Interatomic Potentials <models>`
- {doc}`Hooks: Observe & Modify <hooks>`
- [Dynamics: Optimization and MD](dynamics)

## Distributed Simulations

- {doc}`Design Overview (presentation source) <distributed_design>`
- {doc}`Overview: Domain Decomposition <distributed>`
- {doc}`ShardTensor: Per-Atom Fields Across Ranks <distributed_shardtensor>`
- {doc}`Bring Your Own Model: Authoring a Spec <distributed_byo>`

## Advanced Usage

- [Zarr Compression Tuning](zarr_compression)
- [Agent Skills](agent_skills)

```{toctree}
:caption: About
:maxdepth: 1
:hidden:

about/install
about/intro
about/faq
about/contributing

```

```{toctree}
:caption: Core Components
:maxdepth: 1
:hidden:

data
datapipes
models
hooks
dynamics
```

```{toctree}
:caption: Distributed Simulations
:maxdepth: 1
:hidden:

distributed_design
distributed
distributed_shardtensor
distributed_byo
```

```{toctree}
:caption: Advanced Usage
:maxdepth: 1
:hidden:

zarr_compression
agent_skills
```
