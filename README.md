# flax_model

This is a small library that makes it easier to save and load Flax models. Specifically, it provides a way to save a model_def as a json-serializable dictionary

## TL;DR

```python
>>> from flax_model import get_config, from_config
>>> model_def = nn.Sequential([nn.Dense(10), nn.Relu(), nn.Dense(1)])
>>> config = get_config(model_def) # This is JSON-serializable
>>> model_def == from_config(config)

# Alternatively:
>>> import flax_model.patched
>>> model_def = nn.Sequential([nn.Dense(10), nn.Relu(), nn.Dense(1)])
>>> config = model_def.get_config() # Patched into flax.nn.Module

```

## Installation:

```
pip install git+https://github.com/dibyaghosh/flax_model.git
```
