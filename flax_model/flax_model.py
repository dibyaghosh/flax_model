import copy
import dataclasses
import importlib
import json
from functools import partial
from typing import Any, Callable, Dict, Tuple, TypedDict, Union

import flax
import flax.linen as nn
import flax.training.checkpoints as checkpoints
import jax
import numpy as np
import optax
import tensorflow as tf


class FlaxModel(flax.struct.PyTreeNode):
    model_def: nn.Module = flax.struct.field(pytree_node=False)
    params: dict

    @property
    def apply_fn(self):
        return self.model_def.apply

    def __call__(self, *args, variables=None, **kwargs):
        variables = variables or {"params": self.params}
        return self.model_def.apply(variables, *args, **kwargs)

    def save_metadata(self, save_dir: str):
        _save_model_def(self.model_def, save_dir)

    @classmethod
    def create(cls, model_def, params, **kwargs):
        return cls(model_def=model_def, params=params, **kwargs)

    @classmethod
    def load(cls, save_dir: str, step: int = None, prefix: Tuple[str] = None, **kwargs):
        model_def = _load_model_def(save_dir)
        params = checkpoints.restore_checkpoint(save_dir, step=step, target=None)
        if prefix is not None:
            for p in prefix:
                params = params[p]
        return cls.create(model_def=model_def, params=params, **kwargs)


class FlaxTrainState(FlaxModel):
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    opt_state: optax.OptState = flax.struct.field(pytree_node=True)
    step: int

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            params=new_params, opt_state=new_opt_state, step=self.step + 1, **kwargs
        )

    @classmethod
    def create(cls, *, model_def, params, tx, **kwargs):
        opt_state = tx.init(params)
        return cls(
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            step=0,
            **kwargs,
        )


def _default(field: dataclasses.Field):
    if field.default != dataclasses.MISSING:
        return field.default
    elif field.default_factory != dataclasses.MISSING:
        return field.default_factory()
    else:
        return dataclasses.MISSING


def patch():
    nn.Module.get_config = get_config
    nn.Module.from_config = from_config


def _save_model_def(model_def: nn.Module, save_dir: str):
    tf.io.gfile.makedirs(save_dir, exist_ok=True)
    with tf.io.gfile.GFile(f"{save_dir}/flax_model_config.json", "w") as f:
        json.dump(get_config(model_def), f)


def _load_model_def(load_dir: str):
    with tf.io.gfile.GFile(f"{load_dir}/flax_model_config.json", "r") as f:
        config = json.load(f)
    return from_config(config)


def get_config(module: nn.Module, log_defaults=False, serializable=True):
    config = {}
    config["_nn_module_cls_"] = module.__class__
    for field in dataclasses.fields(module):
        if field.name not in ("parent", "name"):
            value = getattr(module, field.name)
            if log_defaults or _default(field) != value:

                def recurse(x):
                    if isinstance(x, nn.Module):
                        return get_config(x, log_defaults)
                    return x

                config[field.name] = jax.tree_map(
                    recurse, value, is_leaf=lambda x: isinstance(x, nn.Module)
                )
    if serializable:
        config = jax.tree_map(_serialize_object, config)
    return config


def from_config(config):
    assert "_nn_module_cls_" in config
    config = _deserialize_object(config)

    def create_sub_config(x):
        return from_config(x) if isinstance(x, dict) else x

    config = {
        k: jax.tree_map(
            create_sub_config,
            v,
            is_leaf=lambda x: isinstance(x, dict) and "_nn_module_cls_" in x,
        )
        for k, v in config.items()
    }
    cls_kwargs, cls_fn = flax.core.pop(config, "_nn_module_cls_")
    return cls_fn(**cls_kwargs)


def _infer_full_name(o: object):
    if hasattr(o, "__module__") and hasattr(o, "__qualname__"):
        return f"{o.__module__}:{o.__qualname__}"
    else:
        raise ValueError(f"Could not infer identifier for {o}. ")


def _import_from_string(target: str):
    """
    Args:
        target: A fully qualified import string (e.g. "torch.optim:Adam")
    """
    module_string, name = target.split(":")
    try:
        module = importlib.import_module(module_string)
        subs = name.split(".")
        o = module
        for sub in subs:
            o = getattr(o, sub)
        return o
    except Exception as e:
        raise ValueError(f"Could not import {target}") from e


def _compress_partial(p: partial):
    """Compresses a nested partial into a single partial object."""
    args = p.args
    kwargs = p.keywords
    fn = p.func
    while isinstance(fn, partial):
        args = fn.args + args
        kwargs = {**fn.keywords, **kwargs}
        fn = fn.func
    return partial(fn, *args, **kwargs)


def _serialize_object(obj: Any):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, jax.Array) or isinstance(obj, np.ndarray):
            return {
                "_target_": "numpy:array",
                "_args_": [obj.tolist()],
                "_keywords_": {"dtype": str(obj.dtype)},
                "_called_": True,
            }  # Instantiates the numpy array upon deserialization

        if isinstance(obj, partial):
            obj = _compress_partial(obj)
            return {
                "_target_": _infer_full_name(obj.func),
                "_args_": _serialize_object(obj.args),
                "_keywords_": _serialize_object(obj.keywords),
            }
        else:
            return {"_target_": _infer_full_name(obj)}


def _deserialize_object(o: object):

    def _deserialize_recursive(obj):
        return jax.tree_map(
            _deserialize_object,
            obj,
            is_leaf=lambda x: isinstance(x, dict) and "_target_" in x and x != obj,
        )

    o = _deserialize_recursive(o)

    if not isinstance(o, dict) or "_target_" not in o:
        return o

    d = dict(o)
    target = _import_from_string(d.pop("_target_"))
    if "_args_" in d or "_keywords_" in d:
        args = d.pop("_args_", [])
        keywords = d.pop("_keywords_", {})
        if d.get("_called_", False):
            return target(*args, **keywords)
        return partial(target, *args, **keywords)
    else:
        return target
