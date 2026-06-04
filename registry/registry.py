import json
import os
from pathlib import Path
from typing import Dict, Optional


def build_from_cfg(cfg, registry, default_args=None):
    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
    else:
        raise ValueError
    return obj_cls(**args)


def build_from_name(cfg_name, registry, default_args=None):
    project_root = os.getenv('LARY_ROOT', str(Path(__file__).parent.parent.resolve()))
    cfg_path = os.path.join(project_root, 'configs/models/', f'{cfg_name}.json')
    cfg = json.load(open(cfg_path, 'r', encoding='utf-8'))
    obj_type = cfg.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
    else:
        raise ValueError
    return obj_cls(**cfg)


class Registry:
    def __init__(self, name: str, build_func: Optional[callable]=None):
        self._name = name
        self._module_dict: Dict[str, type] = dict()
        self.build_func = build_func

    def __len__(self):
        return len(self._module_dict)

    def get(self, key: str):
        return self._module_dict[key]

    def _register_module(self, module):
        name = module.__name__
        self._module_dict[name] = module

    def register_module(self):
        def _register(module):
            self._register_module(module=module)
            return module
        return _register

    def build(self, cfg, *args, **kwargs):
        if self.build_func is not None:
            return self.build_func(cfg, *args, **kwargs, registry=self)


MODEL = Registry('model', build_func=build_from_name)
DATASET = Registry('dataset', build_func=build_from_name)
