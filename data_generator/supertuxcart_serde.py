import pickle
from pathlib import Path
from typing import Dict, Union

def save_pickle(obj, path: Union[str, Path]):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: Union[str, Path]):
    with open(path, 'rb') as f:
        return pickle.load(f)

def dict_to_str(d: Dict):
    str = "{\n"
    for k, v in d.items():
        if isinstance(v, dict):
            v = dict_to_str(v)
        str += f"'{k}': {v},\n"
    return str + "}"

def save_dict(d: Dict, path: str, as_str: bool=False) -> None:
    if as_str:
        with open(path, 'w', encoding="utf-8") as file:
            file.write(dict_to_str(d))
    else:
        save_pickle(d, path)

def load_dict(path: str) -> Dict:
    try:
        return load_pickle(path)
    except pickle.UnpicklingError as e:
        pass

    with open(path, "r", encoding="utf-8") as file:
        from ast import literal_eval
        s = file.read()
        return dict(literal_eval(s))