from typing import List, Tuple
import re


def infer_type(name: str) -> str:
    for s in ["model", "array", "image", "audio", "video", "json"]:
        if s in name.lower():
            return s
    return None


def find_args(request: str) -> List[Tuple[str, Tuple[str | None, str | None]]]:
    matches = re.findall(r"<<(.+?)>>", request)
    args = {}

    for match in matches:
        name = match.strip().split("|")
        name = [arg.strip() for arg in name]
        name = [arg for arg in name if arg]

        if not len(name):
            continue

        if len(name) == 1:
            name, default = name[0], None
        else:
            name, default = name[0], name[1]

        type_str = infer_type(name)

        if name not in args or args[name][0] is None:
            args[name] = default, type_str

    return list(args.items())


def replace_arg(request: str, arg_name: str, value: str) -> str:
    return re.sub(f"<<{arg_name}(\|.*?)?>>", value, request)
