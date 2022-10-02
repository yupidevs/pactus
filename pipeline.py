from __future__ import annotations

import json
from pathlib import Path
from time import time
from typing import Any, Callable, List

SEPARATOR = "-"
SEP_LENGTH = 80


def _type_name(_type):
    val = str(_type)
    if val.startswith("<class '"):
        return val[8:-2]
    return val


class PipelineStep:
    """
    A pipeline step is a function that takes in some input and
    returns some output.
    """

    def __init__(self, func: Callable, name: str | None = None, cache: bool = False):
        self.func = func
        self.name = name or func.__name__
        self.cache = cache
        self.input_types = "None"
        self.output_type = "None"

        _in_types_dict = func.__annotations__.copy()
        if "return" in _in_types_dict:
            self.output_type = _type_name(_in_types_dict.pop("return"))
        _in_types = list(_in_types_dict.values())
        self.__in_count = len(_in_types)
        if _in_types:
            if len(_in_types) > 1:
                _types = ", ".join(map(_type_name, _in_types))
                self.input_types = f"tuple[{_types}]"
            else:
                self.input_types = _type_name(_in_types[0])

    def __call__(self, data=None):
        if self.input_types.startswith("tuple") and not self.__in_count == 1:
            assert data is not None
            return self.func(*data)
        return self.func(data) if data is not None else self.func()

    @staticmethod
    def build(name: str | None = None, cache: bool = False):
        """
        Builds a pipeline step from a function.

        Parameters
        ----------
        name : str, optional
            Pipeline step name, by default None.

            If None, the name will be the function name.
        """

        def decorator(func: Callable) -> PipelineStep:
            return PipelineStep(func, name, cache)

        return decorator


class Pipeline:
    """
    A pipeline is a sequence of pipeline steps.
    """

    def __init__(self, name: str, *args: PipelineStep | Pipeline):
        self._args = args
        self.name = name
        pipeline: List[PipelineStep] = []
        for arg in args:
            if isinstance(arg, Pipeline):
                pipeline.extend(arg.pipeline)
            else:
                pipeline.append(arg)

        self.pipeline = pipeline

        _name = name.strip().lower().replace(" ", "-")
        self._filename = f"{_name}_pipeline_status.json"
        self._status = self._load_status()
        self.__cache = [None for _ in self.pipeline]
        self.__current_step = 0

        # Check that the pipeline is valid
        for i, funcs in enumerate(zip(self.pipeline, self.pipeline[1:])):
            func_a, func_b = funcs
            _out = func_a.output_type
            _in = func_b.input_types
            if _in != _out:
                raise TypeError(
                    f"Pipeline step {i} ({func_a.name}) has output type '{_out}' "
                    f"but step {i + 1} ({func_b.name}) expects input type '{_in}'"
                )

    def _load_status(self) -> dict[str, Any]:
        if not Path(self._filename).exists():
            return {
                "cache": [None for _ in self.pipeline],
            }

        with open(self._filename, "r", encoding="utf-8") as status_fd:
            return json.load(status_fd)

    def _update_status(self):
        with open(self._filename, "w", encoding="utf-8") as status_fd:
            json.dump(self._status, status_fd, indent=4, ensure_ascii=False)

    def repr(
        self, depth: int = 0, from_step: str | None = None, verbose: bool = False
    ) -> str:
        """
        Returns a string representation of the pipeline.
        """

        def add_level(text: str):
            return "| " + text.replace("\n", "\n" + "| ")

        ans = []
        sep_length = SEP_LENGTH - 2 * depth
        if depth == 0:
            total = len(self.pipeline)
            ans.append(f"{self.name} pipeline    Total steps: {total}")
            ans.append("=" * sep_length)

        for i, step in enumerate(self._args):
            if i > 0 and verbose:
                ans.append(SEPARATOR * sep_length)
            supra_step = f"{from_step}." if from_step is not None else ""
            if isinstance(step, Pipeline):
                ans.append(f"Step {supra_step}{i} ({step.name} pipeline)")
                ans.append(
                    add_level(
                        step.repr(depth + 1, supra_step + str(i), verbose),
                    )
                )
            else:
                ans.append(f"Step {supra_step}{i}: {step.name}")
                if verbose and step.func.__doc__ is not None:
                    ans.append(step.func.__doc__[:-1])
        return "\n".join(ans)

    def show(self, verbose: bool = False) -> None:
        """
        Prints the pipeline steps.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints the pipeline steps along with the steps description
        """
        print(self.repr(verbose=verbose))

    def reset(self, hard: bool = False):
        """
        Resets the pipeline running state.
        """
        self.__current_step = 0
        if hard:
            self.__cache = [None for _ in self.pipeline]

    def run_next_step(self, *init_data, use_cache: bool = True, force: bool = False):
        """
        Runs the next step of the pipeline.
        """
        step = self.__current_step
        if step == len(self.pipeline):
            raise StopIteration
        res = self.run_step(step, *init_data, use_cache=use_cache, force=force)
        self.__current_step += 1
        return res

    def run_step(
        self, step: int, *init_data, use_cache: bool = True, force: bool = False
    ):
        """
        Runs a given step of the pipeline.
        """
        total = len(self.pipeline)
        assert 0 <= step < total, "Step index out of range"
        func = self.pipeline[step]
        data = init_data if step == 0 else self.__cache[step - 1]

        print(f"Step [{step}/{total}]:", func.name)

        if not force and self.__cache[step] is not None:
            return self.__cache[step]

        if use_cache and func.cache:
            cache_val = self._status["cache"][step]
            if cache_val is not None:
                result = cache_val
            else:
                result = func(data)
                self._status["cache"][step] = data
                self._update_status()
        else:
            result = func() if data is None else func(data)
        self.__cache[step] = result
        return result

    def run_steps(
        self,
        *init_data,
        start: int = 0,
        end: int = -1,
        use_cache: bool = True,
        force: bool = False,
    ):
        """
        Runs a group of consecutive steps
        """
        end = len(self.pipeline) if end == -1 else end
        res = init_data
        for i in range(start, end):
            res = self.run_step(i, *init_data, use_cache=use_cache, force=force)
        return res

    def run(self, *data, use_cache: bool = True):
        """
        Runs the pipeline on the given data.
        """
        self.reset()
        res = data
        for _ in self.pipeline:
            res = self.run_next_step(*data, use_cache=use_cache)
        return res
