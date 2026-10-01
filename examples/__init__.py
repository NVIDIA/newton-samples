# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ast
import copy
import gc
import importlib
import math
import os
import time
import warnings
from collections import defaultdict
from collections.abc import Callable

import warp as wp

import newton

_DEPRECATED_WARP_CONFIG_KEYS = {"quiet", "verbose"}


def get_source_directory() -> str:
    return os.path.realpath(os.path.dirname(__file__))


def get_asset_directory() -> str:
    return os.path.join(get_source_directory(), "assets")


def get_asset(filename: str) -> str:
    return os.path.join(get_asset_directory(), filename)


def _enable_example_deprecation_warnings() -> None:
    """Show Newton deprecations during example runs."""
    if "PYTHONWARNINGS" in os.environ or getattr(_enable_example_deprecation_warnings, "_installed", False):
        return
    warnings.filterwarnings("default", category=DeprecationWarning, module=r"newton(\.|$)")
    _enable_example_deprecation_warnings._installed = True


class _ExampleBrowser:
    """Manage the example browser UI and switching/reset logic for the run loop."""

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.switch_target: str | None = None
        self._reset_requested = False
        self.callback = None
        self._tree: dict[str, list[tuple[str, str]]] = {}
        self._initial_args = copy.deepcopy(args) if args is not None else None

        if not hasattr(viewer, "register_ui_callback"):
            return

        examples = get_examples()
        tree: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for name, module_path in examples.items():
            parts = module_path.split(".")
            category = parts[1] if len(parts) > 1 else "other"
            tree[category].append((name, module_path))
        self._tree = dict(sorted(tree.items()))

        def _browser_ui(imgui):
            imgui.set_next_item_open(False, imgui.Cond_.appearing)
            if imgui.collapsing_header("Examples"):
                for category in sorted(self._tree.keys()):
                    if imgui.tree_node(category):
                        for name, module_path in self._tree[category]:
                            clicked, _ = imgui.selectable(name, False)
                            if clicked:
                                self.switch_target = module_path
                        imgui.tree_pop()

        self.callback = _browser_ui
        viewer.register_ui_callback(_browser_ui, position="panel")
        if hasattr(viewer, "set_reset_callback"):
            viewer.set_reset_callback(lambda: setattr(self, "_reset_requested", True))

    def _register_ui(self, example):
        if hasattr(example, "gui") and hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(lambda ui, ex=example: ex.gui(ui), position="side")

    def _show_splash(self, text):
        if not hasattr(self.viewer, "show_loading_splash"):
            return
        self.viewer.show_loading_splash(text)
        for _ in range(2):
            self.viewer.begin_frame(0.0)
            self.viewer.end_frame()

    def _hide_splash(self):
        if hasattr(self.viewer, "hide_loading_splash"):
            self.viewer.hide_loading_splash()

    def switch(self, example_class):
        module_path, self.switch_target = self.switch_target, None
        self._show_splash(f"Loading {module_path.rsplit('.', 1)[-1]}...")
        self.viewer.clear_model()
        try:
            mod = importlib.import_module(module_path)
            parser = getattr(mod.Example, "create_parser", create_parser)()
            new_args = default_args(parser)
            example = mod.Example(self.viewer, new_args)
        except Exception as e:
            warnings.warn(f"Failed to load example {module_path}: {e}", stacklevel=2)
            self._hide_splash()
            return None, example_class
        self._initial_args = copy.deepcopy(new_args)
        self._register_ui(example)
        self._hide_splash()
        return example, type(example)

    def reset(self, example_class):
        self._reset_requested = False
        self._show_splash("Resetting...")
        self.viewer.clear_model()
        try:
            if self._initial_args is not None:
                args = copy.deepcopy(self._initial_args)
            else:
                parser = getattr(example_class, "create_parser", create_parser)()
                args = default_args(parser)
            new_example = example_class(self.viewer, args)
        except Exception as e:
            warnings.warn(f"Failed to reset example: {e}", stacklevel=2)
            self._hide_splash()
            return None
        self._register_ui(new_example)
        self._hide_splash()
        return new_example


def _format_fps(fps: float) -> str:
    if fps >= 10:
        return f"{fps:.1f}"
    if fps >= 1:
        return f"{fps:.2f}"
    return f"{fps:.3f}"


def _positive_float(value: str) -> float:
    import argparse  # noqa: PLC0415

    try:
        result = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid float") from e
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite value greater than 0")
    return result


def _throttle_render_fps(
    frame_start_time: float,
    render_fps: float | None,
    *,
    time_fn: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> float:
    if render_fps is None:
        return 0.0
    if not math.isfinite(render_fps) or render_fps <= 0.0:
        raise ValueError("render_fps must be a finite value greater than 0")
    sleep_time = 1.0 / render_fps - (time_fn() - frame_start_time)
    if sleep_time <= 0.0:
        return 0.0
    sleep_fn(sleep_time)
    return sleep_time


def run(example, args):
    viewer = example.viewer
    example_class = type(example)
    render_fps = getattr(args, "render_fps", None)

    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    perform_test = args is not None and args.test
    test_post_step = perform_test and hasattr(example, "test_post_step")
    test_final = perform_test and hasattr(example, "test_final")

    browser = _ExampleBrowser(viewer, args) if not perform_test else None

    if hasattr(example, "gui") and hasattr(viewer, "register_ui_callback"):
        viewer.register_ui_callback(lambda ui, ex=example: ex.gui(ui), position="side")

    while viewer.is_running():
        frame_start_time = time.perf_counter()

        if browser is not None and browser.switch_target is not None:
            example, example_class = browser.switch(example_class)
            continue

        if browser is not None and browser._reset_requested:
            example = None
            gc.collect()
            example = browser.reset(example_class)
            continue

        if example is None:
            viewer.begin_frame(0.0)
            viewer.end_frame()
            _throttle_render_fps(frame_start_time, render_fps)
            continue

        if viewer.should_step():
            with wp.ScopedTimer("step", active=False):
                example.step()
        if test_post_step:
            example.test_post_step()

        with wp.ScopedTimer("render", active=False):
            example.render()

        _throttle_render_fps(frame_start_time, render_fps)

    if perform_test:
        if test_final:
            example.test_final()
        elif not (test_post_step or test_final):
            raise NotImplementedError("Example does not have a test_final or test_post_step method")

    viewer.close()

    if hasattr(viewer, "benchmark_result"):
        result = viewer.benchmark_result()
        if result is not None:
            print(
                f"Benchmark: {_format_fps(result['fps'])} FPS ({result['frames']} frames in {result['elapsed']:.2f}s)"
            )


def get_examples() -> dict[str, str]:
    """Return a dict mapping example short names to their full module paths."""
    example_map = {}
    examples_dir = get_source_directory()
    for module in sorted(os.listdir(examples_dir)):
        module_dir = os.path.join(examples_dir, module)
        if not os.path.isdir(module_dir) or module.startswith("_"):
            continue
        for filename in sorted(os.listdir(module_dir)):
            if filename.startswith("example_") and filename.endswith(".py"):
                example_name = filename[8:-3]
                example_map[example_name] = f"examples.{module}.{filename[:-3]}"
    return example_map


def _print_examples(examples: dict[str, str]) -> None:
    print("Available examples:")
    for name in examples:
        print(f"  {name}")


def create_parser():
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--viewer",
        type=str,
        default="gl",
        choices=["gl", "usd", "rtx", "rerun", "null", "viser"],
        help="Viewer to use.",
    )
    parser.add_argument(
        "--rerun-address",
        type=str,
        default=None,
        help="Connect to an external Rerun server.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="output.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=100, help="Total number of frames.")
    parser.add_argument(
        "--render-fps",
        type=_positive_float,
        default=None,
        help="Maximum render rate in frames per second. Does not change simulation frame timing.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to initialize the viewer headless.",
    )
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to run the example in test mode.",
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Suppress Warp compilation messages.",
    )
    parser.add_argument(
        "--paused",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start the viewer in a paused state.",
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=False,
        nargs="?",
        const=None,
        metavar="SECONDS",
        help="Run in benchmark mode.",
    )
    parser.add_argument(
        "--warp-config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a warp.config attribute.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        default=False,
        help="Use the most aggressive process priority in benchmark mode.",
    )
    return parser


def add_mujoco_contacts_arg(parser):
    import argparse  # noqa: PLC0415

    parser.add_argument(
        "--use-mujoco-contacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use MuJoCo's native contact solver instead of Newton contacts.",
    )
    return parser


def add_world_count_arg(parser):
    parser.add_argument(
        "--world-count",
        type=int,
        default=1,
        help="Number of simulation worlds.",
    )
    return parser


def add_max_worlds_arg(parser):
    parser.add_argument(
        "--max-worlds",
        type=int,
        default=None,
        help="Maximum number of worlds to render.",
    )
    return parser


def default_args(parser=None):
    if parser is None:
        parser = create_parser()
    return parser.parse_known_args([])[0]


def _apply_warp_config(parser, args):
    if not args.warp_config:
        return
    for entry in args.warp_config:
        if "=" not in entry:
            parser.error(f"invalid --warp-config format '{entry}': expected KEY=VALUE")
        key, value_str = entry.split("=", 1)
        if key in _DEPRECATED_WARP_CONFIG_KEYS:
            parser.error(f"invalid --warp-config key '{key}': use 'log_level' instead")
        if not hasattr(wp.config, key):
            parser.error(f"invalid --warp-config key '{key}': not a recognized warp.config setting")
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str
        setattr(wp.config, key, value)


def _raise_benchmark_priority(realtime=False):
    import sys  # noqa: PLC0415

    def _fail(msg):
        if realtime:
            raise SystemExit(f"Error: {msg}")
        print(f"Warning: Benchmark running at default process priority. Results may vary. {msg}")

    if sys.platform == "linux":
        try:
            os.nice(-20 if realtime else -15)
        except PermissionError:
            _fail("Run with elevated privileges to automatically raise priority.")


def init(parser=None):
    import newton.viewer  # noqa: PLC0415

    _enable_example_deprecation_warnings()

    if parser is None:
        parser = create_parser()
        args = parser.parse_known_args()[0]
    else:
        args = parser.parse_args()

    _apply_warp_config(parser, args)

    if args.quiet:
        wp.config.log_level = max(wp.config.log_level, wp.LOG_WARNING)
    if args.device:
        wp.set_device(args.device)
    if args.benchmark is not False:
        args.viewer = "null"
        _raise_benchmark_priority(realtime=args.realtime)

    visible_gl = args.viewer == "gl" and not args.headless
    if args.viewer == "gl":
        viewer = newton.viewer.ViewerGL(headless=args.headless, paused=args.paused)
    elif args.viewer == "usd":
        if args.output_path is None:
            raise ValueError("--output-path is required when using usd viewer")
        viewer = newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
    elif args.viewer == "rtx":
        viewer = newton.viewer.ViewerRTX(headless=args.headless, paused=args.paused, num_frames=args.num_frames)
    elif args.viewer == "rerun":
        viewer = newton.viewer.ViewerRerun(address=args.rerun_address)
    elif args.viewer == "null":
        viewer = newton.viewer.ViewerNull(
            num_frames=args.num_frames,
            benchmark=args.benchmark is not False,
            benchmark_timeout=args.benchmark or None,
        )
    elif args.viewer == "viser":
        viewer = newton.viewer.ViewerViser()
    else:
        raise ValueError(f"Invalid viewer: {args.viewer}")

    if visible_gl:
        viewer.show_loading_splash("Loading...")
        for _ in range(3):
            viewer.begin_frame(0.0)
            viewer.end_frame()

    return viewer, args


def main():
    import runpy  # noqa: PLC0415
    import sys  # noqa: PLC0415

    _enable_example_deprecation_warnings()
    examples = get_examples()

    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python -m examples <example_name> [options]")
        print("       python -m examples --list")
        print()
        print("Run 'python -m examples <example_name> --help' to see example-specific options.")
        raise SystemExit(0)

    if len(sys.argv) >= 2 and sys.argv[1] == "--list":
        _print_examples(examples)
        raise SystemExit(0)

    if len(sys.argv) < 2:
        _print_examples(examples)
        raise SystemExit(0)

    example_name = sys.argv[1]
    if example_name not in examples:
        print(f"Error: Unknown example '{example_name}'\n")
        _print_examples(examples)
        raise SystemExit(1)

    target_module = examples[example_name]
    sys.argv = [target_module, *sys.argv[2:]]
    runpy.run_module(target_module, run_name="__main__")


__all__ = [
    "add_max_worlds_arg",
    "add_mujoco_contacts_arg",
    "add_world_count_arg",
    "create_parser",
    "default_args",
    "get_asset",
    "get_examples",
    "init",
    "run",
]
