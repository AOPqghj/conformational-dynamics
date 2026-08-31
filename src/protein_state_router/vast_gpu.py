"""Run one bounded Vast.ai job with the official ``vastai`` CLI.

The CLI owns Vast's API details.  SSH is used only after Vast reports a direct
SSH URL: for readiness checks, rsync transfers, and the single remote command.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_IMAGE = "vastai/pytorch:@vastai-automatic-tag"
PROJECT_EXCLUDES = (
    ".git",
    ".venv",
    "venv",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    ".DS_Store",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    "dist",
    "build",
    "data",
    "ml/results",
    "gpu-results",
)
FAILED_STATES = {"exited", "offline", "unknown", "destroyed", "deleted"}


class VastGpuError(RuntimeError):
    """An actionable Vast GPU workflow failure."""


class NoSuitableOfferError(VastGpuError):
    """No offer meets the requested constraints."""


class BudgetExceededError(VastGpuError):
    """The selected offer exceeds the maximum exposure."""


class CleanupError(VastGpuError):
    """An instance could not be verified as destroyed."""


@dataclass(frozen=True)
class Limits:
    max_rate: float
    max_cost: float
    max_hours: float
    min_vram_gb: int = 24
    min_reliability: float = 0.98
    disk_gb: int = 100
    min_inet_down_mbps: int = 100
    min_inet_up_mbps: int = 50
    gpu_name: str | None = None


@dataclass(frozen=True)
class Offer:
    offer_id: int
    gpu_name: str
    vram_gb: float
    rate: float
    reliability: float
    dlperf: float


class LifecycleReporter:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {message}"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(line, flush=True)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _reliability(value: Any) -> float:
    value = _number(value)
    return value / 100 if value > 1 else value


def state_path() -> Path:
    value = os.environ.get("GPU_RUN_STATE_PATH")
    return (
        Path(value).expanduser()
        if value
        else Path.home() / ".local/state/gpu-run/active-instance.json"
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def read_state() -> dict[str, Any] | None:
    path = state_path()
    return json.loads(path.read_text()) if path.is_file() else None


def update_state(phase: str) -> None:
    state = read_state()
    if state:
        state["phase"] = phase
        state["updated_at"] = time.time()
        write_json_atomic(state_path(), state)


def clear_state() -> None:
    state_path().unlink(missing_ok=True)


def load_local_api_key(path: Path = Path(".env")) -> None:
    """Load only VAST_API_KEY from the ignored project env file when needed."""
    if os.environ.get("VAST_API_KEY") or not path.is_file():
        return
    for line in path.read_text().splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == "VAST_API_KEY":
            os.environ["VAST_API_KEY"] = value.strip().strip("'\"")
            return


def vastai(*arguments: str, check: bool = True) -> Any:
    """Run the official CLI and decode its documented machine-readable output."""
    load_local_api_key()
    if not os.environ.get("VAST_API_KEY"):
        raise VastGpuError("Set VAST_API_KEY before contacting Vast.ai.")
    command = ["vastai", "--raw", *arguments]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        raise VastGpuError("Install the official Vast CLI: https://vast.ai/install.sh") from error
    if check and result.returncode:
        raise VastGpuError(
            f"vastai {' '.join(arguments[:3])} failed: {result.stderr.strip()[:500]}"
        )
    if not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise VastGpuError(
            "vastai did not return JSON; update the official CLI and retry."
        ) from error


def offer_from_api(raw: dict[str, Any]) -> Offer:
    offer_id = raw.get("id", raw.get("ask_contract_id"))
    if offer_id is None:
        raise VastGpuError("vastai search returned an offer without an ID.")
    return Offer(
        offer_id=int(offer_id),
        gpu_name=str(raw.get("gpu_name", "unknown")),
        vram_gb=_number(raw.get("gpu_ram")) / 1000,
        rate=_number(raw.get("dph_total", raw.get("dph_base"))),
        reliability=_reliability(raw.get("reliability")),
        dlperf=_number(raw.get("dlperf")),
    )


def compatible(offer: Offer, limits: Limits) -> bool:
    return (
        offer.vram_gb >= limits.min_vram_gb
        and offer.rate <= limits.max_rate
        and offer.reliability >= limits.min_reliability
        and (not limits.gpu_name or limits.gpu_name.lower() in offer.gpu_name.lower())
    )


def score(offer: Offer) -> tuple[float, float, float, float]:
    return (
        offer.dlperf / offer.rate if offer.rate else 0,
        offer.reliability,
        offer.dlperf,
        -offer.rate,
    )


def search_offers(limits: Limits) -> list[Offer]:
    filters = [
        "num_gpus = 1",
        "verified = true",
        "rentable = true",
        "direct_port_count >= 1",
        f"gpu_ram >= {limits.min_vram_gb}",
        f"reliability >= {limits.min_reliability}",
        f"inet_down >= {limits.min_inet_down_mbps}",
        f"inet_up >= {limits.min_inet_up_mbps}",
        f"dph_total <= {limits.max_rate}",
    ]
    if limits.gpu_name:
        filters.append(f"gpu_name = {limits.gpu_name.replace(' ', '_')}")
    payload = vastai(
        "search",
        "offers",
        " ".join(filters),
        "--type",
        "on-demand",
        "--limit",
        "100",
        "--order",
        "dlperf_usd-",
    )
    rows = payload.get("offers", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise VastGpuError("vastai search offers returned an unexpected response.")
    return sorted(
        (offer for row in rows if compatible(offer := offer_from_api(row), limits)),
        key=score,
        reverse=True,
    )


def select_offer(offers: Sequence[Offer], limits: Limits) -> Offer:
    if not offers:
        raise NoSuitableOfferError("No on-demand offers satisfy the requested limits.")
    selected = offers[0]
    if selected.rate * limits.max_hours > limits.max_cost:
        raise BudgetExceededError(
            f"Selected offer costs up to ${selected.rate * limits.max_hours:.2f}, above --max-cost ${limits.max_cost:.2f}."
        )
    return selected


def print_offers(
    offers: Sequence[Offer], limits: Limits, dry_run: bool, report: Callable[[str], None]
) -> Offer:
    selected = select_offer(offers, limits)
    if dry_run:
        report("DRY RUN - no instance will be rented or transferred.")
    for number, offer in enumerate(offers[:5], 1):
        report(
            f"#{number} {offer.gpu_name}: ${offer.rate:.3f}/hr, {offer.vram_gb:.1f} GB VRAM, reliability {offer.reliability:.1%}, DLPerf {offer.dlperf:.1f}"
        )
    report(
        f"Selected {selected.gpu_name}: ${selected.rate:.3f}/hr for at most {limits.max_hours:g} h (${selected.rate * limits.max_hours:.2f} maximum compute exposure)."
    )
    return selected


def create_instance(offer: Offer, limits: Limits, image: str, label: str) -> int:
    payload = vastai(
        "create",
        "instance",
        str(offer.offer_id),
        "--image",
        image,
        "--disk",
        str(limits.disk_gb),
        "--ssh",
        "--direct",
        "--cancel-unavail",
        "--label",
        label,
    )
    if (
        not isinstance(payload, dict)
        or not payload.get("success")
        or payload.get("new_contract") is None
    ):
        raise VastGpuError(f"vastai did not create an instance: {payload}")
    return int(payload["new_contract"])


def instance_details(instance_id: int) -> dict[str, Any] | None:
    payload = vastai("show", "instance", str(instance_id), check=False)
    if payload is None:
        return None
    if isinstance(payload, dict):
        return (
            payload.get("instances", payload)
            if isinstance(payload.get("instances", payload), dict)
            else payload
        )
    return None


def ssh_url(instance_id: int) -> str | None:
    value = vastai("ssh-url", str(instance_id), check=False)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return next(
            (
                str(item)
                for item in value.values()
                if isinstance(item, str) and item.startswith("ssh://")
            ),
            None,
        )
    return None


def _ssh_base(url: str, ssh_key: Path) -> list[str]:
    if not url.startswith("ssh://") or "@" not in url:
        raise VastGpuError(f"vastai ssh-url returned an invalid SSH URL: {url!r}")
    target = url.removeprefix("ssh://")
    host_part, _, port = target.rpartition(":")
    return [
        "ssh",
        "-i",
        str(ssh_key),
        "-p",
        port or "22",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=20",
        host_part,
    ]


def wait_for_ssh(instance_id: int, ssh_key: Path, timeout_seconds: int) -> str:
    deadline, delay, last_state = time.monotonic() + timeout_seconds, 5, "provisioning"
    while time.monotonic() < deadline:
        details = instance_details(instance_id)
        if details is None:
            raise VastGpuError(f"Instance {instance_id} disappeared while waiting for SSH.")
        last_state = str(details.get("actual_status", details.get("status", "unknown"))).lower()
        if last_state in FAILED_STATES:
            raise VastGpuError(f"Instance {instance_id} entered terminal state {last_state!r}.")
        url = ssh_url(instance_id)
        if (
            last_state == "running"
            and url
            and subprocess.run(
                _ssh_base(url, ssh_key) + ["true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        ):
            return url
        time.sleep(delay)
        delay = min(delay * 2, 30)
    raise VastGpuError(
        f"Timed out after {timeout_seconds}s waiting for SSH; last state was {last_state!r}."
    )


def preflight_ssh_key(key: Path) -> None:
    if not key.is_file() or not key.with_suffix(f"{key.suffix}.pub").is_file():
        raise VastGpuError(f"Configure an existing SSH key and public key: {key} and {key}.pub")


def preflight_cuda(url: str, key: Path, log_path: Path) -> None:
    """Prove that PyTorch can execute one CUDA operation before any upload."""
    check = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "torch.zeros(1, device='cuda').sum().item(); "
        "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        result = subprocess.run(
            _ssh_base(url, key) + [f"python -c {shlex.quote(check)}"],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise VastGpuError(f"Remote CUDA preflight failed. See {log_path}.")


def _rsync_shell(url: str, key: Path) -> str:
    base = _ssh_base(url, key)
    return shlex.join(base[:-1])


def rsync_to_remote(
    source: Path, destination: str, url: str, key: Path, excludes: Sequence[str] = ()
) -> None:
    command = ["rsync", "-aP", "-e", _rsync_shell(url, key)]
    command.extend(value for pattern in excludes for value in ("--exclude", pattern))
    command.extend([f"{source.resolve()}/", f"{_ssh_base(url, key)[-1]}:{destination}/"])
    subprocess.run(command, check=True)


def tar_to_remote(source: Path, destination: str, url: str, key: Path) -> None:
    """Keep the existing no-local-archive mode for the 7k dataset config."""
    producer = subprocess.Popen(
        ["tar", "-C", str(source.resolve()), "-cf", "-", "."], stdout=subprocess.PIPE
    )
    assert producer.stdout is not None
    consumer = subprocess.Popen(
        _ssh_base(url, key) + [f"tar -C {shlex.quote(destination)} -xf -"], stdin=producer.stdout
    )
    producer.stdout.close()
    if producer.wait() or consumer.wait():
        raise VastGpuError("Streamed tar dataset transfer failed.")


def cloud_copy(source: str, instance_id: int, destination: str, connection_id: int | None) -> None:
    """Use the supported CLI cloud-to-instance copy, not a private REST endpoint."""
    # ``vastai copy`` is synchronous and supports the documented
    # ``s3.<connection-id>:/path`` form.  Keep accepting the old s3:// YAML
    # spelling so existing private configs have a one-field migration path.
    if source.startswith("s3://"):
        if connection_id is None:
            raise VastGpuError("An s3:// cloud copy requires run.cloud_copy.connection_id.")
        source = f"s3.{connection_id}:/{source.removeprefix('s3://')}"
    vastai("copy", source, f"C.{instance_id}:{destination}")


def run_remote(
    command: Sequence[str], url: str, key: Path, max_hours: float, setup: str | None, log_path: Path
) -> None:
    remote = "cd /workspace/project && "
    if setup:
        remote += f"({setup}) && "
    remote += f"timeout --signal=TERM --kill-after=60s {max_hours:g}h {shlex.join(command)}"
    process = subprocess.Popen(
        _ssh_base(url, key) + [f"bash -lc {shlex.quote(remote)}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max_hours * 3600 + 90
    with log_path.open("w") as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            if time.monotonic() >= deadline:
                process.terminate()
                raise VastGpuError(
                    "Local runtime watchdog expired; retrieving outputs before cleanup."
                )
    if process.wait():
        raise VastGpuError(f"Remote command failed. See {log_path}.")


def remote_exists(path: str, url: str, key: Path) -> bool:
    return (
        subprocess.run(
            _ssh_base(url, key) + [f"test -e {shlex.quote(path)}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def safe_output_path(path: str) -> str:
    value = Path(path)
    if value.is_absolute() or ".." in value.parts:
        raise VastGpuError(
            f"--download must be a relative path inside the remote project: {path!r}"
        )
    return path


def rsync_from_remote(path: str, destination: Path, url: str, key: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-aP",
            "-e",
            _rsync_shell(url, key),
            f"{_ssh_base(url, key)[-1]}:/workspace/project/{path}",
            str(destination),
        ],
        check=True,
    )


def destroy_and_verify(instance_id: int) -> None:
    for attempt in range(3):
        vastai("destroy", "instance", str(instance_id), "-y", check=False)
        time.sleep(2**attempt)
        if instance_details(instance_id) is None:
            clear_state()
            return
    raise CleanupError(
        f"Could not confirm Vast instance {instance_id} was destroyed. Destroy it immediately."
    )


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise VastGpuError(f"Config file does not exist: {path}")
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise VastGpuError("Config must contain a YAML mapping.")
    return value


def add_limits_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-rate", type=float)
    parser.add_argument("--max-cost", type=float)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--min-vram", type=int)
    parser.add_argument("--min-reliability", type=float)
    parser.add_argument("--min-inet-down", type=int)
    parser.add_argument("--min-inet-up", type=int)
    parser.add_argument("--disk-gb", type=int)
    parser.add_argument("--gpu")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="action", required=True)
    offers = commands.add_parser("offers", help="search compatible on-demand offers")
    add_limits_arguments(offers)
    run = commands.add_parser("run", help="rent, run one job, retrieve outputs, then destroy")
    add_limits_arguments(run)
    run.add_argument("--dataset", type=Path)
    run.add_argument("--project", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--download", action="append")
    run.add_argument("--image")
    run.add_argument("--setup")
    run.add_argument("--ssh-key", type=Path)
    run.add_argument("--ssh-timeout", type=int)
    run.add_argument("--label")
    run.add_argument("--dataset-transfer", choices=("rsync", "tar", "cloud_copy", "none"))
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--keep-on-failure", action="store_true")
    run.add_argument("command", nargs=argparse.REMAINDER)
    launch = commands.add_parser("launch", help="detach one YAML-configured run locally")
    launch.add_argument("--config", required=True, type=Path)
    commands.add_parser("status", help="show the locally tracked active instance")
    cleanup = commands.add_parser("cleanup", help="destroy the locally tracked or named instance")
    cleanup.add_argument("instance_id", type=int, nargs="?")
    return root


def limits_from_args(args: argparse.Namespace) -> Limits:
    config, limits, gpu, storage = load_config(args.config), {}, {}, {}
    limits = config.get("limits", {})
    gpu = config.get("gpu", {})
    storage = config.get("storage", {})

    def value(cli: Any, mapping: dict[str, Any], key: str, default: Any) -> Any:
        return cli if cli is not None else mapping.get(key, default)

    max_rate = value(args.max_rate, limits, "max_rate_per_hour", None)
    max_cost = value(args.max_cost, limits, "max_total_cost", None)
    if max_rate is None or max_cost is None:
        raise VastGpuError("Provide --max-rate and --max-cost, or set them in the YAML config.")
    return Limits(
        float(max_rate),
        float(max_cost),
        float(value(args.max_hours, limits, "max_runtime_hours", 6)),
        int(value(args.min_vram, gpu, "min_vram_gb", 24)),
        float(value(args.min_reliability, config.get("vast", {}), "min_reliability", 0.98)),
        int(value(args.disk_gb, storage, "disk_gb", 100)),
        int(value(args.min_inet_down, gpu, "min_inet_down_mbps", 100)),
        int(value(args.min_inet_up, gpu, "min_inet_up_mbps", 50)),
        args.gpu,
    )


def run_args_from_config(args: argparse.Namespace) -> argparse.Namespace:
    config = load_config(args.config)
    run = config.get("run", {})
    if not isinstance(run, dict):
        raise VastGpuError("Config key 'run' must contain a YAML mapping.")

    def value(cli: Any, key: str, default: Any) -> Any:
        return cli if cli is not None else run.get(key, default)

    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        configured = run.get("command", [])
        command = shlex.split(configured) if isinstance(configured, str) else configured
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) for item in command)
    ):
        raise VastGpuError("Provide a remote command or set run.command to a string list.")
    transfer = value(args.dataset_transfer, "dataset_transfer", "rsync")
    dataset = value(args.dataset, "dataset", None)
    if transfer not in {"rsync", "tar", "cloud_copy", "none"}:
        raise VastGpuError("Config key 'run.dataset_transfer' is not supported.")
    if transfer in {"rsync", "tar"} and dataset is None:
        raise VastGpuError(
            "Provide --dataset unless run.dataset_transfer is 'none' or 'cloud_copy'."
        )
    downloads = value(args.download, "download", [])
    downloads = [downloads] if isinstance(downloads, str) else downloads
    if not isinstance(downloads, list) or not all(isinstance(item, str) for item in downloads):
        raise VastGpuError("Config key 'run.download' must be a string or list of strings.")
    cloud = run.get("cloud_copy", {})
    if transfer == "cloud_copy":
        if not isinstance(cloud, dict) or not cloud.get("source"):
            raise VastGpuError("Cloud copy requires run.cloud_copy.source.")
        if str(cloud["source"]).startswith("s3://") and not cloud.get("connection_id"):
            raise VastGpuError("An s3:// cloud copy requires run.cloud_copy.connection_id.")
    return argparse.Namespace(
        **{
            **vars(args),
            "dataset": Path(dataset) if dataset else None,
            "project": Path(value(args.project, "project", Path.cwd())),
            "output": Path(value(args.output, "output", "gpu-results")),
            "download": downloads,
            "image": value(args.image, "image", DEFAULT_IMAGE),
            "setup": value(args.setup, "setup", None),
            "ssh_key": Path(
                value(args.ssh_key, "ssh_key", "~/.ssh/vast_dynamic_router")
            ).expanduser(),
            "ssh_timeout": int(value(args.ssh_timeout, "ssh_timeout", 900)),
            "label": value(args.label, "label", "protein-state-router"),
            "dataset_transfer": transfer,
            "cloud_copy": cloud,
            "command": command,
        }
    )


def launch_paths(config: Path) -> tuple[Path, Path, Path]:
    run = load_config(config).get("run", {})
    output = Path(run.get("output", "gpu-results"))
    return output, output / "active-instance.json", output / "terminal.log"


def launch_monitor_command(config: Path, log: Path) -> str:
    command = load_config(config).get("run", {}).get("monitor_command", f"tail -f {log}")
    if not isinstance(command, str) or not command.strip():
        raise VastGpuError("Config key 'run.monitor_command' must be a non-empty string.")
    return command


def launch_job(config: Path) -> int:
    config = config.resolve()
    output, active, log = launch_paths(config)
    if active.exists():
        raise VastGpuError(
            f"Refusing to launch while instance {json.loads(active.read_text()).get('instance_id')} is recorded at {active}."
        )
    output.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "GPU_RUN_STATE_PATH": str(active.resolve())}
    with log.open("w") as handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "protein_state_router.vast_gpu", "run", "--config", str(config)],
            cwd=Path.cwd(),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (output / "controller.pid").write_text(f"{process.pid}\n")
    print(
        f"Launched controller {process.pid}. Follow progress with: {launch_monitor_command(config, log)}"
    )
    return 0


def run_job(args: argparse.Namespace, reporter: Callable[[str], None] | None = None) -> int:
    args, instance_id, failure = run_args_from_config(args), None, True
    report = reporter or LifecycleReporter(args.output / "controller.log")
    report(f"Started local Vast controller (pid={os.getpid()}).")
    try:
        if read_state():
            raise VastGpuError(
                "An active instance is already recorded. Run gpu-run status, then gpu-run cleanup."
            )
        for path, name in ((args.project, "project"), (args.dataset, "dataset")):
            if path is not None and not path.is_dir():
                raise VastGpuError(f"{name.capitalize()} directory does not exist: {path}")
        preflight_ssh_key(args.ssh_key)
        limits = limits_from_args(args)
        report("Searching compatible on-demand Vast offers.")
        offer = print_offers(search_offers(limits), limits, args.dry_run, report)
        if args.dry_run:
            return 0
        instance_id = create_instance(offer, limits, args.image, args.label)
        write_json_atomic(
            state_path(),
            {
                "instance_id": instance_id,
                "rate": offer.rate,
                "phase": "created",
                "created_at": time.time(),
                "limits": asdict(limits),
            },
        )
        report(f"Waiting for direct SSH on instance {instance_id}.")
        url = wait_for_ssh(instance_id, args.ssh_key, args.ssh_timeout)
        subprocess.run(
            _ssh_base(url, args.ssh_key)
            + ["install -d -m 0755 /workspace/project /workspace/data"],
            check=True,
        )
        report("Running remote CUDA preflight before uploads.")
        preflight_cuda(url, args.ssh_key, args.output / "cuda_preflight.log")
        update_state("uploading_project")
        report("Uploading project with resumable rsync.")
        rsync_to_remote(args.project, "/workspace/project", url, args.ssh_key, PROJECT_EXCLUDES)
        update_state("uploading_data")
        if args.dataset_transfer == "rsync":
            report("Uploading dataset with resumable rsync.")
            rsync_to_remote(args.dataset, "/workspace/data", url, args.ssh_key)
        elif args.dataset_transfer == "tar":
            report("Streaming dataset tar without a local archive.")
            tar_to_remote(args.dataset, "/workspace/data", url, args.ssh_key)
        elif args.dataset_transfer == "cloud_copy":
            report("Starting Vast cloud-to-instance copy.")
            connection_id = args.cloud_copy.get("connection_id")
            cloud_copy(
                str(args.cloud_copy["source"]),
                instance_id,
                "/workspace/data",
                int(connection_id) if connection_id is not None else None,
            )
        else:
            report("Skipping dataset transfer; the remote command owns its data download.")
        update_state("running")
        report("Starting remote command.")
        try:
            run_remote(
                args.command,
                url,
                args.ssh_key,
                limits.max_hours,
                args.setup,
                args.output / "terminal.log",
            )
            failure = False
            report("Remote command completed successfully.")
        finally:
            update_state("downloading")
            for path in args.download:
                safe = safe_output_path(path)
                if remote_exists(f"/workspace/project/{safe}", url, args.ssh_key):
                    rsync_from_remote(safe, args.output, url, args.ssh_key)
    finally:
        if instance_id is not None and not (failure and args.keep_on_failure):
            report(f"Destroying instance {instance_id}.")
            destroy_and_verify(instance_id)
            report(f"Verified destruction of instance {instance_id}.")
        elif instance_id is not None:
            report(
                f"WARNING: keeping instance {instance_id}; it continues billing. Run gpu-run cleanup."
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.action == "offers":
        print_offers(search_offers(limits_from_args(args)), limits_from_args(args), True, print)
        return 0
    if args.action == "run":
        return run_job(args)
    if args.action == "launch":
        return launch_job(args.config)
    if args.action == "status":
        print(json.dumps(read_state() or {"active": False}, indent=2))
        return 0
    instance_id = args.instance_id or (read_state() or {}).get("instance_id")
    if instance_id is None:
        raise VastGpuError("No active instance is recorded. Pass an instance ID explicitly.")
    destroy_and_verify(int(instance_id))
    return 0


def console_main() -> None:
    try:
        raise SystemExit(main())
    except VastGpuError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    console_main()
