"""Client for a remote robometer eval server.

Lightweight shim that lets QIPM rollouts fetch per-frame progress/success rewards
from a running `robometer/evals/eval_server.py` instance (typically on another
Babel node). Only depends on numpy and requests.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests


DEFAULT_URL_FILE = Path(os.environ.get("HOME", "/tmp")) / "QIPM" / ".robometer_server" / "server.url"


# Maps env_name prefix → natural-language task string for the progress model.
# Only envs listed here are supported when progress_rewards is enabled.
ENV_TASK_PROMPTS: Dict[str, str] = {
    "tool_hang": "Hang tool on the hook",
    "transport": "transport the hammer from the source to the target bin",
    "libero_KITCHEN_SCENE3": "turn on the stove and put the moka pot on it",
    "libero_KITCHEN_SCENE4": "put the black bowl in the bottom drawer of the cabinet and close it",
    "libero_KITCHEN_SCENE6": "put the yellow and white mug in the microwave and close it",
}


def get_task_for_env(env_name: str) -> str:
    """Return the robometer task prompt for a supported env_name.

    Matches by prefix so e.g. `tool_hang-ph-low_dim` and `tool_hang-ph-image`
    both resolve to the same task string.
    """
    for prefix, task in ENV_TASK_PROMPTS.items():
        if env_name.startswith(prefix):
            return task
    raise ValueError(
        f"No robometer task prompt registered for env '{env_name}'. "
        f"Supported prefixes: {list(ENV_TASK_PROMPTS.keys())}. "
        f"Add one in ogpo/utils/robometer_client.py::ENV_TASK_PROMPTS."
    )


def is_supported_env(env_name: str) -> bool:
    return any(env_name.startswith(p) for p in ENV_TASK_PROMPTS)


def resolve_server_url(
    explicit_url: Optional[str] = None,
    url_file: Optional[Path] = None,
    wait_timeout_s: float = 0.0,
    poll_interval_s: float = 5.0,
) -> str:
    """Find the eval server URL.

    Resolution order: explicit arg → $ROBOMETER_SERVER_URL → contents of url_file.
    If wait_timeout_s > 0 and the url_file is not yet present, polls until it is.
    """
    if explicit_url:
        return explicit_url.rstrip("/")

    env_url = os.environ.get("ROBOMETER_SERVER_URL")
    if env_url:
        return env_url.rstrip("/")

    url_file = Path(url_file) if url_file else DEFAULT_URL_FILE
    deadline = time.monotonic() + wait_timeout_s
    while True:
        if url_file.exists():
            url = url_file.read_text().strip()
            if url:
                return url.rstrip("/")
        if time.monotonic() >= deadline:
            raise FileNotFoundError(
                f"Robometer server URL not found at {url_file} and no explicit URL or "
                "$ROBOMETER_SERVER_URL set."
            )
        time.sleep(poll_interval_s)


def wait_for_server(url: str, timeout_s: float = 600.0, poll_interval_s: float = 5.0) -> None:
    """Block until GET {url}/health returns 200 or timeout elapses."""
    deadline = time.monotonic() + timeout_s
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            resp = requests.get(url.rstrip("/") + "/health", timeout=10.0)
            if resp.status_code == 200:
                return
        except requests.RequestException as e:
            last_err = e
        time.sleep(poll_interval_s)
    raise TimeoutError(f"Eval server at {url} not healthy within {timeout_s}s; last error: {last_err}")


def _numpy_to_npy_file_tuple(arr: np.ndarray, filename: str) -> Tuple[str, io.BytesIO, str]:
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return (filename, buf, "application/octet-stream")


def _build_multipart_payload(samples: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    files: Dict[str, Any] = {}
    data: Dict[str, str] = {}
    numpy_fields = ["frames", "lang_vector", "video_embeddings"]

    for i, sample in enumerate(samples):
        sample_copy = json.loads(json.dumps(sample, default=str))
        traj = sample.get("trajectory", {})
        traj_copy = sample_copy.get("trajectory", {})

        for field in numpy_fields:
            val = traj.get(field, None)
            if val is None:
                continue
            if hasattr(val, "detach") and hasattr(val, "cpu"):
                val = val.detach().cpu().numpy()
            if isinstance(val, np.ndarray):
                file_key = f"sample_{i}_trajectory_{field}"
                files[file_key] = _numpy_to_npy_file_tuple(val, f"{file_key}.npy")
                traj_copy[field] = {"__numpy_file__": file_key}
            else:
                traj_copy[field] = val

        if "frames_shape" in traj_copy and isinstance(traj_copy["frames_shape"], (tuple, list)):
            traj_copy["frames_shape"] = [int(x) for x in traj_copy["frames_shape"]]

        sample_copy["trajectory"] = traj_copy
        data[f"sample_{i}"] = json.dumps(sample_copy)

    return files, data


def _make_progress_sample(frames: np.ndarray, task: str, sample_id: str, subsequence_length: int) -> Dict[str, Any]:
    return {
        "sample_type": "progress",
        "trajectory": {
            "frames": frames,
            "frames_shape": tuple(frames.shape),
            "task": task,
            "id": sample_id,
            "metadata": {"subsequence_length": int(subsequence_length)},
            "video_embeddings": None,
        },
    }


def _extract_rewards(outputs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    outputs_progress = outputs.get("outputs_progress")
    if outputs_progress is None:
        raise ValueError("No `outputs_progress` in server response")
    progress_pred = outputs_progress.get("progress_pred", [])
    progress_array = (
        np.array(progress_pred[0], dtype=np.float32)
        if progress_pred and len(progress_pred) > 0
        else np.array([], dtype=np.float32)
    )

    outputs_success = outputs.get("outputs_success", {}) or {}
    success_probs = outputs_success.get("success_probs", [])
    success_array = (
        np.array(success_probs[0], dtype=np.float32)
        if success_probs and len(success_probs) > 0
        else np.array([], dtype=np.float32)
    )
    return progress_array, success_array


def post_evaluate_batch(
    server_url: str,
    samples: List[Dict[str, Any]],
    timeout_s: float = 120.0,
    use_frame_steps: bool = False,
    max_retries: int = 3,
    retry_backoff_s: float = 2.0,
) -> Dict[str, Any]:
    """POST samples to /evaluate_batch_npy, with simple retry on connection errors."""
    url = server_url.rstrip("/") + "/evaluate_batch_npy"
    attempt = 0
    while True:
        files, data = _build_multipart_payload(samples)
        data["use_frame_steps"] = "true" if use_frame_steps else "false"
        try:
            resp = requests.post(url, files=files, data=data, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(retry_backoff_s * (2 ** (attempt - 1)))


def compute_rewards(
    server_url: str,
    frames: np.ndarray,
    task: str,
    sample_id: str = "0",
    timeout_s: float = 120.0,
    use_frame_steps: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Send one trajectory; return (progress[T], success_probs[T])."""
    sample = _make_progress_sample(
        frames=frames,
        task=task,
        sample_id=sample_id,
        subsequence_length=int(frames.shape[0]),
    )
    outputs = post_evaluate_batch(server_url, [sample], timeout_s=timeout_s, use_frame_steps=use_frame_steps)
    return _extract_rewards(outputs)


def compute_rewards_batch(
    server_url: str,
    frames_list: List[np.ndarray],
    tasks: List[str],
    timeout_s: float = 300.0,
    use_frame_steps: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Batch multiple trajectories in a single request.

    Useful for rollout batches: one HTTP roundtrip per QIPM step instead of per env.
    """
    if len(frames_list) != len(tasks):
        raise ValueError(f"len(frames_list)={len(frames_list)} != len(tasks)={len(tasks)}")

    samples = [
        _make_progress_sample(frames=f, task=t, sample_id=str(i), subsequence_length=int(f.shape[0]))
        for i, (f, t) in enumerate(zip(frames_list, tasks))
    ]
    outputs = post_evaluate_batch(server_url, samples, timeout_s=timeout_s, use_frame_steps=use_frame_steps)

    progress_all = outputs.get("outputs_progress", {}).get("progress_pred", [])
    success_all = (outputs.get("outputs_success") or {}).get("success_probs", [])

    results: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(samples)):
        p = np.array(progress_all[i], dtype=np.float32) if i < len(progress_all) else np.array([], dtype=np.float32)
        s = np.array(success_all[i], dtype=np.float32) if i < len(success_all) else np.array([], dtype=np.float32)
        results.append((p, s))
    return results


class RobometerRewardClient:
    """High-level client for querying rollout progress rewards from the server.

    Handles env→task mapping, URL resolution, health wait, and trajectory →
    (progress, success) in a single object so callers (e.g. online RL runners)
    stay uncluttered.

    Usage:
        client = RobometerRewardClient.from_env(
            env_name="tool_hang-ph-low_dim",
            wait_s=1800.0,
        )
        progress, success = client.compute_from_trajectory(trajectory_buffer)
    """

    def __init__(
        self,
        server_url: str,
        task: str,
        timeout_s: float = 120.0,
        use_frame_steps: bool = False,
        image_key: str = "images",
        max_frames: int = 256,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.task = task
        self.timeout_s = float(timeout_s)
        self.use_frame_steps = bool(use_frame_steps)
        self.image_key = image_key
        self.max_frames = int(max_frames)
        self._episode_count = 0  # incremented per compute_from_trajectory call

    @classmethod
    def from_env(
        cls,
        env_name: str,
        server_url: Optional[str] = None,
        url_file: Optional[str] = None,
        wait_s: float = 1800.0,
        poll_interval_s: float = 5.0,
        timeout_s: float = 120.0,
        use_frame_steps: bool = False,
        image_key: str = "images",
        max_frames: int = 256,
    ) -> "RobometerRewardClient":
        """Build a client for an env_name. Validates env support, resolves URL,
        and blocks on /health up to `wait_s`."""
        if not is_supported_env(env_name):
            raise ValueError(
                f"env '{env_name}' is not supported for progress_rewards. "
                f"Supported prefixes: {list(ENV_TASK_PROMPTS.keys())}."
            )
        task = get_task_for_env(env_name)

        resolved_url = resolve_server_url(
            explicit_url=server_url,
            url_file=Path(url_file) if url_file else None,
            wait_timeout_s=wait_s,
            poll_interval_s=poll_interval_s,
        )
        print(f"[robometer] server URL: {resolved_url}")
        print(f"[robometer] task: '{task}'")

        if wait_s > 0:
            print(f"[robometer] waiting up to {wait_s:.0f}s for /health ...")
            wait_for_server(resolved_url, timeout_s=wait_s, poll_interval_s=poll_interval_s)
            print("[robometer] server healthy")

        return cls(
            server_url=resolved_url,
            task=task,
            timeout_s=timeout_s,
            use_frame_steps=use_frame_steps,
            image_key=image_key,
            max_frames=max_frames,
        )

    def _frames_from_trajectory(self, trajectory_buffer: List[Dict[str, Any]]) -> np.ndarray:
        if not trajectory_buffer:
            raise ValueError("trajectory_buffer is empty; no frames to send")
        if self.image_key not in trajectory_buffer[0]:
            raise KeyError(
                f"transition has no '{self.image_key}' field; "
                f"progress_rewards requires image observations."
            )
        frames_list = [np.asarray(t[self.image_key]) for t in trajectory_buffer]
        frames = np.stack(frames_list, axis=0)

        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        if frames.ndim == 4 and frames.shape[1] in (1, 3) and frames.shape[-1] not in (1, 3):
            frames = frames.transpose(0, 2, 3, 1)

        # Linspace-subsample max_frames evenly across the entire trajectory.
        # Effective fps depends on episode length: for a 500-step episode at 20 Hz
        # (= 25s), max_frames=64 gives ~2.56 fps coverage (one frame every ~0.39s).
        # Even coverage of the whole trajectory is what matters — the prior
        # "last-N-at-fixed-fps" scheme only showed the model the trajectory's tail
        # and made early-episode reward signal meaningless.
        T = frames.shape[0]
        if self.max_frames > 0 and T > self.max_frames:
            idx = np.linspace(0, T - 1, self.max_frames).round().astype(int)
            frames = frames[idx]

        # Save video at a viewer-friendly fps (env runs at 20 Hz; 5 fps playback is
        # easy to read and shows the linspace-sampled progression).
        self._save_trajectory_video(frames, fps=5)
        return frames

    def _save_trajectory_video(self, frames: np.ndarray, fps: int = 8) -> str:
        """Save the (T, H, W, C) uint8 frames as an mp4 and print the path."""
        import imageio.v2 as imageio

        default_dir = (
            "/data/group_data/maxlab/common_datasets/"
            f"{os.environ.get('USER', 'unknown')}/robometer/videos"
        )
        out_dir = Path(os.environ.get("ROBOMETER_VIDEO_DIR", default_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = str(out_dir / f"robometer_traj_ep{self._episode_count:05d}.mp4")
        imageio.mimwrite(path, list(frames), fps=fps, codec="libx264", quality=8)
        print(f"[robometer] trajectory video saved to: {path}")
        return path

    @staticmethod
    def _upsample(arr: np.ndarray, target_len: int) -> np.ndarray:
        if arr.size == 0 or target_len <= 0:
            return arr
        if arr.shape[0] == target_len:
            return arr
        src_x = np.linspace(0.0, 1.0, arr.shape[0])
        dst_x = np.linspace(0.0, 1.0, target_len)
        return np.interp(dst_x, src_x, arr).astype(np.float32)

    def _plot_rewards(self, progress: np.ndarray, success: np.ndarray) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        if progress.size > 0:
            ax.plot(progress, label="progress", color="tab:blue")
        if success.size > 0:
            ax.plot(success, label="success_prob", color="tab:orange")
        ax.set_xlabel("trajectory step")
        ax.set_ylabel("value")
        ax.set_title(f"robometer rewards | ep{self._episode_count:05d} | task='{self.task}'")
        ax.legend()
        ax.grid(True, alpha=0.3)

        default_dir = (
            "/data/group_data/maxlab/common_datasets/"
            f"{os.environ.get('USER', 'unknown')}/robometer/plots"
        )
        out_dir = Path(os.environ.get("ROBOMETER_PLOT_DIR", default_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = str(out_dir / f"robometer_rewards_ep{self._episode_count:05d}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path

    def compute_from_trajectory(
        self,
        trajectory_buffer: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compile frames from a completed trajectory and query the server.

        Frames are linearly subsampled to `max_frames` before sending; the server
        output is then upsampled back to `len(trajectory_buffer)` for per-step
        alignment. Also dumps a PNG plot of the curves and a side-by-side
        animation (frame + reward sweep) to the videos dir.
        """
        self._episode_count += 1
        target_len = len(trajectory_buffer)
        frames = self._frames_from_trajectory(trajectory_buffer)
        print(f"[FRAMES]: {frames.shape}")
        progress_raw, success_raw = compute_rewards(
            server_url=self.server_url,
            frames=frames,
            task=self.task,
            timeout_s=self.timeout_s,
            use_frame_steps=self.use_frame_steps,
        )
        progress = self._upsample(progress_raw, target_len)
        success = self._upsample(success_raw, target_len)

        plot_path = self._plot_rewards(progress, success)
        print(f"[robometer] reward plot saved to: {plot_path}")

        # Side-by-side animation aligned with the subsampled frames the server saw.
        try:
            self._save_combined_animation(frames, progress_raw, success_raw)
        except Exception as e:
            print(f"[robometer] combined animation skipped: {e}")

        return progress, success

    def _save_combined_animation(
        self,
        frames: np.ndarray,
        progress_arr: np.ndarray,
        success_arr: np.ndarray,
        fps: int = 5,
    ) -> str:
        """Render a side-by-side mp4: subsampled frame on the left, reward curves
        with a moving cursor on the right. One animation tick per server frame, so
        progress/success indices align 1-to-1 with frames.
        """
        import tempfile
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import imageio.v2 as imageio

        T = frames.shape[0]
        if T == 0 or progress_arr.shape[0] != T or success_arr.shape[0] != T:
            return ""

        default_dir = (
            "/data/group_data/maxlab/common_datasets/"
            f"{os.environ.get('USER', 'unknown')}/robometer/videos"
        )
        out_dir = Path(os.environ.get("ROBOMETER_VIDEO_DIR", default_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = str(out_dir / f"robometer_combined_ep{self._episode_count:05d}.mp4")

        # Render each tick: subsampled frame (left) + plot with cursor (right).
        H, W, _ = frames.shape[1:]
        fig_h = max(3.0, H / 100.0)
        out_frames = []
        for i in range(T):
            fig, axs = plt.subplots(1, 2, figsize=(2 * fig_h * (W / H + 0.2), fig_h))
            axs[0].imshow(frames[i])
            axs[0].set_axis_off()
            axs[0].set_title(f"frame {i+1}/{T}")
            axs[1].plot(progress_arr, label="progress", color="tab:blue")
            axs[1].plot(success_arr, label="success_prob", color="tab:orange")
            axs[1].axvline(i, color="red", linestyle="--", alpha=0.7)
            axs[1].scatter([i], [progress_arr[i]], color="tab:blue", zorder=5)
            axs[1].scatter([i], [success_arr[i]], color="tab:orange", zorder=5)
            axs[1].set_xlim(-0.5, T - 0.5)
            axs[1].set_ylim(-0.05, 1.05)
            axs[1].set_xlabel(f"server frame index (1 of {T} subsampled from full traj)")
            axs[1].set_title(f"task='{self.task}'", fontsize=9)
            axs[1].legend(loc="upper left", fontsize=8)
            axs[1].grid(True, alpha=0.3)
            fig.tight_layout()
            fig.canvas.draw()
            # Newer matplotlib: tostring_rgb is gone; use buffer_rgba then drop alpha.
            rgba = np.asarray(fig.canvas.buffer_rgba())
            out_frames.append(rgba[:, :, :3].copy())
            plt.close(fig)

        imageio.mimwrite(path, out_frames, fps=fps, codec="libx264", quality=8)
        print(f"[robometer] combined animation saved to: {path}")
        return path
