import os
import tempfile
from datetime import datetime

import numpy as np
import wandb
from PIL import Image, ImageEnhance


class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row['step'] = step
        filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
        if self.file is None:
            existing_header = self._read_existing_header()
            if existing_header is None:
                self.header = list(filtered_row)
                self.file = open(self.path, 'w')
                self.file.write(','.join(self.header) + '\n')
            else:
                # Append, don't truncate. save_dir now carries the
                # checkpoint-identity axes and experiment.repeat_id
                # (main_helper.setup_experiment_logging), so a CSV already
                # sitting here means the SAME config at the SAME seed — a
                # preemption resume, or a restart after a lost checkpoint. In
                # both cases the pre-existing rows are this run's own history
                # and opening 'w' destroys them. The old header is adopted
                # rather than re-derived so the column order survives the join;
                # keys this phase adds are handled below by _expand_header,
                # exactly as they are mid-run.
                self.header = existing_header
                self.file = open(self.path, 'a')
                # A kill INSIDE the write below (log() flushes after every
                # complete row, so nothing shorter can tear) leaves a row with
                # no terminator, and appending would glue this run's first row
                # onto it — _expand_header would then pad the fused line into a
                # plausible-looking bad row. Writing the missing terminator is
                # the cheap half of the fix: it costs one byte and an O(1)
                # stat, where truncating the torn row instead would mean
                # scanning back for the previous row boundary (a full read on a
                # run-long step_data.csv) to discard fields that are, up to the
                # tear, real logged values. The torn row survives as a SHORT
                # row — one row of this file may have fewer fields than the
                # header, and its last field may itself be partial.
                if not self._ends_with_newline():
                    self.file.write('\n')
        new_keys = [k for k in filtered_row if k not in self.header]
        if new_keys:
            self._expand_header(new_keys)
        self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

    def _read_existing_header(self):
        """Column names already on disk at self.path, or None if there are none.

        A zero-byte file (a run that opened the logger but died before its
        first row) carries no header to adopt, so it is treated as absent and
        overwritten — not as an empty header, which would push every real
        column through _expand_header.
        """
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return None
        with open(self.path, 'r', newline='') as f:
            first_line = f.readline()
        # rstrip('\n') only: the row terminator this class writes. See
        # _expand_header on why splitlines()-style breaking is wrong here.
        return first_line.rstrip('\n').split(',')

    def _ends_with_newline(self):
        """True if the file on disk ends in this class's row terminator.

        Reads the last byte only — the file can be arbitrarily long and this
        runs on every adopt-and-append. Callers only reach it for a non-empty
        file (_read_existing_header returned a header).
        """
        with open(self.path, 'rb') as f:
            f.seek(-1, os.SEEK_END)
            return f.read(1) == b'\n'

    def _expand_header(self, new_keys):
        """Append columns for keys first seen after row 1, rewriting the file.

        The header used to be frozen from the first row, so any key that appeared
        later was dropped for the rest of the run — which cost the 2026-08-04 fleet
        every actor metric in online_agent.csv (the first row was logged during
        q-warmup, when only critic keys exist). Growth is append-only, so existing
        column order — and anything reading these files by column name — is
        preserved; prior rows are padded with empty fields so the whole file keeps
        one column count.

        Re-reading the file rather than keeping every row in memory: step_data.csv
        grows for the whole run and RAM is the scarce resource on the SLURM nodes
        (see mem_monitor.py), so a run-long row list would be an unbounded leak,
        while an expansion happens only a handful of times per run, at phase
        boundaries — and costs one transient read of the file when it does.
        """
        with open(self.path, 'r', newline='') as f:
            # split('\n'), NOT splitlines(): splitlines() also breaks on \v, \f,
            # \x1c-\x1e, \x85 and U+2028/9, none of which this class writes as a
            # row terminator, so treating one as a row break would ragged-ify a
            # file that was still valid. newline='' likewise keeps a lone \r
            # inside its field instead of folding it into a break.
            lines = f.read().split('\n')
        if lines and lines[-1] == '':
            lines.pop()  # trailing terminator of the last row, not a row
        header = self.header + new_keys
        pad = ',' * len(new_keys)
        # Temp file + os.replace, the same atomic-swap pattern as the rolling
        # checkpoints (main_helper.py): a preemption or a full disk mid-rewrite
        # then leaves the original CSV intact instead of truncated. The live
        # handle is closed and self.header advanced only after the swap has
        # succeeded, so a failure propagates (fail-fast) with the logger still
        # writable at its current width rather than permanently bricked.
        tmp_path = self.path + '.tmp'
        with open(tmp_path, 'w') as f:
            f.write(','.join(header) + '\n')
            for line in lines[1:]:  # lines[0] is the old, shorter header
                f.write(line + pad + '\n')
        os.replace(tmp_path, self.path)
        self.file.close()
        self.header = header
        self.file = open(self.path, 'a')

    def close(self):
        if self.file is not None:
            self.file.close()


def get_exp_name(seed):
    """Return the experiment name."""
    exp_name = ''
    exp_name += f'sd{seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    return exp_name


def get_flag_dict(config=None):
    """Return the dictionary of config for logging.

    Args:
        config: Configuration dict/AttrDict to convert. If None, returns empty dict.

    Returns:
        Dictionary representation of the config.
    """
    if config is None:
        return {}

    # Convert AttrDict or nested dict to plain dict
    def to_dict(obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_dict(v) for v in obj]
        else:
            return obj

    return to_dict(config)


def setup_wandb(
    entity=None,
    project='project',
    group=None,
    name=None,
    mode='online',
    config=None,
    id=None,
    resume=None,
):
    """Set up Weights & Biases for logging.

    Args:
        entity: WandB entity (team/user).
        project: WandB project name.
        group: WandB run group.
        name: WandB run name.
        mode: WandB mode ('online', 'offline', 'disabled').
        config: Configuration dict to log.
        id: WandB run ID (for resuming).
        resume: Resume mode ('allow', 'must', 'never', etc.).
    """
    wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None

    init_kwargs = dict(
        config=get_flag_dict(config),
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_stats=False,
        ),
        mode=mode,
        save_code=True,
    )

    if id is not None:
        init_kwargs['id'] = id
    if resume is not None:
        init_kwargs['resume'] = resume

    run = wandb.init(**init_kwargs)

    return run


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video_file(frames, fps=15):
    """Encode (t, h, w, c) uint8 frames to a temp mp4 and return (wandb.Video, path).

    wandb.Video on a raw ndarray requires moviepy, which this project does not
    depend on; passing an mp4 file path instead needs only imageio-ffmpeg, which
    it does (pyproject.toml). wandb copies the file into the run dir when the
    Video is logged, so the caller may delete the temp file once log() returns.

    This is why get_wandb_video below is not reused for capture: it hands wandb a
    raw array and would fail at runtime here.

    fps is applied at encode time by imageio; it is deliberately NOT passed to
    wandb.Video, which ignores it for file inputs and warns once per video.
    """
    import imageio

    frames = np.asarray(frames)
    assert frames.dtype == np.uint8, f"frames must be uint8, got {frames.dtype}"

    f = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    f.close()
    with imageio.get_writer(f.name, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    return wandb.Video(f.name, format='mp4'), f.name


def get_wandb_video(renders=None, n_cols=None, fps=15):
    """Return a Weights & Biases video.

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    return wandb.Video(renders, fps=fps, format='mp4')
