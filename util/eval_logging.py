"""Flushed, per-rank evaluation diagnostics, independent of CUDA/NCCL state."""
import atexit
from contextlib import contextmanager
from datetime import datetime
import faulthandler
import os
import socket
import sys
import time

_STARTED = time.monotonic()
_HOST = socket.gethostname()


def log_stage(stage, **details):
    # stderr bypasses JiT's rank-zero-only print wrapper. Environment ranks survive
    # process-group destruction, so late shutdown messages retain their identity.
    rank = os.environ.get('RANK', os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
    local_rank = os.environ.get('LOCAL_RANK', os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
    fields = ' '.join('{}={!r}'.format(key, value) for key, value in details.items())
    sys.stderr.write('[eval-debug {} +{:.3f}s host={} rank={} local_rank={} pid={}] {} {}\n'.format(
        datetime.now().isoformat(timespec='milliseconds'), time.monotonic() - _STARTED,
        _HOST, rank, local_rank, os.getpid(), stage, fields))
    sys.stderr.flush()


@contextmanager
def trace_stage(stage, **details):
    start = time.monotonic()
    log_stage(stage + '.begin', **details)
    try:
        yield
    except BaseException as exc:
        log_stage(stage + '.error', elapsed_s=round(time.monotonic() - start, 3),
                  error_type=type(exc).__name__, error=str(exc), **details)
        raise
    else:
        log_stage(stage + '.end', elapsed_s=round(time.monotonic() - start, 3), **details)


def enable_exit_diagnostics():
    faulthandler.enable(all_threads=True)
    # Reaching this marker does not prove that later native destructors succeeded.
    atexit.register(log_stage, 'python.atexit')
    log_stage('diagnostics.enabled')
