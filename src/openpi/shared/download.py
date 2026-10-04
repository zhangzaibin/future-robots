import concurrent.futures
import datetime
import logging
import os
import pathlib
import re
import shutil
import stat
import time
import urllib.parse

import filelock
import fsspec
import fsspec.generic
import tqdm_loggable.auto as tqdm

# Environment variable to control cache directory path, ~/.cache/openpi will be used by default.
_OPENPI_DATA_HOME = "OPENPI_DATA_HOME"
DEFAULT_CACHE_DIR = "~/.cache/openpi"

logger = logging.getLogger(__name__)


def get_cache_dir() -> pathlib.Path:
    cache_dir = pathlib.Path(os.getenv(_OPENPI_DATA_HOME, DEFAULT_CACHE_DIR)).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_folder_permission(cache_dir)
    return cache_dir


def maybe_download(url: str, *, force_download: bool = False, **kwargs) -> pathlib.Path:
    """Download a file or directory from a remote filesystem to the local cache, and return the local path.

    If the local file already exists, it will be returned directly.

    It is safe to call this function concurrently from multiple processes.
    See `get_cache_dir` for more details on the cache directory.

    Args:
        url: URL to the file to download.
        force_download: If True, the file will be downloaded even if it already exists in the cache.
        **kwargs: Additional arguments to pass to fsspec.

    Returns:
        Local path to the downloaded file or directory. That path is guaranteed to exist and is absolute.
    """
    # Don't use fsspec to parse the url to avoid unnecessary connection to the remote filesystem.
    parsed = urllib.parse.urlparse(url)

    # Short circuit if this is a local path.
    if parsed.scheme == "":
        path = pathlib.Path(url)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {url}")
        return path.resolve()

    cache_dir = get_cache_dir()

    local_path = cache_dir / parsed.netloc / parsed.path.strip("/")
    local_path = local_path.resolve()

    # Check if the cache should be invalidated.
    invalidate_cache = False
    if local_path.exists():
        if force_download or _should_invalidate_cache(cache_dir, local_path):
            invalidate_cache = True
        else:
            return local_path

    try:
        lock_path = local_path.with_suffix(".lock")
        with filelock.FileLock(lock_path):
            # Ensure consistent permissions for the lock file.
            _ensure_permissions(lock_path)
            # First, remove the existing cache if it is expired.
            if invalidate_cache:
                logger.info(f"Removing expired cached entry: {local_path}")
                if local_path.is_dir():
                    shutil.rmtree(local_path)
                else:
                    local_path.unlink()

            # Download the data to a local cache.
            logger.info(f"Downloading {url} to {local_path}")
            scratch_path = local_path.with_suffix(".partial")
            _download_fsspec(url, scratch_path, **kwargs)

            shutil.move(scratch_path, local_path)
            _ensure_permissions(local_path)

    except PermissionError as e:
        msg = (
            f"Local file permission error was encountered while downloading {url}. "
            f"Please try again after removing the cached data using: `rm -rf {local_path}*`"
        )
        raise PermissionError(msg) from e

    return local_path


def _download_fsspec(url: str, local_path: pathlib.Path, max_retries: int = 3, **kwargs) -> None:
    """Download a file from a remote filesystem to the local cache, and return the local path."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Connecting to {url}... (attempt {attempt + 1}/{max_retries})")
            fs, _ = fsspec.core.url_to_fs(url, **kwargs)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                error_msg = (
                    f"Failed to connect to {url} after {max_retries} attempts. "
                    f"This might be due to network issues or authentication problems. "
                    f"Error: {e}\n\n"
                    f"Alternative download methods:\n"
                    f"1. Use gsutil: gsutil -m cp -r '{url}' '{local_path.parent}'\n"
                    f"2. Run: bash scripts/download_checkpoint.sh <model_name>\n"
                    f"3. See docs/manual_download.md for more options"
                )
                if url.startswith("gs://"):
                    error_msg += (
                        '\n4. For public GCS buckets, try adding: gs={"token": "anon"} '
                        "as a parameter to maybe_download()."
                    )
                logger.error(error_msg)
                raise ConnectionError(error_msg) from e
            logger.warning(f"Connection attempt {attempt + 1} failed, retrying...")
            time.sleep(2**attempt)  # Exponential backoff

    logger.info(f"Getting file info for {url}...")
    try:
        info = fs.info(url)
    except Exception as e:
        error_msg = (
            f"Failed to get file info from {url}. "
            f"This might be due to network issues, authentication problems, or the file doesn't exist. "
            f"Error: {e}"
        )
        logger.error(error_msg)
        raise ConnectionError(error_msg) from e

    # Folders are represented by 0-byte objects with a trailing forward slash.
    if is_dir := (info["type"] == "directory" or (info["size"] == 0 and info["name"].endswith("/"))):
        logger.info(f"Calculating directory size for {url}...")
        try:
            total_size = fs.du(url)
        except Exception as e:
            error_msg = f"Failed to calculate directory size for {url}. " f"Error: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
    else:
        total_size = info["size"]

    logger.info(f"Starting download of {total_size} bytes from {url} to {local_path}...")
    with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fs.get, url, local_path, recursive=is_dir)

        # Add timeout handling
        timeout_seconds = 300  # 5 minutes timeout for the download
        start_time = time.time()
        last_size = 0
        no_progress_count = 0

        while not future.done():
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                future.cancel()
                error_msg = (
                    f"Download timeout after {timeout_seconds} seconds. "
                    f"The download might be stuck or the network connection is too slow."
                )
                logger.error(error_msg)
                raise TimeoutError(error_msg)

            try:
                current_size = sum(f.stat().st_size for f in [*local_path.rglob("*"), local_path] if f.is_file())
                pbar.update(current_size - pbar.n)

                # Check for progress
                if current_size == last_size:
                    no_progress_count += 1
                    if no_progress_count > 60:  # No progress for 60 seconds
                        logger.warning(
                            f"No download progress detected for 60 seconds. Current size: {current_size} bytes"
                        )
                        no_progress_count = 0
                else:
                    no_progress_count = 0
                    last_size = current_size
            except (FileNotFoundError, OSError):
                # Files might not exist yet, this is normal
                pass

            time.sleep(1)

        # Check for exceptions in the future
        try:
            future.result()
        except Exception as e:
            error_msg = (
                f"Download failed: {e}. "
                f"This might be due to network issues, authentication problems, or insufficient disk space."
            )
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

        pbar.update(total_size - pbar.n)
    logger.info(f"Download completed successfully to {local_path}")


def _set_permission(path: pathlib.Path, target_permission: int):
    """chmod requires executable permission to be set, so we skip if the permission is already match with the target."""
    if path.stat().st_mode & target_permission == target_permission:
        logger.debug(f"Skipping {path} because it already has correct permissions")
        return
    path.chmod(target_permission)
    logger.debug(f"Set {path} to {target_permission}")


def _set_folder_permission(folder_path: pathlib.Path) -> None:
    """Set folder permission to be read, write and searchable."""
    _set_permission(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


def _ensure_permissions(path: pathlib.Path) -> None:
    """Since we are sharing cache directory with containerized runtime as well as training script, we need to
    ensure that the cache directory has the correct permissions.
    """

    def _setup_folder_permission_between_cache_dir_and_path(path: pathlib.Path) -> None:
        cache_dir = get_cache_dir()
        relative_path = path.relative_to(cache_dir)
        moving_path = cache_dir
        for part in relative_path.parts:
            _set_folder_permission(moving_path / part)
            moving_path = moving_path / part

    def _set_file_permission(file_path: pathlib.Path) -> None:
        """Set all files to be read & writable, if it is a script, keep it as a script."""
        file_rw = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
        if file_path.stat().st_mode & 0o100:
            _set_permission(file_path, file_rw | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        else:
            _set_permission(file_path, file_rw)

    _setup_folder_permission_between_cache_dir_and_path(path)
    for root, dirs, files in os.walk(str(path)):
        root_path = pathlib.Path(root)
        for file in files:
            file_path = root_path / file
            _set_file_permission(file_path)

        for dir in dirs:
            dir_path = root_path / dir
            _set_folder_permission(dir_path)


def _get_mtime(year: int, month: int, day: int) -> float:
    """Get the mtime of a given date at midnight UTC."""
    date = datetime.datetime(year, month, day, tzinfo=datetime.UTC)
    return time.mktime(date.timetuple())


# Map of relative paths, defined as regular expressions, to expiration timestamps (mtime format).
# Partial matching will be used from top to bottom and the first match will be chosen.
# Cached entries will be retained only if they are newer than the expiration timestamp.
_INVALIDATE_CACHE_DIRS: dict[re.Pattern, float] = {
    re.compile("openpi-assets/checkpoints/pi0_aloha_pen_uncap"): _get_mtime(2025, 2, 17),
    re.compile("openpi-assets/checkpoints/pi0_libero"): _get_mtime(2025, 2, 6),
    re.compile("openpi-assets/checkpoints/"): _get_mtime(2025, 2, 3),
}


def _should_invalidate_cache(cache_dir: pathlib.Path, local_path: pathlib.Path) -> bool:
    """Invalidate the cache if it is expired. Return True if the cache was invalidated."""

    assert local_path.exists(), f"File not found at {local_path}"

    relative_path = str(local_path.relative_to(cache_dir))
    for pattern, expire_time in _INVALIDATE_CACHE_DIRS.items():
        if pattern.match(relative_path):
            # Remove if not newer than the expiration timestamp.
            return local_path.stat().st_mtime <= expire_time

    return False
