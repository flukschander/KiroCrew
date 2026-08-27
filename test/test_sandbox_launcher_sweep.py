"""Regression tests for namespace launcher file placement and cleanup sweep.

Verifies:
  (a) namespace_argv() writes the launcher to ~/.kirocrew/run/ with PID in name
  (b) cleanup_stale_sandbox_profiles() removes dead-PID files (.py and .sb)
  (c) cleanup_stale_sandbox_profiles() removes old-mtime live-PID files (age-based)
  (d) cleanup_stale_sandbox_profiles() sweeps legacy /tmp files (age threshold only)
  (e) OverflowError from absurdly long PID strings is handled gracefully
  (f) makedirs-failure falls back to system tmpdir
  (g) run_dir is created with mode 0o700
  (h) non-conforming filenames are left untouched
  (i) the bind-mount source janitor reclaims dead-PID kirocrew_sb_* sources,
      keeps live-PID ones, and never follows a planted symlink
  (j) the generated launcher tags all four staging sites with _sb_prefix
"""

from __future__ import annotations

import ast
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from kiro_crew import platform_compat
from kiro_crew.sandbox import (
    _LAUNCHER_MAX_AGE_SECONDS,
    _SB_MOUNT_SRC_MAX_AGE_SECONDS,
    _build_launcher_script,
    _cleanup_stale_sandbox_mount_sources,
    _ensure_run_dir,
    cleanup_stale_sandbox_profiles,
    namespace_argv,
)


def _dead_pid() -> int:
    """A PID that is reliably not running, found by probing rather than hardcoded.

    A hardcoded "obviously dead" number is a latent flake: on a host with a
    raised ``pid_max`` it can genuinely be in use, and the test then asserts the
    opposite of what it means to.
    """
    for candidate in range(4_000_000, 4_001_000):
        if not platform_compat.pid_exists(candidate):
            return candidate
    raise AssertionError("no dead PID found in the probed range")


@pytest.fixture()
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect HOME so the sandbox run dir resolves under ``tmp_path/.kirocrew``.

    The run dir moved from ``os.path.expanduser("~")/".kirocrew"/"run"`` to
    ``config_dir()/run`` (data home now ``~/.kiro/crew``). ``config_dir()`` reads
    ``KIROCREW_HOME`` (pinned to a different tmp dir by conftest), so also
    redirect ``sandbox.config_dir`` to ``tmp_path/".kirocrew"`` — keeping the
    ``.kirocrew/run`` layout these tests assert. ``expanduser``/``HOME`` are still
    patched for the non-run-dir ``~`` lookups in this module.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path) + p[1:] if p.startswith("~") else p)
    monkeypatch.setattr("kiro_crew.sandbox.config_dir", lambda: tmp_path / ".kirocrew")
    return tmp_path


@pytest.fixture(autouse=True)
def _isolated_legacy_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the legacy /tmp sweep at an empty per-test dir.

    Without this, cleanup_stale_sandbox_profiles() sweeps the REAL /tmp and
    any stale kirocrew_sandbox_*.py files on the host inflate removal counts.
    Tests that exercise the legacy sweep pass legacy_dir= explicitly.
    """
    empty = tmp_path / "isolated_legacy"
    empty.mkdir(exist_ok=True)
    monkeypatch.setattr("kiro_crew.sandbox._LEGACY_LAUNCHER_DIR", str(empty))
    return empty


@pytest.fixture(autouse=True)
def _isolated_mount_source_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the bind-mount-source sweep at an empty per-test dir.

    Without this, cleanup_stale_sandbox_profiles() would scan the REAL
    /run/user/$UID, /dev/shm and system temp dir, so a unit test could delete
    a live agent's staging temps on the developer's own machine.
    """
    empty = tmp_path / "isolated_mount_src"
    empty.mkdir(exist_ok=True)
    monkeypatch.setattr("kiro_crew.sandbox._sb_mount_source_dirs", lambda: [str(empty)])
    return empty


class TestNamespaceArgvPlacement:
    """namespace_argv() launcher lands in ~/.kirocrew/run/ with PID."""

    @patch("kiro_crew.sandbox.detect_backend", return_value="namespace")
    def test_launcher_in_run_dir(self, _mock_detect, fake_home: Path):
        run_dir = fake_home / ".kirocrew" / "run"
        result = namespace_argv(["kiro-cli", "--version"])

        # Result should be [python, launcher_path, *real_argv]
        assert len(result) >= 2
        launcher = result[1]
        assert launcher.startswith(str(run_dir)), f"launcher {launcher} not under {run_dir}"

    @patch("kiro_crew.sandbox.detect_backend", return_value="namespace")
    def test_launcher_has_pid_in_name(self, _mock_detect, fake_home: Path):
        result = namespace_argv(["kiro-cli", "--version"])
        launcher = Path(result[1])
        # Filename pattern: kirocrew_sandbox_{pid}_{random}.py
        assert launcher.name.startswith(f"kirocrew_sandbox_{os.getpid()}_")
        assert launcher.suffix == ".py"

    @patch("kiro_crew.sandbox.detect_backend", return_value="namespace")
    def test_launcher_is_executable(self, _mock_detect, fake_home: Path):
        result = namespace_argv(["kiro-cli", "--version"])
        launcher = result[1]
        stat = os.stat(launcher)
        assert stat.st_mode & 0o700 == 0o700


class TestRunDirMode:
    """_ensure_run_dir() creates directory with 0o700 permissions."""

    def test_run_dir_mode_0o700(self, fake_home: Path):
        run_dir = _ensure_run_dir()
        stat = os.stat(run_dir)
        assert stat.st_mode & 0o777 == 0o700

    def test_run_dir_mode_enforced_on_existing(self, fake_home: Path):
        """Even if the dir already exists with wrong perms, chmod fixes it."""
        run_dir = fake_home / ".kirocrew" / "run"
        run_dir.mkdir(parents=True, mode=0o755)
        result = _ensure_run_dir()
        stat = os.stat(result)
        assert stat.st_mode & 0o777 == 0o700

    def test_makedirs_failure_falls_back(self, fake_home: Path, monkeypatch: pytest.MonkeyPatch):
        """If makedirs raises, fall back to system tmpdir."""
        # Create a regular file at the expected dir path to cause makedirs failure
        kirocrew_dir = fake_home / ".kirocrew"
        kirocrew_dir.mkdir(parents=True, exist_ok=True)
        # Put a regular file where "run" dir should be
        (kirocrew_dir / "run").write_text("blocker")

        run_dir = _ensure_run_dir()
        import tempfile
        assert run_dir == tempfile.gettempdir()


class TestCleanupSweep:
    """cleanup_stale_sandbox_profiles() sweeps both .py and .sb dead-PID files."""

    def test_removes_dead_pid_py(self, fake_home: Path):
        run_dir = fake_home / ".kirocrew" / "run"
        run_dir.mkdir(parents=True)
        # PID 99999999 is almost certainly dead
        dead_file = run_dir / "kirocrew_sandbox_99999999_abc123.py"
        dead_file.write_text("# dead launcher")

        removed = cleanup_stale_sandbox_profiles()
        assert not dead_file.exists()
        assert removed == 1

    def test_removes_dead_pid_sb(self, fake_home: Path):
        run_dir = fake_home / ".kirocrew" / "run"
        run_dir.mkdir(parents=True)
        dead_file = run_dir / "kirocrew_sandbox_99999999_xyz789.sb"
        dead_file.write_text("(version 1)")

        removed = cleanup_stale_sandbox_profiles()
        assert not dead_file.exists()
        assert removed == 1

    def test_removes_old_mtime_live_pid(self, fake_home: Path):
        """Age-based reaping: old file removed even if tagged PID is alive."""
        run_dir = fake_home / ".kirocrew" / "run"
        run_dir.mkdir(parents=True)
        # Use our own PID — definitely alive
        live_file = run_dir / f"kirocrew_sandbox_{os.getpid()}_old123.py"
        live_file.write_text("# old launcher")
        # Set mtime to 2 hours ago (well past threshold)
        old_time = time.time() - _LAUNCHER_MAX_AGE_SECONDS - 100
        os.utime(live_file, (old_time, old_time))

        removed = cleanup_stale_sandbox_profiles()
        assert not live_file.exists()
        assert removed == 1

    def test_keeps_fresh_live_pid(self, fake_home: Path):
        """Fresh file with live PID is kept."""
        run_dir = fake_home / ".kirocrew" / "run"
        run_dir.mkdir(parents=True)
        live_file = run_dir / f"kirocrew_sandbox_{os.getpid()}_fresh123.py"
        live_file.write_text("# fresh launcher")

        removed = cleanup_stale_sandbox_profiles()
        assert live_file.exists()
        assert removed == 0

    def test_keeps_live_pid_sb(self, fake_home: Path):
        run_dir = fake_home / ".kirocrew" / "run"
        run_dir.mkdir(parents=True)
        live_file = run_dir / f"kirocrew_sandbox_{os.getpid()}_live456.sb"
        live_file.write_text("(version 1)")

        removed = cleanup_stale_sandbox_profiles()
        assert live_file.exists()
        assert removed == 0

    def test_overflow_error_resilience(self, fake_home: Path):
        """Absurdly long digit string doesn't crash the sweep."""
        run_dir = fake_home / ".kirocrew" / "run"
        run_dir.mkdir(parents=True)
        # PID that exceeds sys.maxsize causing OverflowError in os.kill
        huge_pid = "9" * 30  # 30 digits > sys.maxsize on 64-bit
        bad_file = run_dir / f"kirocrew_sandbox_{huge_pid}_x.py"
        bad_file.write_text("# overflow")
        # Also add a normal dead-PID file to confirm sweep continues
        normal_dead = run_dir / "kirocrew_sandbox_99999999_y.py"
        normal_dead.write_text("# dead")

        removed = cleanup_stale_sandbox_profiles()
        # Both should be removed (huge via age fallback or error, normal via dead-PID)
        assert not normal_dead.exists()
        assert removed >= 1  # at least the normal dead one

    def test_legacy_tmp_sweep(self, fake_home: Path, tmp_path: Path):
        """Legacy /tmp files are swept by age threshold."""
        legacy_dir = tmp_path / "legacy_tmp"
        legacy_dir.mkdir()
        # Old legacy file
        old_legacy = legacy_dir / "kirocrew_sandbox_abc123.py"
        old_legacy.write_text("# old legacy")
        old_time = time.time() - _LAUNCHER_MAX_AGE_SECONDS - 100
        os.utime(old_legacy, (old_time, old_time))
        # Fresh legacy file (should be kept)
        fresh_legacy = legacy_dir / "kirocrew_sandbox_fresh.py"
        fresh_legacy.write_text("# fresh legacy")

        removed = cleanup_stale_sandbox_profiles(legacy_dir=str(legacy_dir))
        assert not old_legacy.exists()
        assert fresh_legacy.exists()
        assert removed == 1

    def test_legacy_sweep_ignores_non_py(self, fake_home: Path, tmp_path: Path):
        """Legacy sweep only touches .py files."""
        legacy_dir = tmp_path / "legacy_tmp2"
        legacy_dir.mkdir()
        non_py = legacy_dir / "kirocrew_sandbox_abc.txt"
        non_py.write_text("not a launcher")
        old_time = time.time() - _LAUNCHER_MAX_AGE_SECONDS - 100
        os.utime(non_py, (old_time, old_time))

        removed = cleanup_stale_sandbox_profiles(legacy_dir=str(legacy_dir))
        assert non_py.exists()
        assert removed == 0

    def test_ignores_nonconforming_filenames(self, fake_home: Path):
        run_dir = fake_home / ".kirocrew" / "run"
        run_dir.mkdir(parents=True)
        # Wrong prefix
        f1 = run_dir / "other_file_99999999_abc.py"
        f1.write_text("# unrelated")
        # Wrong suffix
        f2 = run_dir / "kirocrew_sandbox_99999999_abc.txt"
        f2.write_text("# unrelated")
        # No PID (no underscore separator)
        f3 = run_dir / "kirocrew_sandbox_nopid.py"
        f3.write_text("# unrelated")
        # Right prefix, right suffix, but non-digit PID
        f4 = run_dir / "kirocrew_sandbox_notapid_abc.py"
        f4.write_text("# unrelated")

        removed = cleanup_stale_sandbox_profiles()
        assert f1.exists()
        assert f2.exists()
        assert f3.exists()
        assert f4.exists()
        assert removed == 0

    def test_no_run_dir_is_noop(self, fake_home: Path):
        """If ~/.kirocrew/run/ doesn't exist, no crash."""
        removed = cleanup_stale_sandbox_profiles()  # should not raise
        assert removed == 0


class TestMountSourceSweep:
    """_cleanup_stale_sandbox_mount_sources() reclaims the launcher's bind-mount sources.

    The launcher stages one dir per hidden credential directory, one file per
    hidden file, and one dir holding the exposed known_hosts copy. It cannot
    unlink them itself (the kernel pins a bind-mount source for the life of the
    mount), so they are reclaimed here by PID liveness once the namespace is gone.
    """

    def test_removes_dead_pid_dir_with_contents(self, _isolated_mount_source_dirs: Path):
        """A non-empty dead-PID dir is removed — the real .ssh shape.

        The staged .ssh source holds the known_hosts copy written through the
        mount, so it is never empty. A naive os.rmdir implementation would fail
        ENOTEMPTY here and leak exactly the entries that filled the tmpfs.
        """
        stale = _isolated_mount_source_dirs / f"kirocrew_sb_{_dead_pid()}_ssh0"
        stale.mkdir()
        (stale / "known_hosts").write_text("github.com ssh-ed25519 AAAA...\n")

        removed = _cleanup_stale_sandbox_mount_sources()
        assert not stale.exists()
        assert removed == 1

    def test_removes_dead_pid_file(self, _isolated_mount_source_dirs: Path):
        """A dead-PID empty file source (the SENSITIVE_FILES shape) is removed."""
        stale = _isolated_mount_source_dirs / f"kirocrew_sb_{_dead_pid()}_f0"
        stale.write_bytes(b"")

        removed = _cleanup_stale_sandbox_mount_sources()
        assert not stale.exists()
        assert removed == 1

    def test_removes_over_age_live_pid(self, _isolated_mount_source_dirs: Path):
        """Age is the backstop for a PID recycled after the launcher died."""
        stale = _isolated_mount_source_dirs / f"kirocrew_sb_{os.getpid()}_old0"
        stale.mkdir()
        old_time = time.time() - _SB_MOUNT_SRC_MAX_AGE_SECONDS - 100
        os.utime(stale, (old_time, old_time))

        removed = _cleanup_stale_sandbox_mount_sources()
        assert not stale.exists()
        assert removed == 1

    def test_keeps_live_pid_within_age_window(self, _isolated_mount_source_dirs: Path):
        """A running agent's mount source is never disturbed."""
        live = _isolated_mount_source_dirs / f"kirocrew_sb_{os.getpid()}_live0"
        live.mkdir()
        (live / "known_hosts").write_text("still mounted\n")

        removed = _cleanup_stale_sandbox_mount_sources()
        assert live.exists()
        assert (live / "known_hosts").exists()
        assert removed == 0

    def test_keeps_legacy_bare_tmp_names(self, _isolated_mount_source_dirs: Path):
        """Bare tmp* is left alone — it belongs to every other app on the box.

        These dirs are world-writable (/dev/shm, /tmp) or shared with the whole
        session (/run/user/$UID), so matching tmp* would delete other programs'
        temps. Affected hosts get a documented one-time manual sweep instead.
        """
        legacy_dir = _isolated_mount_source_dirs / "tmpab12cd34"
        legacy_dir.mkdir()
        legacy_file = _isolated_mount_source_dirs / "tmpef56gh78"
        legacy_file.write_bytes(b"")
        old_time = time.time() - _SB_MOUNT_SRC_MAX_AGE_SECONDS - 100
        os.utime(legacy_dir, (old_time, old_time))
        os.utime(legacy_file, (old_time, old_time))

        removed = _cleanup_stale_sandbox_mount_sources()
        assert legacy_dir.exists()
        assert legacy_file.exists()
        assert removed == 0

    def test_keeps_unparseable_pid(self, _isolated_mount_source_dirs: Path):
        """A name whose PID segment is not digits is left to its owner."""
        odd = _isolated_mount_source_dirs / "kirocrew_sb_notapid_x"
        odd.mkdir()
        old_time = time.time() - _SB_MOUNT_SRC_MAX_AGE_SECONDS - 100
        os.utime(odd, (old_time, old_time))

        removed = _cleanup_stale_sandbox_mount_sources()
        assert odd.exists()
        assert removed == 0

    def test_keeps_launcher_scripts(self, _isolated_mount_source_dirs: Path):
        """kirocrew_sandbox_* belongs to the other sweeper, not this one."""
        launcher = _isolated_mount_source_dirs / "kirocrew_sandbox_1_x.py"
        launcher.write_text("# launcher script")
        old_time = time.time() - _SB_MOUNT_SRC_MAX_AGE_SECONDS - 100
        os.utime(launcher, (old_time, old_time))

        removed = _cleanup_stale_sandbox_mount_sources()
        assert launcher.exists()
        assert removed == 0

    def test_sweep_is_idempotent(self, _isolated_mount_source_dirs: Path):
        """A second pass finds nothing left to do."""
        stale = _isolated_mount_source_dirs / f"kirocrew_sb_{_dead_pid()}_d0"
        stale.mkdir()
        (stale / "known_hosts").write_text("x\n")

        assert _cleanup_stale_sandbox_mount_sources() == 1
        assert _cleanup_stale_sandbox_mount_sources() == 0

    def test_symlink_is_unlinked_and_target_survives(
        self, _isolated_mount_source_dirs: Path, tmp_path: Path
    ):
        """A planted symlink is removed AS a symlink, never followed.

        follow_symlinks=False on both the stat and the is_dir probe is the
        security property: without it, a symlink named kirocrew_sb_<deadpid>_x
        pointing at a real directory would have its TARGET rmtree'd.
        """
        target_dir = tmp_path / "precious"
        target_dir.mkdir()
        target_file = target_dir / "keep_me"
        target_file.write_text("must survive")

        planted = _isolated_mount_source_dirs / f"kirocrew_sb_{_dead_pid()}_link"
        planted.symlink_to(target_dir, target_is_directory=True)

        removed = _cleanup_stale_sandbox_mount_sources()
        assert not planted.exists()
        assert not planted.is_symlink()
        assert removed == 1
        assert target_dir.is_dir()
        assert target_file.read_text() == "must survive"

    def test_counted_in_cleanup_stale_sandbox_profiles(
        self, fake_home: Path, _isolated_mount_source_dirs: Path
    ):
        """The public sweep includes mount-source removals in its returned count."""
        run_dir = fake_home / ".kirocrew" / "run"
        run_dir.mkdir(parents=True)
        dead_launcher = run_dir / "kirocrew_sandbox_99999999_abc123.py"
        dead_launcher.write_text("# dead launcher")

        dead_source = _isolated_mount_source_dirs / f"kirocrew_sb_{_dead_pid()}_s0"
        dead_source.mkdir()

        removed = cleanup_stale_sandbox_profiles()
        assert not dead_launcher.exists()
        assert not dead_source.exists()
        assert removed == 2

    def test_missing_source_dir_is_noop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """A candidate dir that does not exist is skipped, not an error."""
        monkeypatch.setattr(
            "kiro_crew.sandbox._sb_mount_source_dirs", lambda: [str(tmp_path / "nope")]
        )
        assert _cleanup_stale_sandbox_mount_sources() == 0


class TestLauncherTagsMountSources:
    """Part A reaches the generated launcher.

    Asserted against the rendered script because the launcher body is emitted
    from an outer f-string: a brace-rendering mistake would silently drop the
    prefix and disable the whole fix with no test failing anywhere else.
    """

    @pytest.mark.parametrize("level", ["strict", "cc", "standard"])
    def test_launcher_parses_and_tags_every_staging_site(self, level: str):
        script = _build_launcher_script(level)

        ast.parse(script)  # raises SyntaxError on a brace-rendering bug
        assert "_sb_prefix = " in script

        # Every staging call must carry prefix=. An untagged one is invisible to
        # the janitor and leaks for the life of the host.
        assert "mkdtemp(dir=_tmpfs_src)" not in script
        assert "mkstemp(dir=_tmpfs_src)" not in script
        assert "mkdtemp(dir=_candidate)" not in script
        assert script.count("prefix=_sb_prefix") == 4
