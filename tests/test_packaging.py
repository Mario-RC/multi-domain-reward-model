import importlib.util
from pathlib import Path
import tempfile


def _package_module():
    script = Path(__file__).resolve().parents[1] / "stage-3_package_model.py"
    spec = importlib.util.spec_from_file_location("stage3_package_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identical_weight_shards_are_replaced_with_verified_hardlinks():
    module = _package_module()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        generated = root / "generated"
        reference = root / "reference"
        generated.mkdir()
        reference.mkdir()
        (generated / "model-00001.safetensors").write_bytes(b"a" * 60)
        (reference / "model-00001.safetensors").write_bytes(b"a" * 60)
        (generated / "model-00002.safetensors").write_bytes(b"b" * 40)
        (reference / "model-00002.safetensors").write_bytes(b"c" * 40)

        result = module._reuse_identical_weight_files(
            str(generated), str(reference), minimum_fraction=0.5,
        )

        assert result["reused_fraction"] == 0.6
        assert result["reused_files"] == ["model-00001.safetensors"]
        assert (generated / "model-00001.safetensors").samefile(
            reference / "model-00001.safetensors"
        )
        assert not (generated / "model-00002.safetensors").samefile(
            reference / "model-00002.safetensors"
        )


def test_weight_reuse_fails_closed_below_the_required_fraction():
    module = _package_module()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        generated = root / "generated"
        reference = root / "reference"
        generated.mkdir()
        reference.mkdir()
        (generated / "model.safetensors").write_bytes(b"generated")
        (reference / "model.safetensors").write_bytes(b"reference")

        try:
            module._reuse_identical_weight_files(
                str(generated), str(reference), minimum_fraction=0.5,
            )
        except RuntimeError as error:
            assert "below the required threshold" in str(error)
        else:
            raise AssertionError("Insufficient verified weight reuse was accepted.")


def test_staging_creates_missing_parents_and_reserves_exclusively():
    module = _package_module()
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / 'nested' / 'packages' / 'model'
        staging = Path(module._create_package_staging_directory(str(output)))
        assert staging.is_dir() and not output.exists()
        (staging/'sentinel').write_bytes(b'preserve')
        try:
            module._create_package_staging_directory(str(output))
        except FileExistsError:
            assert (staging/'sentinel').read_bytes() == b'preserve'
        else:
            raise AssertionError('Existing staging directory was reused.')


def test_staging_retries_transient_missing_parent_only():
    from unittest.mock import patch
    module = _package_module()
    with tempfile.TemporaryDirectory() as directory:
        output = str(Path(directory)/'release')
        actual_mkdir = module.os.mkdir
        calls = []
        def delayed(path, *args, **kwargs):
            if str(path) == output+'.incomplete':
                calls.append(path)
                if len(calls) == 1:
                    raise FileNotFoundError(2, 'injected parent visibility delay', str(path))
            return actual_mkdir(path, *args, **kwargs)
        with patch.object(module.os,'mkdir',side_effect=delayed), patch.object(module.time,'sleep') as sleep:
            staging = module._create_package_staging_directory(output)
            assert Path(staging).is_dir() and len(calls) == 2
            sleep.assert_called_once_with(1)


def test_staging_permission_errors_are_not_retried():
    from unittest.mock import patch
    module = _package_module()
    with tempfile.TemporaryDirectory() as directory:
        output = str(Path(directory)/'release')
        actual_mkdir = module.os.mkdir
        def denied(path, *args, **kwargs):
            if str(path) == output+'.incomplete':
                raise PermissionError(13, 'injected access denial', str(path))
            return actual_mkdir(path, *args, **kwargs)
        with patch.object(module.os,'mkdir',side_effect=denied), patch.object(module.time,'sleep') as sleep:
            try:
                module._create_package_staging_directory(output)
            except PermissionError:
                sleep.assert_not_called()
            else:
                raise AssertionError('Permission failure was suppressed.')


def test_staging_retry_limit_is_bounded():
    from unittest.mock import patch
    module = _package_module()
    with tempfile.TemporaryDirectory() as directory:
        output = str(Path(directory)/'release')
        actual_mkdir = module.os.mkdir
        calls = []
        def missing(path, *args, **kwargs):
            if str(path) == output+'.incomplete':
                calls.append(path)
                raise FileNotFoundError(2, 'injected persistent missing parent', str(path))
            return actual_mkdir(path, *args, **kwargs)
        with patch.object(module.os,'mkdir',side_effect=missing), patch.object(module.time,'sleep') as sleep:
            try:
                module._create_package_staging_directory(output, attempts=3)
            except FileNotFoundError:
                assert len(calls) == 3 and sleep.call_count == 2
            else:
                raise AssertionError('Persistent filesystem failure was suppressed.')


def test_staging_refuses_existing_output_and_dangling_staging_link():
    module = _package_module()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        completed = root/'completed'; completed.mkdir()
        (completed/'sentinel').write_bytes(b'preserve')
        output = root/'other'
        link = root/'other.incomplete'; link.symlink_to(root/'missing')
        for path in (completed, output):
            try:
                module._create_package_staging_directory(str(path))
            except FileExistsError:
                pass
            else:
                raise AssertionError('Existing output or dangling staging link was overwritten.')
        assert (completed/'sentinel').read_bytes() == b'preserve' and link.is_symlink()
