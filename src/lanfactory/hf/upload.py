"""Upload utilities for HuggingFace Hub.

This module provides functions to upload trained LANfactory models
to HuggingFace Hub with proper organization and metadata.
"""

import json
import logging
import tempfile
from pathlib import Path

from lanfactory.hf import DEFAULT_REPO_ID, VALID_NETWORK_TYPES

logger = logging.getLogger(__name__)

# Default file patterns to include in uploads
DEFAULT_INCLUDE_PATTERNS = [
    "*.onnx",
    "*.pt",
    "*.jax",
    "*_config.pickle",
    "*_data_details.pickle",
    "validation_report.json",
    "*.csv",
    "model_card.yaml",
]

# Root-level filename suffix per network type. This mirrors HSSM's
# ``missing_data_networks_suffix`` (src/hssm/defaults.py): HSSM builds
# ``f"{model_name}{suffix}.onnx"`` and passes it straight to
# ``hf_hub_download(repo_id="franklab/HSSM", filename=...)``, i.e. it resolves
# a file at the *repository root*. Keep these in lockstep.
ROOT_ONNX_SUFFIX = {
    "lan": "",
    "cpn": "_cpn",
    "opn": "_opn",
    "gonogo": "_gonogo",
}

MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1


def canonical_root_filename(network_type: str, model_name: str) -> str:
    """The repo-root filename HSSM will look for.

    HSSM downloads likelihood networks by constructing this name; a network
    uploaded only into ``{network_type}/{model_name}/`` is invisible to it.
    """
    if network_type not in ROOT_ONNX_SUFFIX:
        raise ValueError(
            f"No root filename convention for network_type {network_type!r}; "
            f"known: {sorted(ROOT_ONNX_SUFFIX)}"
        )
    return f"{model_name}{ROOT_ONNX_SUFFIX[network_type]}.onnx"


def select_canonical_onnx(files: list[Path], model_name: str) -> Path:
    """Pick the ONNX artifact to publish at the repo root.

    The chosen file is published under a name that released HSSM versions
    download, so picking the wrong one silently swaps a production likelihood.
    The artifact name must therefore *corroborate* ``model_name``: LANfactory's
    trainers embed the model in every filename
    (``{model}_{network_type}_{uuid}__model.onnx``), so a mismatch means the
    folder and the ``--model-name`` disagree — usually a copy-paste or a
    scripted-loop slip. Pass ``canonical_onnx=`` to override deliberately.
    """
    onnx_files = sorted(f for f in files if f.suffix == ".onnx")
    if not onnx_files:
        raise FileNotFoundError(
            "No .onnx file found to publish at the repository root. HSSM loads "
            "networks by their root filename, so an upload without one is not "
            "consumable."
        )

    matching = [f for f in onnx_files if model_name in f.name]
    if len(matching) == 1:
        return matching[0]
    if len(matching) > 1:
        raise ValueError(
            f"Cannot determine which ONNX file is canonical for {model_name!r}: "
            f"{[f.name for f in matching]}. Upload from a folder with a single "
            "network, or pass canonical_onnx=<path> explicitly."
        )

    raise ValueError(
        f"No ONNX artifact in this folder names the model {model_name!r}: "
        f"{[f.name for f in onnx_files]}. Publishing one of these at the "
        f"repository root would overwrite {model_name}'s network with a "
        "different model's weights. Check --model-name, or pass "
        "canonical_onnx=<path> to override deliberately."
    )


def build_manifest_entry(
    network_type: str,
    model_name: str,
    root_filename: str | None,
    folder_path: str,
    files: list[Path],
    extra: dict | None = None,
) -> dict:
    """One manifest record describing a published network."""
    entry = {
        "model": model_name,
        "network_type": network_type,
        "onnx_root": root_filename,
        "folder": folder_path,
        "files": sorted(f.name for f in files),
    }
    if extra:
        entry.update(extra)
    return entry


def merge_manifest(existing: dict | None, entry: dict) -> dict:
    """Read-modify-write a root manifest, replacing any entry for this key.

    Keyed by (network_type, model): re-publishing a network updates its record
    in place rather than appending a duplicate.
    """
    manifest = dict(existing or {})
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    networks = list(manifest.get("networks", []))
    key = (entry["network_type"], entry["model"])
    networks = [n for n in networks if (n.get("network_type"), n.get("model")) != key]
    networks.append(entry)
    manifest["networks"] = sorted(
        networks, key=lambda n: (n.get("network_type", ""), n.get("model", ""))
    )
    return manifest


def upload_model(
    model_folder: Path,
    network_type: str,
    model_name: str,
    repo_id: str = DEFAULT_REPO_ID,
    commit_message: str = "Upload model",
    private: bool = False,
    create_repo: bool = False,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    revision: str | None = None,
    token: str | None = None,
    dry_run: bool = False,
    publish_root_alias: bool = True,
    update_manifest: bool = True,
    require_model_card: bool = False,
    canonical_onnx: Path | None = None,
    overwrite_root: bool = False,
) -> str | None:
    """Upload a trained model to HuggingFace Hub.

    Parameters
    ----------
    model_folder : Path
        Path to the folder containing trained model artifacts.
    network_type : str
        Network type (e.g., "lan", "cpn", "opn").
    model_name : str
        Model name (e.g., "ddm", "angle").
    repo_id : str
        HuggingFace repository ID (default: "franklab/HSSM").
    commit_message : str
        Git commit message for the upload.
    private : bool
        Whether to create a private repository.
    create_repo : bool
        Whether to create the repository if it doesn't exist.
    include_patterns : list[str] | None
        Glob patterns for files to include.
    exclude_patterns : list[str] | None
        Glob patterns for files to exclude.
    revision : str | None
        Branch or tag name for versioning.
    token : str | None
        HuggingFace API token.
    dry_run : bool
        If True, show what would be uploaded without uploading.
    publish_root_alias : bool
        Also publish the canonical ONNX at the repository root under the
        filename HSSM resolves (``{model}{suffix}.onnx``). On by default: an
        upload without it is not consumable by any released HSSM version.
    update_manifest : bool
        Read-modify-write ``manifest.json`` at the repository root to record
        this network.
    require_model_card : bool
        Fail when ``model_card.yaml`` is absent instead of generating one from
        the artifact pickles.
    canonical_onnx : Path | None
        Explicit choice of the ONNX artifact to publish at the root; inferred
        from the folder when its name corroborates ``model_name``.
    overwrite_root : bool
        Allow replacing a root network that is already published. Off by
        default: every released HSSM downloads root filenames from ``main``
        without pinning a revision, so replacing one changes the likelihood
        for all existing users.

    Returns
    -------
    str | None
        URL of the uploaded model, or None if dry_run is True.

    Raises
    ------
    ImportError
        If huggingface_hub is not installed.
    FileNotFoundError
        If model_folder doesn't exist or is missing required files.
    ValueError
        If network_type is not valid.
    """
    # Validate inputs
    model_folder = Path(model_folder)
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder does not exist: {model_folder}")

    if network_type not in VALID_NETWORK_TYPES:
        raise ValueError(
            f"network_type must be one of {list(VALID_NETWORK_TYPES)}, got: {network_type}"
        )

    # model_card.yaml: generate a minimal one from the artifacts when absent.
    # The card's substance (architecture, training config) is recovered from
    # the pickles by load_model_card_yaml's _fill_from_pickle_configs anyway,
    # so hard-failing here blocked every automated publish for a file the
    # library can write itself. The file is only *written* on the real upload
    # path — a dry run must not modify the artifact folder.
    model_card_path = model_folder / "model_card.yaml"
    will_generate_model_card = not model_card_path.exists()
    if will_generate_model_card and require_model_card:
        raise FileNotFoundError(
            f"model_card.yaml not found in {model_folder} and "
            "require_model_card=True. Create the file, or allow it to be "
            "generated from the artifact configs."
        )

    # Use default patterns if not specified
    if include_patterns is None:
        include_patterns = DEFAULT_INCLUDE_PATTERNS

    # Collect files to upload
    files_to_upload = _collect_files(model_folder, include_patterns, exclude_patterns)

    if not files_to_upload:
        raise FileNotFoundError(
            f"No files matching patterns {include_patterns} found in {model_folder}"
        )

    # Dual layout: the full artifact set lives under {network_type}/{model}/,
    # and the canonical ONNX is ALSO published at the repository root under
    # the name HSSM resolves. Root placement is what every released HSSM
    # version can actually load; the folder carries everything else.
    path_in_repo = f"{network_type}/{model_name}"
    root_filename = None
    root_source = None
    if publish_root_alias:
        root_source = (
            Path(canonical_onnx)
            if canonical_onnx is not None
            else select_canonical_onnx(files_to_upload, model_name)
        )
        if not root_source.exists():
            raise FileNotFoundError(f"canonical_onnx does not exist: {root_source}")
        root_filename = canonical_root_filename(network_type, model_name)

    logger.info(f"Upload destination: {repo_id}/{path_in_repo}")
    if root_filename is not None and root_source is not None and not dry_run:
        # WARNING, not INFO: the CLI defaults to WARNING, and writing a root
        # network is the consequential half of this operation — it must never
        # happen silently.
        logger.warning(
            f"Publishing root network {repo_id}/{root_filename} "
            f"(from {root_source.name}) — this is the file HSSM downloads."
        )
    logger.info(f"Files to upload ({len(files_to_upload)}):")
    for f in files_to_upload:
        logger.info(f"  - {f.name}")

    if dry_run:
        logger.info("DRY RUN: No files were uploaded.")
        print(
            f"\nDRY RUN: Would upload {len(files_to_upload)} files to "
            f"{repo_id}/{path_in_repo}"
        )
        for f in files_to_upload:
            print(f"  - {f.name}")
        if will_generate_model_card:
            print("  - model_card.yaml (generated at upload time)")
        if root_filename is not None and root_source is not None:
            print(f"  - {root_filename} (root alias of {root_source.name})")
            print(
                f"    NOTE: {root_filename} is the file HSSM downloads for "
                f"{model_name}. If it already exists in {repo_id}, the real "
                "upload refuses unless --overwrite-root is given."
            )
        if update_manifest:
            print(f"  - {MANIFEST_FILENAME} (updated at root)")
        return None

    return _upload_to_hf(  # pragma: no cover
        model_folder=model_folder,
        model_name=model_name,
        network_type=network_type,
        files_to_upload=files_to_upload,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        commit_message=commit_message,
        private=private,
        create_repo=create_repo,
        revision=revision,
        token=token,
        root_filename=root_filename,
        root_source=root_source,
        update_manifest=update_manifest,
        overwrite_root=overwrite_root,
        will_generate_model_card=will_generate_model_card,
    )


def plan_upload_placements(
    files_to_upload: list[Path],
    path_in_repo: str,
    root_filename: str | None,
    root_source: Path | None,
    manifest_path: Path | None,
) -> list[tuple[str, Path]]:
    """Map every local file to its destination path in the repository.

    Returns ``(path_in_repo, local_path)`` pairs: the full artifact set under
    ``{network_type}/{model}/``, plus the canonical ONNX duplicated at the
    root under HSSM's expected filename, plus the manifest.
    """
    placements: list[tuple[str, Path]] = []
    seen: set[str] = set()

    def add(destination: str, source: Path) -> None:
        # Two operations for one destination make huggingface_hub warn and
        # apply them in order; dedup so the payload says what it means.
        if destination in seen:
            return
        seen.add(destination)
        placements.append((destination, source))

    for f in files_to_upload:
        add(f"{path_in_repo}/{f.name}", f)
    if root_filename is not None:
        if root_source is None:
            raise ValueError("root_filename given without root_source")
        add(root_filename, root_source)
    if manifest_path is not None:
        add(MANIFEST_FILENAME, manifest_path)
    return placements


def write_default_model_card(
    model_folder: Path, network_type: str, model_name: str
) -> Path:
    """Write a minimal model_card.yaml so an automated publish can proceed.

    Only the descriptive shell is written here; ``load_model_card_yaml`` fills
    architecture and training details from the artifact pickles when it reads
    the card back.
    """
    import yaml

    card = {
        "tags": [network_type, "ssm", "hssm"],
        "library_name": "onnx",
        "license": "mit",
        "title": f"{model_name} ({network_type.upper()})",
        "description": (
            f"{network_type.upper()} for the {model_name} sequential sampling "
            "model, trained with LANfactory. This card was generated at upload "
            "time from the training artifacts."
        ),
    }
    path = model_folder / "model_card.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(card, f, sort_keys=False)
    return path


class ManifestUnavailableError(RuntimeError):
    """The existing manifest could not be read, so it cannot be updated safely."""


class RootArtifactExistsError(RuntimeError):
    """A root network of this name is already published and would be replaced."""


def existing_root_filename(
    manifest: dict | None, network_type: str, model_name: str
) -> str | None:
    """The root filename a previous publish recorded for this network, if any."""
    for record in (manifest or {}).get("networks", []):
        if (record.get("network_type"), record.get("model")) == (
            network_type,
            model_name,
        ):
            return record.get("onnx_root")
    return None


def list_root_files(  # pragma: no cover - network path
    api, repo_id: str, revision: str | None
) -> set[str]:
    """Filenames at the repository root (no directory component)."""
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        files = api.list_repo_files(repo_id=repo_id, revision=revision)
    except RepositoryNotFoundError:
        return set()  # brand-new repo: nothing to overwrite
    return {name for name in files if "/" not in name}


def fetch_existing_manifest(  # pragma: no cover - network path
    repo_id: str, revision: str | None, token: str | None
) -> dict | None:
    """Read the current root manifest, or None when the repo genuinely has none.

    The distinction matters more than it looks: the manifest is rewritten
    wholesale from ``merge_manifest(existing, entry)``, so treating "I could
    not tell" as "there is none" would republish a manifest listing only the
    network being uploaded — silently erasing every previously published one.
    Only a definitive *absent* answer returns None; anything else raises.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=MANIFEST_FILENAME,
            revision=revision,
            token=token,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None  # definitively no manifest yet
    except Exception as e:
        raise ManifestUnavailableError(
            f"Could not read the existing {MANIFEST_FILENAME} from {repo_id}: "
            f"{e}. Refusing to continue, because rewriting the manifest now "
            "would drop every network already recorded in it. Retry when the "
            "Hub is reachable, or pass update_manifest=False "
            "(--no-update-manifest) to publish without touching it."
        ) from e
    with open(path) as f:
        return json.load(f)


def _upload_to_hf(  # pragma: no cover
    model_folder: Path,
    model_name: str,
    network_type: str,
    files_to_upload: list[Path],
    path_in_repo: str,
    repo_id: str,
    commit_message: str,
    private: bool,
    create_repo: bool,
    revision: str | None,
    token: str | None,
    root_filename: str | None = None,
    root_source: Path | None = None,
    update_manifest: bool = True,
    overwrite_root: bool = False,
    will_generate_model_card: bool = False,
) -> str:
    """HF-dependent implementation of upload_model."""
    try:
        from huggingface_hub import HfApi, create_repo as hf_create_repo
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for HuggingFace uploads. "
            "Install it with: pip install lanfactory[hf]"
        ) from exc

    api = HfApi(token=token)

    if create_repo:
        try:
            hf_create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True,
                token=token,
            )
            logger.info(f"Repository created/verified: {repo_id}")
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            raise

    # Refuse to silently replace a network that released HSSM versions are
    # already downloading. The root namespace is shared with the legacy 2023
    # artifacts, and HSSM resolves those names at `main` with no revision pin,
    # so an unguarded overwrite changes the likelihood under every installed
    # copy of HSSM on the next cache revalidation.
    if root_filename is not None and not overwrite_root:
        existing_root = list_root_files(api, repo_id, revision)
        if root_filename in existing_root:
            raise RootArtifactExistsError(
                f"{repo_id} already publishes {root_filename} at its root, and "
                "every released HSSM version downloads that exact file. "
                "Replacing it changes the likelihood for all existing users. "
                "Re-run with overwrite_root=True (--overwrite-root) if that is "
                "intended, publish to a staging repo first "
                "(--repo-id franklab/HSSM_staging), or pass "
                "--no-publish-root-alias to upload the folder artifacts only."
            )

    if will_generate_model_card:
        write_default_model_card(model_folder, network_type, model_name)
        logger.info(f"Generated a default model_card.yaml in {model_folder}")
        card_path = model_folder / "model_card.yaml"
        if card_path not in files_to_upload:
            files_to_upload.append(card_path)

    from lanfactory.hf.model_card import load_model_card_yaml, write_readme

    config = load_model_card_yaml(model_folder)
    readme_path = write_readme(model_folder, config, model_name)
    if readme_path not in files_to_upload:
        files_to_upload.append(readme_path)

    existing_manifest = (
        fetch_existing_manifest(repo_id, revision, token) if update_manifest else None
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        manifest_path = None
        if update_manifest:
            entry = build_manifest_entry(
                network_type=network_type,
                model_name=model_name,
                # Publishing without a root alias leaves whatever root file is
                # already live in charge; recording null here would make the
                # manifest misreport what HSSM actually loads.
                root_filename=root_filename
                or existing_root_filename(existing_manifest, network_type, model_name),
                folder_path=path_in_repo,
                files=files_to_upload,
            )
            manifest_path = tmp_path / MANIFEST_FILENAME
            with open(manifest_path, "w") as f:
                json.dump(merge_manifest(existing_manifest, entry), f, indent=2)
                f.write("\n")

        placements = plan_upload_placements(
            files_to_upload=files_to_upload,
            path_in_repo=path_in_repo,
            root_filename=root_filename,
            root_source=root_source,
            manifest_path=manifest_path,
        )

        # One atomic commit for the folder, the root alias and the manifest.
        # Separate calls would leave the repo in a half-published state (e.g.
        # a root alias whose folder upload failed) that HSSM would happily
        # download.
        from huggingface_hub import CommitOperationAdd

        operations = [
            CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(source))
            for dest, source in placements
        ]
        # parent_commit makes the write conditional: a concurrent publish that
        # landed after we read the manifest causes a clean failure instead of
        # silently dropping that publisher's manifest entry.
        parent_commit = None
        if update_manifest:
            try:
                parent_commit = api.repo_info(repo_id=repo_id, revision=revision).sha
            except Exception as e:  # noqa: BLE001 - optimistic locking is a bonus
                logger.warning(f"Could not resolve parent commit for {repo_id}: {e}")

        try:
            api.create_commit(
                repo_id=repo_id,
                operations=operations,
                commit_message=commit_message,
                revision=revision,
                token=token,
                parent_commit=parent_commit,
            )
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise

    url = f"https://huggingface.co/{repo_id}/tree/{revision or 'main'}/{path_in_repo}"
    logger.info(f"Upload successful: {url}")
    print("\nUpload successful!")
    print(f"View your model at: {url}")

    return url


def _collect_files(
    folder: Path,
    include_patterns: list[str],
    exclude_patterns: list[str] | None,
) -> list[Path]:
    """Collect files matching include patterns and not matching exclude patterns.

    Parameters
    ----------
    folder : Path
        Folder to search for files.
    include_patterns : list[str]
        Glob patterns for files to include.
    exclude_patterns : list[str] | None
        Glob patterns for files to exclude.

    Returns
    -------
    list[Path]
        List of file paths to upload.
    """
    matched: set[Path] = set()

    # Collect files matching include patterns
    for pattern in include_patterns:
        matched.update(folder.glob(pattern))

    # Remove files matching exclude patterns
    if exclude_patterns:
        for pattern in exclude_patterns:
            excluded = set(folder.glob(pattern))
            matched -= excluded

    # Filter to only regular files (not directories)
    return sorted(f for f in matched if f.is_file())
