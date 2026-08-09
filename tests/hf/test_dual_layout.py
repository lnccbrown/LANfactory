"""Tests for the dual HF layout (feat/hf-dual-layout-upload).

The load-bearing property: whatever else an upload writes, the canonical ONNX
must land at the repository *root* under the exact filename HSSM constructs.
HSSM does `hf_hub_download(repo_id="franklab/HSSM", filename=f"{model}{suffix}.onnx")`,
so a network published only into `{network_type}/{model}/` is invisible to it —
which is what the previous upload path did.
"""

import json
from pathlib import Path

import pytest
import yaml

from lanfactory.hf.upload import (
    MANIFEST_FILENAME,
    ROOT_ONNX_SUFFIX,
    build_manifest_entry,
    canonical_root_filename,
    merge_manifest,
    names_model,
    plan_upload_placements,
    select_canonical_onnx,
    upload_model,
    write_default_model_card,
)


@pytest.fixture
def model_folder(tmp_path):
    """A trained-network folder shaped like the trainers' output."""
    folder = tmp_path / "ddm"
    folder.mkdir()
    (folder / "abc123_lan_ddm__model.onnx").write_bytes(b"onnx-bytes")
    (folder / "abc123_lan_ddm__train_state.jax").write_bytes(b"jax-bytes")
    (folder / "abc123_lan_ddm__network_config.pickle").write_bytes(b"pickle")
    (folder / "abc123_lan_ddm__data_details.pickle").write_bytes(b"pickle")
    (folder / "validation_report.json").write_text('{"gates": {}}')
    (folder / "abc123_lan_ddm__training_history.csv").write_text("epoch,loss\n")
    return folder


def install_fake_hub(monkeypatch, root_files=(), commits=None, repo_sha="abc"):
    """Register a stub huggingface_hub (plus .utils) and capture commits."""
    import sys

    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, revision=None):
            return list(root_files) + ["lan/other/x.onnx"]

        def repo_info(self, repo_id, revision=None):
            return type("Info", (), {"sha": repo_sha})()

        def create_commit(self, **kwargs):
            if commits is not None:
                # Snapshot payload bytes now: real huggingface_hub reads the
                # files during the commit, while ours live in a
                # TemporaryDirectory that is gone once upload_model returns.
                for op in kwargs["operations"]:
                    source = Path(op.path_or_fileobj)
                    op.content = source.read_bytes() if source.exists() else None
                commits.append(kwargs)

    class FakeOperationAdd:
        def __init__(self, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class FakeRepositoryNotFoundError(Exception):
        pass

    utils = type("Utils", (), {"RepositoryNotFoundError": FakeRepositoryNotFoundError})
    hub = type(
        "FakeHub",
        (),
        {
            "HfApi": FakeApi,
            "create_repo": lambda **kwargs: None,
            "CommitOperationAdd": FakeOperationAdd,
            "utils": utils,
        },
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    return hub


class TestCanonicalRootFilename:
    @pytest.mark.parametrize(
        ("network_type", "expected"),
        [
            ("lan", "ddm.onnx"),
            ("cpn", "ddm_cpn.onnx"),
            ("opn", "ddm_opn.onnx"),
            ("gonogo", "ddm_gonogo.onnx"),
        ],
    )
    def test_matches_hssm_lookup(self, network_type, expected):
        assert canonical_root_filename(network_type, "ddm") == expected

    def test_suffix_map_matches_hssm_source(self):
        """Guard against drift from HSSM's missing_data_networks_suffix.

        Skips when HSSM is not importable — this repo does not depend on it.
        """
        hssm_defaults = pytest.importorskip("hssm.defaults")
        hssm_suffixes = {
            key.name.lower(): value
            for key, value in hssm_defaults.missing_data_networks_suffix.items()
        }
        # HSSM's NONE (no missing-data network) is our plain "lan" case.
        expected = {
            "lan": hssm_suffixes["none"],
            "cpn": hssm_suffixes["cpn"],
            "opn": hssm_suffixes["opn"],
            "gonogo": hssm_suffixes["gonogo"],
        }
        assert ROOT_ONNX_SUFFIX == expected

    def test_unknown_network_type_rejected(self):
        with pytest.raises(ValueError, match="No root filename convention"):
            canonical_root_filename("nope", "ddm")


class TestSelectCanonicalOnnx:
    def test_single_onnx_selected(self, tmp_path):
        onnx = tmp_path / "a_lan_ddm__model.onnx"
        onnx.write_bytes(b"x")
        other = tmp_path / "a_lan_ddm__train_state.jax"
        other.write_bytes(b"x")
        assert select_canonical_onnx([onnx, other], "ddm") == onnx

    def test_no_onnx_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .onnx file"):
            select_canonical_onnx([tmp_path / "x.jax"], "ddm")

    def test_single_onnx_of_a_different_model_is_refused(self, tmp_path):
        """A wrong --model-name must not publish another model's weights.

        The root filename is what released HSSM downloads, so accepting an
        angle network as `ddm.onnx` would silently swap a production
        likelihood. The trainers always embed the model in the filename, so a
        mismatch is a real signal.
        """
        onnx = tmp_path / "run7_lan_angle__model.onnx"
        onnx.write_bytes(b"ANGLE")
        with pytest.raises(ValueError, match="No ONNX artifact in this folder"):
            select_canonical_onnx([onnx], "ddm")

    def test_ambiguous_onnx_refuses_to_guess(self, tmp_path):
        # Two runs of the same model in one folder: both legitimately name
        # ddm, so there is no basis to pick one. Publishing either under the
        # canonical name would be a silent coin flip; fail instead.
        a = tmp_path / "run1_lan_ddm__model.onnx"
        b = tmp_path / "run2_lan_ddm__model.onnx"
        a.write_bytes(b"x")
        b.write_bytes(b"x")
        with pytest.raises(ValueError, match="Cannot determine which ONNX"):
            select_canonical_onnx([a, b], "ddm")

    def test_model_name_disambiguates(self, tmp_path):
        mine = tmp_path / "run_lan_ddm__model.onnx"
        other = tmp_path / "run_lan_angle__model.onnx"
        mine.write_bytes(b"x")
        other.write_bytes(b"x")
        assert select_canonical_onnx([mine, other], "ddm") == mine


class TestPlacements:
    def test_root_alias_duplicates_the_folder_artifact(self, tmp_path):
        onnx = tmp_path / "abc_lan_ddm__model.onnx"
        onnx.write_bytes(b"x")
        placements = plan_upload_placements(
            files_to_upload=[onnx],
            path_in_repo="lan/ddm",
            root_filename="ddm.onnx",
            root_source=onnx,
            manifest_path=None,
        )
        destinations = [dest for dest, _ in placements]
        assert "lan/ddm/abc_lan_ddm__model.onnx" in destinations
        assert "ddm.onnx" in destinations
        # same local file published twice, under both names
        sources = {dest: src for dest, src in placements}
        assert sources["ddm.onnx"] == sources["lan/ddm/abc_lan_ddm__model.onnx"]

    def test_manifest_goes_to_root(self, tmp_path):
        onnx = tmp_path / "m.onnx"
        onnx.write_bytes(b"x")
        manifest = tmp_path / MANIFEST_FILENAME
        manifest.write_text("{}")
        placements = plan_upload_placements(
            files_to_upload=[onnx],
            path_in_repo="lan/ddm",
            root_filename=None,
            root_source=None,
            manifest_path=manifest,
        )
        assert (MANIFEST_FILENAME, manifest) in placements

    def test_root_filename_without_source_is_a_bug(self, tmp_path):
        with pytest.raises(ValueError, match="without root_source"):
            plan_upload_placements([], "lan/ddm", "ddm.onnx", None, None)


class TestManifestMerge:
    def entry(self, model="ddm", network_type="lan"):
        return build_manifest_entry(
            network_type=network_type,
            model_name=model,
            root_filename=canonical_root_filename(network_type, model),
            folder_path=f"{network_type}/{model}",
            files=[Path("a.onnx")],
        )

    def test_creates_manifest_from_nothing(self):
        merged = merge_manifest(None, self.entry())
        assert merged["schema_version"] == 1
        assert [n["model"] for n in merged["networks"]] == ["ddm"]

    def test_republish_replaces_rather_than_duplicates(self):
        first = merge_manifest(None, self.entry())
        updated = build_manifest_entry(
            network_type="lan",
            model_name="ddm",
            root_filename="ddm.onnx",
            folder_path="lan/ddm",
            files=[Path("a.onnx"), Path("b.pickle")],
        )
        merged = merge_manifest(first, updated)
        assert len(merged["networks"]) == 1
        assert merged["networks"][0]["files"] == ["a.onnx", "b.pickle"]

    def test_other_networks_are_preserved(self):
        first = merge_manifest(None, self.entry(model="angle"))
        merged = merge_manifest(first, self.entry(model="ddm"))
        assert [n["model"] for n in merged["networks"]] == ["angle", "ddm"]

    def test_same_model_different_network_type_coexists(self):
        first = merge_manifest(None, self.entry(model="ddm", network_type="lan"))
        merged = merge_manifest(first, self.entry(model="ddm", network_type="cpn"))
        assert {(n["network_type"], n["model"]) for n in merged["networks"]} == {
            ("lan", "ddm"),
            ("cpn", "ddm"),
        }

    def test_unknown_top_level_keys_survive(self):
        existing = {"networks": [], "generated_by": "someone else"}
        merged = merge_manifest(existing, self.entry())
        assert merged["generated_by"] == "someone else"


class TestModelCardGeneration:
    def test_missing_card_is_generated_on_the_real_upload(
        self, model_folder, monkeypatch
    ):
        """Generated on upload, not on dry run (see TestDryRunPurity)."""
        import lanfactory.hf.upload as upload_module

        assert not (model_folder / "model_card.yaml").exists()
        TestProductionSafety().fake_hub(monkeypatch)
        monkeypatch.setattr(
            upload_module, "fetch_existing_manifest", lambda *a, **k: None
        )
        upload_model(
            model_folder=model_folder,
            network_type="lan",
            model_name="ddm",
            repo_id="franklab/HSSM_staging",
        )
        card_path = model_folder / "model_card.yaml"
        assert card_path.exists()
        card = yaml.safe_load(card_path.read_text())
        assert "ddm" in card["title"]
        assert "lan" in card["tags"]

    def test_require_model_card_restores_hard_failure(self, model_folder):
        with pytest.raises(FileNotFoundError, match="require_model_card=True"):
            upload_model(
                model_folder=model_folder,
                network_type="lan",
                model_name="ddm",
                dry_run=True,
                require_model_card=True,
            )

    def test_existing_card_is_not_overwritten(self, model_folder):
        card = model_folder / "model_card.yaml"
        card.write_text(yaml.safe_dump({"title": "hand written"}))
        upload_model(
            model_folder=model_folder,
            network_type="lan",
            model_name="ddm",
            dry_run=True,
        )
        assert yaml.safe_load(card.read_text())["title"] == "hand written"

    def test_generated_card_is_loadable(self, model_folder):
        from lanfactory.hf.model_card import load_model_card_yaml

        write_default_model_card(model_folder, "cpn", "angle")
        config = load_model_card_yaml(model_folder)
        assert "angle" in config.title


class TestDryRunReporting:
    def test_dry_run_names_the_root_alias_and_manifest(self, model_folder, capsys):
        upload_model(
            model_folder=model_folder,
            network_type="cpn",
            model_name="ddm",
            dry_run=True,
        )
        out = capsys.readouterr().out
        assert "ddm_cpn.onnx (root alias" in out
        assert MANIFEST_FILENAME in out

    def test_opting_out_reports_no_alias(self, model_folder, capsys):
        upload_model(
            model_folder=model_folder,
            network_type="lan",
            model_name="ddm",
            dry_run=True,
            publish_root_alias=False,
            update_manifest=False,
        )
        out = capsys.readouterr().out
        assert "root alias" not in out
        assert MANIFEST_FILENAME not in out

    def test_new_artifacts_are_included(self, model_folder, capsys):
        # data_details carries param_bounds; validation_report carries the gate
        # results — both are needed by the registry and were previously dropped.
        upload_model(
            model_folder=model_folder,
            network_type="lan",
            model_name="ddm",
            dry_run=True,
        )
        out = capsys.readouterr().out
        assert "abc123_lan_ddm__data_details.pickle" in out
        assert "validation_report.json" in out


class TestUploadCommit:
    """The upload itself, with huggingface_hub stubbed out."""

    def test_single_atomic_commit_contains_folder_root_and_manifest(
        self, model_folder, monkeypatch
    ):
        import lanfactory.hf.upload as upload_module

        commits = []
        install_fake_hub(monkeypatch, commits=commits)
        monkeypatch.setattr(
            upload_module, "fetch_existing_manifest", lambda *a, **k: None
        )

        upload_model(
            model_folder=model_folder,
            network_type="lan",
            model_name="ddm",
            repo_id="franklab/HSSM_staging",
        )

        assert len(commits) == 1, "upload must be one atomic commit"
        destinations = {op.path_in_repo for op in commits[0]["operations"]}
        assert "ddm.onnx" in destinations, "root alias missing — HSSM cannot load it"
        assert MANIFEST_FILENAME in destinations
        assert "lan/ddm/abc123_lan_ddm__model.onnx" in destinations
        assert "lan/ddm/abc123_lan_ddm__data_details.pickle" in destinations

    def test_manifest_content_written_to_the_commit(self, model_folder, monkeypatch):
        import lanfactory.hf.upload as upload_module

        commits = []
        install_fake_hub(monkeypatch, commits=commits)
        # an existing manifest with another network already published
        monkeypatch.setattr(
            upload_module,
            "fetch_existing_manifest",
            lambda *a, **k: {
                "schema_version": 1,
                "networks": [{"model": "angle", "network_type": "lan"}],
            },
        )

        upload_model(
            model_folder=model_folder,
            network_type="lan",
            model_name="ddm",
            repo_id="franklab/HSSM_staging",
        )

        manifest_op = next(
            op
            for op in commits[0]["operations"]
            if op.path_in_repo == MANIFEST_FILENAME
        )
        manifest = json.loads(manifest_op.content)
        models = {n["model"] for n in manifest["networks"]}
        assert models == {"angle", "ddm"}, "existing entries must survive"
        ddm = next(n for n in manifest["networks"] if n["model"] == "ddm")
        assert ddm["onnx_root"] == "ddm.onnx"
        assert ddm["folder"] == "lan/ddm"


class TestProductionSafety:
    """Guards on writes that reach every installed copy of HSSM.

    Root filenames are resolved by HSSM at `main` with no revision pin, so
    replacing one changes the likelihood for all existing users on their next
    cache revalidation.
    """

    def fake_hub(self, monkeypatch, root_files=(), commits=None, repo_sha="abc"):
        return install_fake_hub(monkeypatch, root_files, commits, repo_sha)

    def test_existing_root_network_is_not_silently_replaced(
        self, model_folder, monkeypatch
    ):
        import lanfactory.hf.upload as upload_module

        self.fake_hub(monkeypatch, root_files=["ddm.onnx"])
        monkeypatch.setattr(
            upload_module, "fetch_existing_manifest", lambda *a, **k: None
        )
        with pytest.raises(
            upload_module.RootArtifactExistsError, match="overwrite-root"
        ):
            upload_model(
                model_folder=model_folder,
                network_type="lan",
                model_name="ddm",
                repo_id="franklab/HSSM",
            )

    def test_overwrite_root_allows_the_replacement(self, model_folder, monkeypatch):
        import lanfactory.hf.upload as upload_module

        commits = []
        self.fake_hub(monkeypatch, root_files=["ddm.onnx"], commits=commits)
        monkeypatch.setattr(
            upload_module, "fetch_existing_manifest", lambda *a, **k: None
        )
        upload_model(
            model_folder=model_folder,
            network_type="lan",
            model_name="ddm",
            repo_id="franklab/HSSM",
            overwrite_root=True,
        )
        assert "ddm.onnx" in {op.path_in_repo for op in commits[0]["operations"]}

    def test_first_publish_of_a_new_model_is_allowed(self, model_folder, monkeypatch):
        import lanfactory.hf.upload as upload_module

        commits = []
        self.fake_hub(monkeypatch, root_files=["angle.onnx"], commits=commits)
        monkeypatch.setattr(
            upload_module, "fetch_existing_manifest", lambda *a, **k: None
        )
        upload_model(
            model_folder=model_folder,
            network_type="lan",
            model_name="ddm",
            repo_id="franklab/HSSM",
        )
        assert "ddm.onnx" in {op.path_in_repo for op in commits[0]["operations"]}

    def test_commit_is_conditional_on_the_parent(self, model_folder, monkeypatch):
        """Concurrent publishes must fail loudly, not drop each other's entries."""
        import lanfactory.hf.upload as upload_module

        commits = []
        self.fake_hub(monkeypatch, commits=commits, repo_sha="parent-sha")
        monkeypatch.setattr(
            upload_module, "fetch_existing_manifest", lambda *a, **k: None
        )
        upload_model(
            model_folder=model_folder,
            network_type="lan",
            model_name="ddm",
            repo_id="franklab/HSSM_staging",
        )
        assert commits[0]["parent_commit"] == "parent-sha"


class TestManifestSafety:
    def test_unreadable_manifest_aborts_instead_of_wiping(
        self, model_folder, monkeypatch
    ):
        """A transient Hub failure must not erase every published network.

        merge_manifest rewrites the file wholesale, so "could not read" must
        never be treated as "there is none".
        """
        import lanfactory.hf.upload as upload_module

        def boom(*a, **k):
            raise upload_module.ManifestUnavailableError("503 Service Unavailable")

        monkeypatch.setattr(upload_module, "fetch_existing_manifest", boom)
        TestProductionSafety().fake_hub(monkeypatch)
        with pytest.raises(upload_module.ManifestUnavailableError):
            upload_model(
                model_folder=model_folder,
                network_type="lan",
                model_name="ddm",
                repo_id="franklab/HSSM",
            )

    def test_absent_manifest_still_starts_a_new_one(self, model_folder, monkeypatch):
        import lanfactory.hf.upload as upload_module

        commits = []
        TestProductionSafety().fake_hub(monkeypatch, commits=commits)
        monkeypatch.setattr(
            upload_module, "fetch_existing_manifest", lambda *a, **k: None
        )
        upload_model(
            model_folder=model_folder,
            network_type="lan",
            model_name="ddm",
            repo_id="franklab/HSSM_staging",
        )
        assert MANIFEST_FILENAME in {op.path_in_repo for op in commits[0]["operations"]}

    def test_no_root_alias_keeps_the_recorded_onnx_root(
        self, model_folder, monkeypatch
    ):
        """Re-publishing without an alias must not claim onnx_root: null.

        The previously published root file stays live and is still what HSSM
        loads; recording null would make the manifest misreport reality.
        """
        import lanfactory.hf.upload as upload_module

        commits = []
        TestProductionSafety().fake_hub(monkeypatch, commits=commits)
        monkeypatch.setattr(
            upload_module,
            "fetch_existing_manifest",
            lambda *a, **k: {
                "schema_version": 1,
                "networks": [
                    {"model": "ddm", "network_type": "lan", "onnx_root": "ddm.onnx"}
                ],
            },
        )
        upload_model(
            model_folder=model_folder,
            network_type="lan",
            model_name="ddm",
            repo_id="franklab/HSSM_staging",
            publish_root_alias=False,
        )
        manifest_op = next(
            op
            for op in commits[0]["operations"]
            if op.path_in_repo == MANIFEST_FILENAME
        )
        manifest = json.loads(manifest_op.content)
        ddm = next(n for n in manifest["networks"] if n["model"] == "ddm")
        assert ddm["onnx_root"] == "ddm.onnx"


class TestDryRunPurity:
    def test_dry_run_does_not_write_into_the_artifact_folder(self, model_folder):
        before = sorted(p.name for p in model_folder.iterdir())
        upload_model(
            model_folder=model_folder,
            network_type="lan",
            model_name="ddm",
            dry_run=True,
        )
        assert sorted(p.name for p in model_folder.iterdir()) == before
        assert not (model_folder / "model_card.yaml").exists()

    def test_dry_run_announces_the_generated_card(self, model_folder, capsys):
        upload_model(
            model_folder=model_folder,
            network_type="lan",
            model_name="ddm",
            dry_run=True,
        )
        assert "model_card.yaml (generated at upload time)" in capsys.readouterr().out

    def test_dry_run_warns_about_root_replacement(self, model_folder, capsys):
        upload_model(
            model_folder=model_folder,
            network_type="lan",
            model_name="ddm",
            dry_run=True,
        )
        out = capsys.readouterr().out
        assert "--overwrite-root" in out

    def test_failed_dry_run_leaves_nothing_behind(self, tmp_path):
        folder = tmp_path / "empty_but_carded"
        folder.mkdir()
        (folder / "notes.csv").write_text("x\n")
        with pytest.raises(FileNotFoundError, match="No .onnx file"):
            upload_model(
                model_folder=folder,
                network_type="lan",
                model_name="ddm",
                dry_run=True,
            )
        assert not (folder / "model_card.yaml").exists()


class TestPlacementDedup:
    def test_duplicate_destinations_are_collapsed(self, tmp_path):
        onnx = tmp_path / "ddm.onnx"
        onnx.write_bytes(b"x")
        readme = tmp_path / "README.md"
        readme.write_text("stale")
        placements = plan_upload_placements(
            files_to_upload=[onnx, readme, readme],
            path_in_repo="lan/ddm",
            root_filename="ddm.onnx",
            root_source=onnx,
            manifest_path=None,
        )
        destinations = [dest for dest, _ in placements]
        assert len(destinations) == len(set(destinations))


class TestReviewFixes:
    """Regressions for the CodeRabbit/Copilot findings on #110."""

    @pytest.mark.parametrize(
        ("filename", "model", "expected"),
        [
            ("abc123_lan_ddm__model.onnx", "ddm", True),  # jax convention
            ("ddm_lan_abc123_model.onnx", "ddm", True),  # torch convention
            ("run_lan_ddm_seq2__model.onnx", "ddm", False),  # prefix, not the model
            ("run_lan_ddm2__model.onnx", "ddm", False),  # prefix, not the model
            ("run_lan_ddm_seq2__model.onnx", "ddm_seq2", True),  # underscored name
            ("model.onnx", "ddm", False),  # no convention -> explicit choice
        ],
    )
    def test_model_match_is_exact_not_substring(self, filename, model, expected):
        # A substring test made "ddm" match ddm_seq2/ddm2 and publish another
        # model's weights as ddm.onnx — the swap this guard exists to stop.
        assert names_model(filename, model) is expected

    @pytest.mark.parametrize("bad", ["foo/bar", "..", "a\\b", ""])
    def test_model_name_must_be_one_path_component(self, model_folder, bad):
        # model_name becomes both the folder and the root filename, so a
        # separator would publish outside the root entirely.
        with pytest.raises(ValueError, match="single path component"):
            upload_model(
                model_folder=model_folder,
                network_type="lan",
                model_name=bad,
                dry_run=True,
            )

    def test_canonical_onnx_outside_the_folder_is_rejected(
        self, model_folder, tmp_path
    ):
        # Otherwise the root gets a network the published folder does not
        # contain, and the manifest records a folder that lacks it.
        stray = tmp_path / "elsewhere.onnx"
        stray.write_bytes(b"x")
        with pytest.raises(ValueError, match="must live in"):
            upload_model(
                model_folder=model_folder,
                network_type="lan",
                model_name="ddm",
                dry_run=True,
                canonical_onnx=stray,
            )

    def test_canonical_onnx_inside_the_folder_is_accepted(self, model_folder):
        chosen = model_folder / "abc123_lan_ddm__model.onnx"
        assert (
            upload_model(
                model_folder=model_folder,
                network_type="lan",
                model_name="ddm",
                dry_run=True,
                canonical_onnx=chosen,
            )
            is None
        )

    def test_gonogo_is_uploadable(self, tmp_path):
        # ROOT_ONNX_SUFFIX claimed gonogo support while VALID_NETWORK_TYPES
        # rejected it, so a trainable, HSSM-loadable type was unpublishable.
        folder = tmp_path / "gonogo"
        folder.mkdir()
        (folder / "abc_gonogo_ddm__model.onnx").write_bytes(b"x")
        assert (
            upload_model(
                model_folder=folder,
                network_type="gonogo",
                model_name="ddm",
                dry_run=True,
            )
            is None
        )

    def test_parent_commit_is_resolved_before_the_manifest_is_read(
        self, model_folder, monkeypatch
    ):
        """Order matters: reading first would let a concurrent publish slip in
        between the read and the parent, and the lock would accept the write."""
        import lanfactory.hf.upload as upload_module

        calls = []
        commits = []
        install_fake_hub(monkeypatch, commits=commits, repo_sha="sha1")

        real_repo_info_marker = "repo_info"

        def spy_fetch(repo_id, revision, token):
            calls.append(("fetch_manifest", revision))
            return None

        monkeypatch.setattr(upload_module, "fetch_existing_manifest", spy_fetch)

        import huggingface_hub

        original = huggingface_hub.HfApi.repo_info

        def spy_repo_info(self, repo_id, revision=None):
            calls.append((real_repo_info_marker, revision))
            return original(self, repo_id, revision)

        monkeypatch.setattr(huggingface_hub.HfApi, "repo_info", spy_repo_info)

        upload_model(
            model_folder=model_folder,
            network_type="lan",
            model_name="ddm",
            repo_id="franklab/HSSM_staging",
        )

        kinds = [c[0] for c in calls]
        assert kinds.index("repo_info") < kinds.index("fetch_manifest")
        # and the manifest is read AT that parent, not at the moving branch
        assert dict(calls)["fetch_manifest"] == "sha1"
        assert commits[0]["parent_commit"] == "sha1"
