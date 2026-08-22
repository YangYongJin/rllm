"""Exact hosted-SFT checkpointing and resume, exercised on fakes."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

tinker = pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook")

from rllm.data import Dataset  # noqa: E402
from rllm.trainer.sft import SFTSpec  # noqa: E402
from rllm.trainer.sft.backend import SFTConfigError  # noqa: E402
from rllm.trainer.sft.fireworks_backend import (  # noqa: E402
    FireworksSFTBackend,
    _fireworks_output_model_id,
)
from rllm.trainer.sft.tinker_backend import (  # noqa: E402
    TinkerSFTBackend,
    iter_training_batches_from_step,
    validate_tinker_resume_cursor,
)
from rllm.trainer.sft.tinker_dataset import TinkerSFTDataset  # noqa: E402


class _TokenRenderer:
    def render(self, messages, *, tools=None, add_generation_prompt=False):
        from rllm.renderers.types import RenderedTokens

        del tools, add_generation_prompt
        token = int(messages[-1]["content"])
        return RenderedTokens(
            token_ids=[0, token, 0],
            message_indices=[-1, 1, -1],
        )


def test_fireworks_output_model_id_trims_truncation_separator():
    model_id = _fireworks_output_model_id(
        "rllm-deepswe-sft",
        "deepswe-n3super-b2-lr2.0e-4-e10-constant-wu10-dbac8cfdc-r987e71f8",
    )

    assert model_id == "rllm-deepswe-sft-deepswe-n3super-b2-lr2-0e-4-e10-constant-wu10"
    assert len(model_id) <= 63
    assert not model_id.endswith("-")


def _source(size: int = 7) -> Dataset:
    return Dataset(
        data=[
            {
                "messages": [
                    {"role": "user", "content": "q", "trainable": False},
                    {"role": "assistant", "content": str(index), "trainable": True},
                ]
            }
            for index in range(size)
        ],
        name="resume-order",
        split="train",
    )


def _ordered_batches(dataset, *, total_epochs: int, start_step: int = 0):
    batches = []
    current_epoch = None
    for _, epoch, batch in iter_training_batches_from_step(
        n_batches=len(dataset),
        total_epochs=total_epochs,
        start_step=start_step,
    ):
        if epoch != current_epoch:
            dataset.set_epoch(seed=epoch)
            current_epoch = epoch
        datums = dataset.get_batch(batch)
        batches.append(tuple(datum.model_input.to_ints()[1] for datum in datums))
    return batches


def test_resumed_epoch_order_matches_uninterrupted_next_batches():
    source = _source()
    uninterrupted = TinkerSFTDataset(source, _TokenRenderer(), batch_size=3)
    full_order = _ordered_batches(uninterrupted, total_epochs=3)

    resumed = TinkerSFTDataset(source, _TokenRenderer(), batch_size=3)
    assert _ordered_batches(resumed, total_epochs=3, start_step=4) == full_order[4:]


def test_raw_row_cursor_round_trips_partial_batches_and_epochs():
    # drop_last=False keeps the trailing partial batch; the default (True,
    # since #870) trains full batches only and is covered separately below.
    dataset = TinkerSFTDataset(_source(5), _TokenRenderer(), batch_size=2, drop_last=False)
    cursors = [0, 2, 4, 5, 7, 9, 10]
    assert [dataset.data_cursor_for_step(step) for step in range(7)] == cursors
    assert [dataset.step_for_data_cursor(cursor) for cursor in cursors] == list(range(7))

    for cursor in (-1, 1, 3, 6):
        with pytest.raises(SFTConfigError, match="cursor.*non-negative|cursor.*exact"):
            dataset.step_for_data_cursor(cursor)


def test_raw_row_cursor_round_trips_dropped_tail_batches():
    # Default batching (drop_last=True) trains floor(rows/batch) batches per
    # epoch while an epoch still consumes every source row.
    dataset = TinkerSFTDataset(_source(5), _TokenRenderer(), batch_size=2)
    cursors = [0, 2, 5, 7, 10, 12]
    assert [dataset.data_cursor_for_step(step) for step in range(6)] == cursors
    assert [dataset.step_for_data_cursor(cursor) for cursor in cursors] == list(range(6))


@pytest.mark.parametrize(
    "cursor",
    [
        {"epoch": 0, "batch": 2, "step": 1},
        {"epoch": 0, "batch": 3, "step": 3},
        {"epoch": 0, "batch": 1},
    ],
)
def test_tinker_cursor_rejects_off_by_one_or_incomplete_state(cursor):
    with pytest.raises(SFTConfigError, match="inconsistent|loop_state.step"):
        validate_tinker_resume_cursor(cursor, n_batches=3, total_steps=3)


@pytest.mark.parametrize(
    ("backend_cls", "default_root"),
    [
        (TinkerSFTBackend, "/tmp/rllm-tinker-sft-checkpoints"),
        (FireworksSFTBackend, "/tmp/rllm-fireworks-sft-checkpoints"),
    ],
)
def test_default_paths_are_isolated_and_explicit_path_is_resume_identity(tmp_path, backend_cls, default_root):
    first = backend_cls(SFTSpec(train_dataset=_source(), experiment="same"))
    second = backend_cls(SFTSpec(train_dataset=_source(), experiment="same"))
    assert first.checkpoint_dir != second.checkpoint_dir
    assert first.checkpoint_dir.startswith(f"{default_root}/same/")

    explicit_path = str(tmp_path / "resume")
    explicit = backend_cls(SFTSpec(train_dataset=_source(), output_dir=explicit_path))
    assert explicit.checkpoint_dir == explicit_path


def test_fireworks_reattach_requires_explicit_cursor_directory():
    backend = FireworksSFTBackend(
        SFTSpec(
            train_dataset=_source(),
            overrides={"fireworks_infra": {"trainers": {"policy": {"job_id": "job-1"}}}},
        )
    )
    with pytest.raises(SFTConfigError, match=r"job_id requires.*explicit --output"):
        backend.build_config()


@pytest.mark.parametrize(
    ("case", "job_id"),
    [("fresh-run-failure", None), ("reattached-run", "job-1")],
)
def test_fireworks_provision_never_requests_trainer_deletion(monkeypatch, tmp_path, case, job_id):
    provision_module = pytest.importorskip("training.provision")

    overrides = {}
    if job_id is not None:
        overrides = {"fireworks_infra": {"trainers": {"policy": {"job_id": job_id}}}}
    backend = FireworksSFTBackend(
        SFTSpec(
            train_dataset=_source(),
            output_dir=str(tmp_path / case),
            overrides=overrides,
        )
    )
    config = backend.build_config()
    provision_config = object()
    returned_infra = object()
    call = {}

    monkeypatch.setattr(
        provision_module,
        "load_yaml_provision",
        lambda **_kwargs: ("sft", provision_config),
    )

    def init_fireworks_infra(mode, config_arg, **kwargs):
        call.update(mode=mode, config=config_arg, **kwargs)
        return returned_infra

    monkeypatch.setattr(provision_module, "init_fireworks_infra", init_fireworks_infra)

    assert backend._provision(config, "fake-key", "https://example.invalid") is returned_infra
    assert call["mode"] == "sft"
    assert call["config"] is provision_config
    assert call["cleanup_on_close"] is False
    assert call["cleanup_existing"] is False


def _one_datum():
    import torch
    from tinker_cookbook.supervised.common import datum_from_model_input_weights

    return datum_from_model_input_weights(
        tinker.ModelInput.from_ints([1, 2, 3]),
        torch.tensor([0.0, 1.0, 0.0]),
        max_length=None,
        reduction="none",
    )


class _HostedDataset:
    def __init__(self, events, batches=3, *, batch_size=1, row_count=None):
        self.events = events
        self.batches = batches
        self.batch_size = batch_size
        self.dataset = [object()] * (row_count if row_count is not None else batches * batch_size)
        self.datum = _one_datum()
        self.training_batches = []
        self.preflighting = False

    def __len__(self):
        return self.batches

    def get_batch(self, index):
        if not self.preflighting:
            self.training_batches.append(index)
            self.events.append(f"batch-{index}")
        return [self.datum]

    def set_epoch(self, seed):
        self.events.append(f"epoch-{seed}")

    def preflight(self, label="train", planned_batches=None):
        self.preflighting = True
        self.events.append(f"preflight-{label}")
        try:
            if planned_batches is not None:
                for _epoch, batch in planned_batches:
                    self.get_batch(batch)
            else:
                for batch in range(self.batches):
                    self.get_batch(batch)
        finally:
            self.preflighting = False

    def data_cursor_for_step(self, completed_steps):
        completed_epochs, batches_in_epoch = divmod(completed_steps, self.batches)
        rows_in_epoch = min(batches_in_epoch * self.batch_size, len(self.dataset))
        return completed_epochs * len(self.dataset) + rows_in_epoch

    def step_for_data_cursor(self, data_consumed):
        completed_epochs, rows_in_epoch = divmod(data_consumed, len(self.dataset))
        batches_in_epoch = (rows_in_epoch + self.batch_size - 1) // self.batch_size
        step = completed_epochs * self.batches + batches_in_epoch
        if self.data_cursor_for_step(step) != data_consumed:
            raise SFTConfigError("checkpoint cursor is not exact")
        return step


class _AsyncFuture:
    def __init__(self, value, events, event):
        self.value = value
        self.events = events
        self.event = event

    async def result_async(self):
        self.events.append(self.event)
        return self.value


class _TinkerTrainingClient:
    def __init__(self, events, datum):
        self.events = events
        self.datum = datum
        self.submitted = 0

    def _output(self):
        weights = self.datum.loss_fn_inputs["weights"]
        logprobs = tinker.TensorData(
            data=[-1.0] * len(weights.data),
            dtype=weights.dtype,
            shape=list(weights.shape),
        )
        return SimpleNamespace(loss_fn_outputs=[{"logprobs": logprobs}])

    async def forward_backward_async(self, data, loss_fn):
        del data, loss_fn
        step = self.submitted
        self.submitted += 1
        self.events.append(f"submit-{step}")
        return _AsyncFuture(self._output(), self.events, f"finish-fb-{step}")

    async def optim_step_async(self, adam):
        del adam
        step = self.submitted - 1
        return _AsyncFuture(SimpleNamespace(metrics={}), self.events, f"finish-opt-{step}")


class _Tracking:
    def __init__(self, **kwargs):
        del kwargs

    def log(self, data, step):
        del data, step

    def finish(self):
        pass


def test_tinker_checkpoint_drains_pipeline_and_records_next_unseen_cursor(monkeypatch, tmp_path):
    from tinker_cookbook import checkpoint_utils, display

    import rllm.trainer.sft.tinker_backend as module
    import rllm.utils.tracking as tracking_module

    events = []
    train = _HostedDataset(events)
    training_client = _TinkerTrainingClient(events, train.datum)
    saves = []

    class _Service:
        def __init__(self, **kwargs):
            del kwargs

        async def create_lora_training_client_async(self, **kwargs):
            del kwargs
            return training_client

    async def save_checkpoint(**kwargs):
        events.append(f"save-{kwargs['name']}")
        saves.append(kwargs)

    monkeypatch.setattr(module, "build_sft_data", lambda *_: (object(), train, None))
    monkeypatch.setattr(tinker, "ServiceClient", _Service)
    monkeypatch.setattr(checkpoint_utils, "get_last_checkpoint", lambda *_: None)
    monkeypatch.setattr(checkpoint_utils, "save_checkpoint_async", save_checkpoint)
    monkeypatch.setattr(display, "colorize_example", lambda *_: "example")
    monkeypatch.setattr(tracking_module, "Tracking", _Tracking)

    backend = TinkerSFTBackend(
        SFTSpec(
            train_dataset=_source(),
            output_dir=str(tmp_path),
            overrides={"trainer": {"max_steps": 3, "save_freq": 2, "test_freq": -1}},
        )
    )
    config = backend.build_config()
    config.data.resolved_renderer_name = "qwen3_5"
    asyncio.run(backend._fit_async())

    assert events.index("finish-opt-1") < events.index("save-000002") < events.index("submit-2")
    assert events.index("finish-opt-2") < events.index("save-final")
    assert saves[0]["loop_state"]["epoch"] == 0
    assert saves[0]["loop_state"]["batch"] == 2
    assert saves[0]["loop_state"]["step"] == 2
    assert saves[1]["loop_state"]["step"] == 3
    assert saves[1]["loop_state"]["final"] is True
    assert all(save["kind"] == "both" for save in saves)


def test_tinker_resume_restores_optimizer_and_starts_at_next_unseen_batch(monkeypatch, tmp_path):
    from tinker_cookbook import checkpoint_utils, display

    import rllm.trainer.sft.tinker_backend as module
    import rllm.utils.tracking as tracking_module

    events = []
    train = _HostedDataset(events)
    training_client = _TinkerTrainingClient(events, train.datum)

    class _Checkpoint:
        state_path = "tinker://run/weights/step"

        def get(self, key, default=None):
            return {
                "epoch": 0,
                "batch": 2,
                "step": 2,
            }.get(key, default)

    class _Rest:
        async def get_weights_info_by_tinker_path(self, path):
            assert path == "tinker://run/weights/step"
            return SimpleNamespace(
                base_model="Qwen/Qwen3.5-4B",
                lora_rank=32,
                train_unembed=True,
                train_attn=True,
                train_mlp=True,
            )

        async def get_training_run_by_tinker_path_async(self, path):
            assert path == "tinker://run/weights/step"
            return SimpleNamespace(user_metadata={checkpoint_utils.RENDERER_NAME_METADATA_KEY: "qwen3_5"})

    class _Service:
        def __init__(self, **kwargs):
            del kwargs

        def create_rest_client(self):
            return _Rest()

        async def create_training_client_from_state_with_optimizer_async(self, path, **kwargs):
            del kwargs
            events.append("resume-with-optimizer")
            assert path == "tinker://run/weights/step"
            return training_client

        async def create_training_client_from_state_async(self, *args, **kwargs):
            del args, kwargs
            pytest.fail("weights-only resume must never be used")

    async def save_checkpoint(**kwargs):
        events.append(f"save-{kwargs['name']}")

    monkeypatch.setattr(module, "build_sft_data", lambda *_: (object(), train, None))
    monkeypatch.setattr(tinker, "ServiceClient", _Service)
    monkeypatch.setattr(checkpoint_utils, "get_last_checkpoint", lambda *_: _Checkpoint())
    monkeypatch.setattr(checkpoint_utils, "save_checkpoint_async", save_checkpoint)
    monkeypatch.setattr(display, "colorize_example", lambda *_: "example")
    monkeypatch.setattr(tracking_module, "Tracking", _Tracking)

    backend = TinkerSFTBackend(
        SFTSpec(
            train_dataset=_source(),
            output_dir=str(tmp_path),
            overrides={"trainer": {"max_steps": 3, "save_freq": -1, "test_freq": -1}},
        )
    )
    config = backend.build_config()
    config.data.resolved_renderer_name = "qwen3_5"
    asyncio.run(backend._fit_async())

    assert "resume-with-optimizer" in events
    assert train.training_batches == [2]


class _SyncFuture:
    def __init__(self, value, events, event):
        self.value = value
        self.events = events
        self.event = event

    def result(self, timeout=None):
        del timeout
        self.events.append(self.event)
        return self.value


class _FireworksClient:
    def __init__(self, events):
        self.events = events
        self.submitted = 0

    def submit_forward_backward(self, data, loss_fn):
        del data, loss_fn
        step = self.submitted
        self.submitted += 1
        self.events.append(f"submit-{step}")
        result = SimpleNamespace(metrics={"response_tokens": 1, "loss:sum": 1.0})
        return _SyncFuture(result, self.events, f"finish-fb-{step}")

    def submit_optim_step(self, adam):
        del adam
        step = self.submitted - 1
        return _SyncFuture(None, self.events, f"finish-opt-{step}")


class _FireworksCheckpoints:
    def __init__(self, events, *, data_consumed=None, provider_step=0):
        self.events = events
        self.data_consumed = data_consumed
        self.provider_step = provider_step
        self.saves = []
        self.log_path = None

    def resume(self):
        if self.data_consumed is None:
            return None
        return SimpleNamespace(step=self.provider_step, data_consumed=self.data_consumed)

    def save(self, name, **kwargs):
        self.events.append(f"save-{name}")
        self.saves.append((name, kwargs))

    def promote_latest(self, output_model_id, base_model):
        del base_model
        return {"name": output_model_id}


def _run_fireworks(monkeypatch, tmp_path, *, data_consumed=None, provider_step=0):
    checkpoint_module = pytest.importorskip("training.utils.checkpoints")

    import rllm.trainer.sft.fireworks_backend as module
    import rllm.utils.tracking as tracking_module

    events = []
    # Five rows at batch size three yields cursors 0 -> 3 -> 5 -> 8 across
    # the partial final batch and the next epoch.
    train = _HostedDataset(events, batches=2, batch_size=3, row_count=5)
    client = _FireworksClient(events)
    checkpoints = _FireworksCheckpoints(
        events,
        data_consumed=data_consumed,
        provider_step=provider_step,
    )
    infra = SimpleNamespace(
        policy=client,
        service=object(),
        policy_job_id="fake-job",
        close=lambda: events.append("infra-close"),
    )

    overrides = {"trainer": {"max_steps": 3, "save_freq": 2, "test_freq": -1}}
    if data_consumed is not None:
        overrides["fireworks_infra"] = {"trainers": {"policy": {"job_id": "fake-job"}}}
    backend = FireworksSFTBackend(
        SFTSpec(
            train_dataset=_source(),
            output_dir=str(tmp_path),
            epochs=2,
            overrides=overrides,
        )
    )
    config = backend.build_config()
    config.data.resolved_renderer_name = "qwen3_5"
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake")
    monkeypatch.setattr(module, "build_sft_data", lambda *_: (None, train, None))
    monkeypatch.setattr(backend, "_provision", lambda *_: infra)
    monkeypatch.setattr(tracking_module, "Tracking", _Tracking)

    def checkpoint_factory(*_args, **kwargs):
        checkpoints.log_path = kwargs["log_path"]
        return checkpoints

    monkeypatch.setattr(checkpoint_module, "TrainingCheckpoints", checkpoint_factory)
    backend.fit()
    return events, train, checkpoints


def test_fireworks_checkpoint_drains_pipeline_and_persists_raw_cursor(monkeypatch, tmp_path):
    events, _train, checkpoints = _run_fireworks(monkeypatch, tmp_path)
    assert checkpoints.log_path == str(tmp_path)
    assert events.index("finish-opt-1") < events.index("save-step-2") < events.index("submit-2")
    assert events.index("finish-opt-2") < events.index("save-step-3")
    assert checkpoints.saves == [
        (
            "step-2",
            {"resumable": True, "promotable": False, "data_consumed": 5},
        ),
        (
            "step-3",
            {"resumable": True, "promotable": True, "data_consumed": 8},
        ),
    ]


def test_fireworks_resume_ignores_provider_renamed_step_and_uses_raw_cursor(monkeypatch, tmp_path):
    events, train, checkpoints = _run_fireworks(
        monkeypatch,
        tmp_path,
        data_consumed=5,
        provider_step=999,
    )
    assert train.training_batches == [0]
    assert "epoch-1" in events
    assert "batch-1" not in events
    assert checkpoints.saves[-1][1]["data_consumed"] == 8


def test_fireworks_relaunch_without_job_id_refuses_silent_retrain(monkeypatch, tmp_path):
    # A relaunch under the same --output provisions a NEW trainer job whose
    # resume() is empty; the local dataloader.json left by the previous run is
    # the evidence that silent from-scratch retraining would lose progress.
    (tmp_path / "dataloader.json").write_text(json.dumps({"step-2": {"data_consumed": 5}}))
    with pytest.raises(SFTConfigError, match="no resumable checkpoint"):
        _run_fireworks(monkeypatch, tmp_path, data_consumed=None)


def test_fireworks_resume_rejects_non_boundary_raw_cursor(monkeypatch, tmp_path):
    with pytest.raises(SFTConfigError, match="cursor is not exact"):
        _run_fireworks(
            monkeypatch,
            tmp_path,
            data_consumed=1,
            provider_step=1,
        )
