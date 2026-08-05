import os
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def test_wan_s2v_runner_registration():
    import xfuser.model_executor.models.runner_models
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    assert "Wan-AI/Wan2.2-S2V-14B" in MODEL_REGISTRY
    assert "Wan2.2-S2V" in MODEL_REGISTRY
    assert "Wan-AI/Wan-Dancer-14B" in MODEL_REGISTRY
    assert "Wan-Dancer-14B" in MODEL_REGISTRY
    assert MODEL_REGISTRY["Wan2.2-S2V"].capabilities.use_fp8_gemms
    assert MODEL_REGISTRY["Wan2.2-S2V"].capabilities.use_fp4_gemms


def test_wan_audio_cli_inputs():
    from xfuser import xFuserArgs
    from xfuser.config import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    args = xFuserArgs.add_runner_args(parser).parse_args(
        [
            "--model",
            "Wan2.2-S2V",
            "--input_images",
            "reference.png",
            "--input_audio",
            "speech.wav",
            "--input_video",
            "pose.mp4",
        ]
    )

    assert args.input_images == ["reference.png"]
    assert args.input_audio == "speech.wav"
    assert args.input_video == "pose.mp4"


def test_external_runtime_uses_provided_engine_config(monkeypatch):
    from xfuser.core.distributed.runtime_state import ExternalRuntimeState, RuntimeState

    engine_config = object()
    monkeypatch.setattr(
        RuntimeState,
        "__init__",
        lambda self, config: setattr(self, "received_config", config),
    )
    runtime = ExternalRuntimeState(config=engine_config)

    assert runtime.received_config is engine_config


def test_wan_s2v_runtime_adapter_adds_attention_head_alias():
    from xfuser.model_executor.models.runner_models.wan_audio import (
        _WanS2VRuntimePipeline,
    )

    transformer = SimpleNamespace(config=SimpleNamespace(num_heads=40))
    engine = SimpleNamespace(noise_model=transformer)

    pipe = _WanS2VRuntimePipeline(engine)

    assert pipe.transformer is transformer
    assert pipe.transformer.config.num_attention_heads == 40


def test_wan_s2v_quantization_policy():
    from xfuser.model_executor.models.runner_models.wan_audio import (
        xFuserWan22S2VModel,
    )

    assert xFuserWan22S2VModel.settings.fp8_gemm_module_list == ["transformer.blocks"]
    assert xFuserWan22S2VModel.settings.fp4_gemm_module_list == ["transformer.blocks"]
    assert xFuserWan22S2VModel.settings.fp8_precision_overrides == (
        "0.",
        "1.",
        "38.",
        "39.",
    )


@pytest.mark.skipif(
    not os.environ.get("WAN22_REPO_PATH"),
    reason="official Wan2.2 checkout is not available",
)
def test_wan_s2v_xdit_attention_matches_reference(monkeypatch):
    from xfuser.model_executor.models.transformers.transformer_wan_s2v import (
        import_official_wan,
    )

    import_official_wan()
    import wan.modules.s2v.model_s2v as model_s2v
    from wan.modules.s2v.model_s2v import WanS2VSelfAttention

    from xfuser.model_executor.models.transformers import transformer_wan_s2v

    def attention(query=None, key=None, value=None, q=None, k=None, v=None, **kwargs):
        query = query if query is not None else q
        key = key if key is not None else k
        value = value if value is not None else v
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        ).transpose(1, 2)

    def usp(query, key, value):
        return F.scaled_dot_product_attention(query, key, value)

    monkeypatch.setattr(model_s2v, "rope_apply", lambda tensor, *args: tensor)
    monkeypatch.setattr(model_s2v, "flash_attention", attention)
    monkeypatch.setattr(transformer_wan_s2v, "USP", usp)

    module = (
        WanS2VSelfAttention(
            dim=32,
            num_heads=2,
            window_size=(-1, -1),
            qk_norm=True,
            eps=1e-6,
        )
        .to(torch.bfloat16)
        .eval()
    )
    hidden_states = torch.randn(1, 8, 32, dtype=torch.bfloat16)
    sequence_lengths = torch.tensor([8])
    grid_sizes = torch.tensor([[2, 2, 2]])
    rotary_frequencies = torch.empty(0)

    with torch.no_grad():
        expected = module(
            hidden_states,
            sequence_lengths,
            grid_sizes,
            rotary_frequencies,
        )
        module.forward = types.MethodType(
            transformer_wan_s2v._xdit_s2v_self_attention,
            module,
        )
        actual = module(
            hidden_states,
            sequence_lengths,
            grid_sizes,
            rotary_frequencies,
        )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("degree", [3, 6, 16])
def test_wan_s2v_rejects_invalid_ulysses_degree(monkeypatch, degree):
    from xfuser.model_executor.models.runner_models.base_model import xFuserModel
    from xfuser.model_executor.models.runner_models.wan_audio import (
        xFuserWan22S2VModel,
    )

    monkeypatch.setattr(xFuserModel, "_validate_config", lambda self, config: None)
    model = object.__new__(xFuserWan22S2VModel)
    config = SimpleNamespace(
        ulysses_degree=degree,
        batch_size=None,
        dataset_path=None,
    )

    with pytest.raises(ValueError, match="must divide 40"):
        model._validate_config(config)


def test_wan_dancer_requires_u8(monkeypatch):
    from xfuser.model_executor.models.runner_models.base_model import xFuserModel
    from xfuser.model_executor.models.runner_models.wan_audio import (
        xFuserWanDancerModel,
    )

    monkeypatch.setattr(xFuserModel, "_validate_config", lambda self, config: None)
    model = object.__new__(xFuserWanDancerModel)
    config = SimpleNamespace(
        task="global",
        ulysses_degree=4,
        batch_size=None,
        dataset_path=None,
    )

    with pytest.raises(ValueError, match="requires --ulysses_degree 8"):
        model._validate_config(config)


@pytest.mark.skipif(
    not os.environ.get("WAN_DANCER_REPO_PATH"),
    reason="official Wan-Dancer checkout is not available",
)
def test_wan_dancer_pipeline_imports(monkeypatch):
    from xfuser.model_executor.models.runner_models.wan_audio import (
        xFuserWanDancerModel,
    )

    repo_path = xFuserWanDancerModel._resolve_repo_path()
    monkeypatch.syspath_prepend(repo_path)

    import transformers
    import transformers.modeling_utils as modeling_utils

    if not hasattr(modeling_utils, "PretrainedConfig"):
        monkeypatch.setattr(
            modeling_utils,
            "PretrainedConfig",
            transformers.PreTrainedConfig,
            raising=False,
        )
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    assert WanVideoPipeline.__name__ == "WanVideoPipeline"
    assert ModelConfig.__name__ == "ModelConfig"


@pytest.mark.skipif(
    not os.environ.get("WAN_DANCER_REPO_PATH"),
    reason="official Wan-Dancer checkout is not available",
)
def test_wan_dancer_xdit_attention_matches_reference(monkeypatch):
    import transformers
    import transformers.modeling_utils as modeling_utils

    from xfuser.model_executor.models.runner_models.wan_audio import (
        xFuserWanDancerModel,
    )
    from xfuser.model_executor.models.transformers import transformer_wan_dancer

    repo_path = xFuserWanDancerModel._resolve_repo_path()
    monkeypatch.syspath_prepend(repo_path)
    if not hasattr(modeling_utils, "PretrainedConfig"):
        monkeypatch.setattr(
            modeling_utils,
            "PretrainedConfig",
            transformers.PreTrainedConfig,
            raising=False,
        )
    from diffsynth.models.wan_video_dit import SelfAttention
    import diffsynth.distributed.xdit_context_parallel as dancer_parallel

    def usp(query, key, value):
        return F.scaled_dot_product_attention(query, key, value)

    monkeypatch.setattr(
        dancer_parallel,
        "rope_apply",
        lambda tensor, *args: tensor,
    )
    monkeypatch.setattr(transformer_wan_dancer, "USP", usp)
    module = SelfAttention(dim=32, num_heads=2).to(torch.bfloat16).eval()
    hidden_states = torch.randn(1, 8, 32, dtype=torch.bfloat16)
    frequencies = torch.empty(0)

    with torch.no_grad():
        query = module.norm_q(module.q(hidden_states)).unflatten(2, (2, 16))
        key = module.norm_k(module.k(hidden_states)).unflatten(2, (2, 16))
        value = module.v(hidden_states).unflatten(2, (2, 16))
        expected = module.o(
            F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
            )
            .transpose(1, 2)
            .flatten(2)
        )
        actual = transformer_wan_dancer._xdit_dancer_self_attention(
            module,
            hidden_states,
            frequencies,
        )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not os.environ.get("WAN_DANCER_REPO_PATH"),
    reason="official Wan-Dancer checkout is not available",
)
def test_wan_dancer_music_feature_extraction(tmp_path):
    import numpy as np

    from xfuser.model_executor.models.runner_models.wan_audio import (
        extract_dancer_music_feature,
        xFuserWanDancerModel,
    )

    music_path = os.path.join(
        xFuserWanDancerModel._resolve_repo_path(),
        "gen_video",
        "music",
        "ChineseClassicDance.WAV",
    )
    output_path = tmp_path / "music.npy"

    extract_dancer_music_feature(music_path, str(output_path))
    features = np.load(output_path)

    assert features.ndim == 2
    assert features.shape[1] == 35
