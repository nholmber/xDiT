from __future__ import annotations

import copy
import importlib.util
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download

from xfuser.core.distributed import (
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
    initialize_runtime_state,
)
from xfuser.core.utils.runner_utils import (
    log,
    quantize_linear_layers_to_fp4,
    quantize_linear_layers_to_fp8,
)
from xfuser.model_executor.models.runner_models.base_model import (
    DefaultInputValues,
    DiffusionOutput,
    ModelCapabilities,
    ModelSettings,
    register_model,
    xFuserModel,
)


class WanAudioDiffusionOutput(DiffusionOutput):
    def __init__(self, videos, pipe_args, audio_path: str):
        super().__init__(videos=videos, pipe_args=pipe_args)
        self.audio_path = audio_path


class _WanS2VRuntimePipeline:
    def __init__(self, engine):
        self.engine = engine
        self.transformer = engine.noise_model
        config = self.transformer.config
        if not hasattr(config, "num_attention_heads"):
            config.num_attention_heads = config.num_heads

    def to(self, device):
        self.transformer.to(device)
        return self


def _load_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_shared_snapshot(
    model_name: str,
    allow_patterns: list[str] | None = None,
) -> str:
    world_group = get_world_group()
    snapshot_path = [None]
    if world_group.rank == world_group.first_rank:
        snapshot_path[0] = snapshot_download(
            model_name,
            allow_patterns=allow_patterns,
        )
    torch.distributed.broadcast_object_list(
        snapshot_path,
        src=world_group.first_rank,
        group=world_group.cpu_group,
    )
    return snapshot_path[0]


def _resolve_wan_common_snapshot() -> str:
    return _resolve_shared_snapshot(
        "Wan-AI/Wan2.2-S2V-14B",
        allow_patterns=[
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth",
            "google/umt5-xxl/**",
        ],
    )


def extract_dancer_music_feature(
    music_path: str,
    output_path: str,
    fps: int = 30,
) -> None:
    import librosa
    from librosa.feature.rhythm import tempo
    from scipy.signal import find_peaks

    hop_length = 512
    data, _ = librosa.load(music_path, sr=fps * hop_length)
    feature_sample_rate = 22050
    envelope = librosa.onset.onset_strength(
        y=data,
        sr=feature_sample_rate,
    )
    mfcc = librosa.feature.mfcc(
        y=data,
        sr=feature_sample_rate,
        n_mfcc=20,
    ).T
    chroma = librosa.feature.chroma_cens(
        y=data,
        sr=feature_sample_rate,
        hop_length=hop_length,
        n_chroma=12,
    ).T
    prominence = max(float(envelope.std()) * 0.25, 1e-6)
    peak_indices, _ = find_peaks(envelope, prominence=prominence)
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_indices] = 1.0
    original_audio, _ = librosa.load(music_path)
    start_bpm = float(tempo(y=original_audio)[0])
    _, beat_indices = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=feature_sample_rate,
        hop_length=hop_length,
        start_bpm=start_bpm,
        tightness=100,
    )
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_indices] = 1.0
    audio_feature = np.concatenate(
        [
            envelope[:, None],
            mfcc,
            chroma,
            peak_onehot[:, None],
            beat_onehot[:, None],
        ],
        axis=-1,
    )
    np.save(output_path, audio_feature)


class _DancerAudioClip:
    def __init__(self, path: str):
        import soundfile as sf

        self.path = path
        self.duration = sf.info(path).duration

    def close(self):
        return


class _DancerVideoClip:
    def __init__(self, path: str):
        self.path = path
        self.audio = None
        self.crop = None

    def write_videofile(self, output_path: str, **kwargs):
        if self.audio is None and self.crop is None:
            shutil.copyfile(self.path, output_path)
            return
        command = ["ffmpeg", "-y", "-i", self.path]
        if self.audio is not None:
            command.extend(["-i", self.audio.path])
        if self.crop is not None:
            x, y, width, height = self.crop
            command.extend(
                [
                    "-vf",
                    f"crop={width}:{height}:{x}:{y}",
                    "-c:v",
                    "libopenh264",
                ]
            )
        else:
            command.extend(["-c:v", "copy"])
        if self.audio is not None:
            command.extend(["-c:a", "aac", "-shortest"])
        command.append(output_path)
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def close(self):
        return


class _DancerCrop:
    def __init__(self, x1, y1, x2, y2):
        self.crop = (
            int(x1),
            int(y1),
            int(x2 - x1),
            int(y2 - y1),
        )

    def apply(self, video):
        video.crop = self.crop
        return video


@register_model("Wan-AI/Wan2.2-S2V-14B")
@register_model("Wan2.2-S2V")
class xFuserWan22S2VModel(xFuserModel):
    default_input_values = DefaultInputValues(
        height=480,
        width=832,
        num_frames=80,
        num_inference_steps=40,
        negative_prompt=(
            "画面模糊，最差质量，细节模糊不清，情绪激动剧烈，手快速抖动，"
            "字幕，丑陋的，残缺的，多余的手指，畸形的，静止不动的画面"
        ),
        guidance_scale=4.5,
        flow_shift=3.0,
    )
    settings = ModelSettings(
        model_name="Wan-AI/Wan2.2-S2V-14B",
        output_name="wan2.2_s2v",
        model_output_type="video",
        fps=16,
        fp8_gemm_module_list=["transformer.blocks"],
        fp4_gemm_module_list=["transformer.blocks"],
        fp8_precision_overrides=("0.", "1.", "38.", "39."),
        fp8_precision_override_suffixes=(".ffn.0", ".ffn.2"),
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=False,
        data_parallel_degree=False,
        use_cfg_parallel=False,
        fully_shard_degree=False,
        use_fp8_gemms=True,
        use_fp4_gemms=True,
        use_parallel_vae=False,
        enable_slicing=False,
        enable_tiling=False,
    )

    def _validate_config(self, config) -> None:
        super()._validate_config(config)
        if config.ulysses_degree not in (1, 2, 4, 5, 8, 10, 20, 40):
            raise ValueError(
                "Wan2.2-S2V has 40 attention heads; --ulysses_degree must divide 40."
            )
        if config.batch_size is not None or config.dataset_path is not None:
            raise ValueError("Wan2.2-S2V currently supports one request at a time.")

    def _resolve_checkpoint(self) -> str:
        local_path = os.environ.get("WAN_S2V_MODEL_PATH")
        if local_path:
            return local_path
        return _resolve_shared_snapshot(
            self.settings.model_name,
            allow_patterns=[
                "config.json",
                "configuration.json",
                "diffusion_pytorch_model*.safetensors*",
                "models_t5_umt5-xxl-enc-bf16.pth",
                "Wan2.1_VAE.pth",
                "google/umt5-xxl/**",
                "wav2vec2-large-xlsr-53-english/config.json",
                "wav2vec2-large-xlsr-53-english/preprocessor_config.json",
                "wav2vec2-large-xlsr-53-english/special_tokens_map.json",
                "wav2vec2-large-xlsr-53-english/vocab.json",
                "wav2vec2-large-xlsr-53-english/model.safetensors",
            ],
        )

    def _load_model(self):
        from xfuser.model_executor.models.transformers.transformer_wan_s2v import (
            import_official_wan,
        )

        wan = import_official_wan()
        from wan.configs.wan_s2v_14B import s2v_14B

        checkpoint_path = self._resolve_checkpoint()
        world_group = get_world_group()
        log(f"Loading official Wan2.2-S2V engine from {checkpoint_path}")
        output_owner_rank = 0 if world_group.rank == world_group.world_size - 1 else 1
        engine = wan.WanS2V(
            config=copy.deepcopy(s2v_14B),
            checkpoint_dir=checkpoint_path,
            device_id=world_group.local_rank,
            rank=output_owner_rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,
            init_on_cpu=True,
            convert_model_dtype=False,
        )
        return _WanS2VRuntimePipeline(engine)

    def initialize(self, input_args: dict) -> None:
        if not torch.distributed.is_initialized():
            init_distributed_environment()
        self.engine_config, _ = self.config.create_config()
        self.pipe = self._load_model_checked()
        initialize_runtime_state(self.pipe, self.engine_config)
        from xfuser.model_executor.models.transformers.transformer_wan_s2v import (
            patch_wan_s2v_for_xdit,
        )

        patch_wan_s2v_for_xdit(self.pipe.transformer)
        device = torch.device(f"cuda:{get_world_group().local_rank}")
        self.pipe.transformer.to(device)
        if self.config.use_fp8_gemms:
            log("Quantizing Wan2.2-S2V transformer blocks to FP8.")
            quantize_linear_layers_to_fp8(
                self.pipe.transformer.blocks,
                device=device,
            )
        elif self.config.use_fp4_gemms:
            log("Quantizing Wan2.2-S2V transformer blocks to FP4.")
            from xfuser.model_executor.models.transformers.transformer_wan_s2v import (
                patch_s2v_block_for_low_precision,
            )

            for block in self.pipe.transformer.blocks:
                patch_s2v_block_for_low_precision(block)
            quantize_linear_layers_to_fp4(
                self.pipe.transformer.blocks,
                fp8_layers=self.settings.fp8_precision_overrides,
                fp8_suffix_layers=self.settings.fp8_precision_override_suffixes,
                device=device,
            )
        self.pipe.engine.init_on_cpu = False
        if self.config.use_torch_compile:
            for index, block in enumerate(self.pipe.transformer.blocks):
                self.pipe.transformer.blocks[index] = torch.compile(
                    block,
                    mode=self._get_compile_mode(),
                )

    def _run_warmup_calls(self, input_args: dict) -> None:
        if not self.config.warmup_calls:
            return
        warmup_args = copy.deepcopy(input_args)
        warmup_args["num_inference_steps"] = 1
        for iteration in range(self.config.warmup_calls):
            log(f"S2V compile warmup {iteration + 1}/{self.config.warmup_calls}")
            self._run_timed_pipe(warmup_args)

    def preprocess_args(self, input_args: dict) -> dict:
        args = copy.deepcopy(input_args)
        for key in DefaultInputValues.__annotations__:
            if args.get(key) is None:
                value = getattr(self.default_input_values, key)
                if value is not None:
                    args[key] = value
        if isinstance(args.get("prompt"), list) and len(args["prompt"]) == 1:
            args["prompt"] = args["prompt"][0]
        os.makedirs(args["output_directory"], exist_ok=True)
        self._validate_args(args)
        return args

    def _validate_args(self, input_args: dict) -> None:
        images = input_args.get("input_images") or []
        if len(images) != 1:
            raise ValueError("Wan2.2-S2V requires exactly one reference image.")
        audio_path = input_args.get("input_audio")
        if not audio_path:
            raise ValueError("Wan2.2-S2V requires --input_audio.")
        if not os.path.isfile(images[0]):
            raise ValueError(f"Reference image does not exist: {images[0]}")
        if not os.path.isfile(audio_path):
            raise ValueError(f"Input audio does not exist: {audio_path}")
        prompt = input_args.get("prompt")
        if isinstance(prompt, list) and len(prompt) != 1:
            raise ValueError("Wan2.2-S2V supports one prompt per request.")
        if input_args["num_frames"] % 4:
            raise ValueError("Wan2.2-S2V --num_frames must be divisible by 4.")

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        video = self.pipe.engine.generate(
            input_prompt=input_args["prompt"],
            ref_image_path=input_args["input_images"][0],
            audio_path=input_args["input_audio"],
            enable_tts=False,
            tts_prompt_audio=None,
            tts_prompt_text=None,
            tts_text=None,
            num_repeat=None,
            pose_video=input_args.get("input_video"),
            max_area=input_args["height"] * input_args["width"],
            infer_frames=input_args["num_frames"],
            shift=input_args["flow_shift"],
            sampling_steps=input_args["num_inference_steps"],
            guide_scale=input_args["guidance_scale"],
            n_prompt=input_args["negative_prompt"],
            seed=input_args["seed"],
            offload_model=False,
        )
        if video is None:
            return WanAudioDiffusionOutput(
                videos=[],
                pipe_args=input_args,
                audio_path=input_args["input_audio"],
            )
        video = video.detach().float().cpu()
        if video.ndim == 4 and video.shape[0] in (1, 3):
            video = video.permute(1, 2, 3, 0)
        video = ((video + 1.0) / 2.0).clamp(0, 1).numpy()
        return WanAudioDiffusionOutput(
            videos=[video],
            pipe_args=input_args,
            audio_path=input_args["input_audio"],
        )

    def save_output(self, output: DiffusionOutput) -> None:
        for index, (video, pipe_args) in enumerate(output.get_outputs()):
            output_name = self.get_output_name(pipe_args)
            output_path = f"{self.config.output_directory}/{output_name}_{index}.mp4"
            silent_path = (
                f"{self.config.output_directory}/.{output_name}_{index}.silent.mp4"
            )
            export_to_video(video, silent_path, fps=self.settings.fps)
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        silent_path,
                        "-i",
                        output.audio_path,
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-shortest",
                        output_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            finally:
                if os.path.exists(silent_path):
                    os.remove(silent_path)
            log(f"Output video with audio saved to {output_path}")


@register_model("Wan-AI/Wan-Dancer-14B")
@register_model("Wan-Dancer-14B")
class xFuserWanDancerModel(xFuserModel):
    default_input_values = DefaultInputValues(
        height=1280,
        width=720,
        num_frames=149,
        num_inference_steps=48,
        negative_prompt=None,
        guidance_scale=5.0,
        flow_shift=5.0,
    )
    settings = ModelSettings(
        model_name="Wan-AI/Wan-Dancer-14B",
        output_name="wan_dancer",
        model_output_type="video",
        fps=30,
        valid_tasks=["global", "local"],
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=False,
        data_parallel_degree=False,
        use_cfg_parallel=False,
        fully_shard_degree=False,
        use_fp8_gemms=False,
        use_fp4_gemms=False,
        use_parallel_vae=False,
        enable_slicing=False,
        enable_tiling=False,
    )

    def _validate_config(self, config) -> None:
        if config.task is None:
            config.task = "global"
        super()._validate_config(config)
        if config.ulysses_degree != 8:
            raise ValueError("Wan-Dancer currently requires --ulysses_degree 8.")
        if config.batch_size is not None or config.dataset_path is not None:
            raise ValueError("Wan-Dancer currently supports one request at a time.")

    @staticmethod
    def _resolve_repo_path() -> str:
        repo_path = os.environ.get("WAN_DANCER_REPO_PATH")
        if not repo_path:
            raise ImportError(
                "Wan-Dancer support requires the official Wan-Dancer checkout. "
                "Set WAN_DANCER_REPO_PATH to the repository path."
            )
        if not os.path.isfile(
            os.path.join(repo_path, "diffsynth", "pipelines", "wan_video_new.py")
        ):
            raise ImportError(f"Invalid WAN_DANCER_REPO_PATH: {repo_path}")
        return repo_path

    @staticmethod
    def _patch_dancer_distributed_initialization(pipeline_class) -> None:
        def initialize_usp(pipe, usp_config=None):
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend="nccl",
                    init_method="env://",
                )
            init_distributed_environment(
                rank=torch.distributed.get_rank(),
                world_size=torch.distributed.get_world_size(),
            )
            initialize_model_parallel(
                data_parallel_degree=1,
                sequence_parallel_degree=torch.distributed.get_world_size(),
                ring_degree=1,
                ulysses_degree=torch.distributed.get_world_size(),
            )
            torch.cuda.set_device(get_world_group().local_rank)

        pipeline_class.initialize_usp = initialize_usp

    def _load_model(self):
        repo_path = self._resolve_repo_path()
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        import transformers
        import transformers.modeling_utils as modeling_utils

        if not hasattr(modeling_utils, "PretrainedConfig"):
            modeling_utils.PretrainedConfig = transformers.PreTrainedConfig
        from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

        self._patch_dancer_distributed_initialization(WanVideoPipeline)
        model_filename = (
            "global_model.safetensors"
            if self.config.task == "global"
            else "local_model.safetensors"
        )
        snapshot_path = _resolve_shared_snapshot(
            self.settings.model_name,
            allow_patterns=[
                model_filename,
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ],
        )
        common_snapshot_path = _resolve_wan_common_snapshot()
        model_configs = [
            ModelConfig(
                path=os.path.join(snapshot_path, model_filename),
                offload_device="cpu",
            ),
            ModelConfig(
                path=os.path.join(
                    common_snapshot_path,
                    "models_t5_umt5-xxl-enc-bf16.pth",
                ),
                offload_device="cpu",
            ),
            ModelConfig(
                path=os.path.join(common_snapshot_path, "Wan2.1_VAE.pth"),
                offload_device="cpu",
            ),
            ModelConfig(
                path=os.path.join(
                    snapshot_path,
                    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                ),
                offload_device="cpu",
            ),
        ]
        tokenizer_config = ModelConfig(
            path=os.path.join(common_snapshot_path, "google", "umt5-xxl"),
        )
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            skip_download=True,
            redirect_common_files=False,
            use_usp=True,
            usp_config={
                "data_parallel_degree": 1,
                "sequence_parallel_degree": 8,
                "ring_degree": 1,
                "ulysses_degree": 8,
            },
            dit_model_type=1,
            enable_music_inject=True,
            enable_refimage=True,
            enable_global=True,
            enable_dynamicfps=True,
            enable_unimodel=True,
        )
        pipe.enable_vram_management()
        script_name = (
            "gen_video_global.py"
            if self.config.task == "global"
            else "gen_video_local.py"
        )
        self._dancer_script = _load_module_from_path(
            f"wan_dancer_{self.config.task}",
            os.path.join(repo_path, "gen_video", script_name),
        )
        self._dancer_script.get_music_base_feature = extract_dancer_music_feature
        self._dancer_script.mpy.VideoFileClip = _DancerVideoClip
        self._dancer_script.mpy.AudioFileClip = _DancerAudioClip
        self._dancer_script.mpy.video.fx.Crop = _DancerCrop
        return pipe

    def initialize(self, input_args: dict) -> None:
        self.pipe = self._load_model_checked()
        self.engine_config, _ = self.config.create_config()
        initialize_runtime_state(engine_config=self.engine_config)
        from xfuser.model_executor.models.transformers.transformer_wan_dancer import (
            patch_wan_dancer_for_xdit,
        )

        patch_wan_dancer_for_xdit(self.pipe)
        if self.config.use_torch_compile:
            for index, block in enumerate(self.pipe.dit.blocks):
                self.pipe.dit.blocks[index] = torch.compile(
                    block,
                    mode=self._get_compile_mode(),
                )

    def _run_warmup_calls(self, input_args: dict) -> None:
        if not self.config.warmup_calls:
            return
        warmup_args = copy.deepcopy(input_args)
        warmup_args["num_inference_steps"] = 1
        for iteration in range(self.config.warmup_calls):
            log(
                f"Wan-Dancer compile warmup "
                f"{iteration + 1}/{self.config.warmup_calls}"
            )
            self._run_timed_pipe(warmup_args)

    def preprocess_args(self, input_args: dict) -> dict:
        args = copy.deepcopy(input_args)
        for key in DefaultInputValues.__annotations__:
            if args.get(key) is None:
                value = getattr(self.default_input_values, key)
                if value is not None:
                    args[key] = value
        if isinstance(args.get("prompt"), list) and len(args["prompt"]) == 1:
            args["prompt"] = args["prompt"][0]
        os.makedirs(args["output_directory"], exist_ok=True)
        self._validate_args(args)
        return args

    def _validate_args(self, input_args: dict) -> None:
        images = input_args.get("input_images") or []
        if len(images) != 1:
            raise ValueError("Wan-Dancer requires exactly one reference image.")
        if not input_args.get("input_audio"):
            raise ValueError("Wan-Dancer requires --input_audio.")
        if not os.path.isfile(images[0]):
            raise ValueError(f"Reference image does not exist: {images[0]}")
        if not os.path.isfile(input_args["input_audio"]):
            raise ValueError(f"Input music does not exist: {input_args['input_audio']}")
        if input_args["num_frames"] != 149:
            raise ValueError("Wan-Dancer currently requires 149 frames per segment.")
        if self.config.task == "local":
            input_video = input_args.get("input_video")
            if not input_video:
                raise ValueError("Wan-Dancer local refinement requires --input_video.")
            if not os.path.isfile(input_video):
                raise ValueError(f"Input global video does not exist: {input_video}")

    def _run_global(self, input_args: dict, output_path: str) -> None:
        output_name = self.get_output_name(input_args)
        feature_path = os.path.join(
            self.config.output_directory,
            f".{output_name}.music.npy",
        )
        silent_path = os.path.join(
            self.config.output_directory,
            f".{output_name}.silent.mp4",
        )
        if get_world_group().rank == get_world_group().first_rank:
            self._dancer_script.get_music_base_feature(
                input_args["input_audio"],
                feature_path,
                fps=self.settings.fps,
            )
        torch.distributed.barrier()
        try:
            self._dancer_script.gen_video(
                self.pipe,
                feature_path,
                input_args["input_images"][0],
                input_args["prompt"],
                silent_path,
                seed=input_args["seed"],
                height=input_args["height"],
                width=input_args["width"],
                num_frames=input_args["num_frames"],
                enable_refimage=True,
                refimage_path=input_args["input_images"][0],
                enable_global=True,
                enable_dynamicfps=True,
                enable_vae_decode_framewise=True,
                enable_skip_layer=True,
                enable_unimodel=True,
                sigma_shift=input_args["flow_shift"],
                num_inference_steps=input_args["num_inference_steps"],
                cfg_scale=input_args["guidance_scale"],
            )
        except (OSError, subprocess.CalledProcessError):
            if get_world_group().rank != get_world_group().first_rank:
                raise
            temporary_path = silent_path[:-4] + "_tmp.mp4"
            if not os.path.isfile(temporary_path):
                raise
            if os.path.exists(silent_path):
                os.remove(silent_path)
            os.replace(temporary_path, silent_path)
        torch.distributed.barrier()
        if get_world_group().rank == get_world_group().first_rank:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    silent_path,
                    "-i",
                    input_args["input_audio"],
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    output_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.remove(silent_path)
            os.remove(feature_path)
            log(f"Wan-Dancer global video saved to {output_path}")

    def _run_local(self, input_args: dict, output_path: str) -> None:
        import soundfile as sf

        output_name = self.get_output_name(input_args)
        work_directory = os.path.join(
            self.config.output_directory,
            f".{output_name}.segments",
        )
        if get_world_group().rank == get_world_group().first_rank:
            os.makedirs(work_directory, exist_ok=True)
        torch.distributed.barrier()

        world_group = get_world_group()
        duration = [None]
        if world_group.rank == world_group.first_rank:
            duration[0] = sf.info(input_args["input_audio"]).duration
        torch.distributed.broadcast_object_list(
            duration,
            src=world_group.first_rank,
            group=world_group.cpu_group,
        )
        total_duration = duration[0]
        total_frames = int(total_duration * self.settings.fps)
        keyframes, keyframe_masks = (
            self._dancer_script.process_global_video_firstlastframe(
                input_args["input_video"],
                input_args["height"],
                input_args["width"],
                total_frames,
            )
        )
        if get_world_group().rank == get_world_group().first_rank:
            self._dancer_script.get_music_clip_149f(
                input_args["input_audio"],
                work_directory,
            )
            self._dancer_script.get_music_features(work_directory)
        torch.distributed.barrier()

        segment_paths = []
        audio_segments = sorted(
            name for name in os.listdir(work_directory) if name.endswith(".wav")
        )
        for index, name in enumerate(audio_segments):
            music_path = os.path.join(work_directory, name)
            feature_path = os.path.join(
                work_directory,
                name.replace(".wav", "_librosa_feature.npy"),
            )
            segment_path = os.path.join(
                work_directory,
                f"segment_{index:03d}.mp4",
            )
            music_segment_path = segment_path[:-4] + "_music.mp4"
            try:
                self._dancer_script.gen_video(
                    self.pipe,
                    music_path,
                    feature_path,
                    input_args["prompt"],
                    segment_path,
                    seed=input_args["seed"] + index * 10,
                    height=input_args["height"],
                    width=input_args["width"],
                    num_frames=input_args["num_frames"],
                    enable_refimage=True,
                    refimage_path=input_args["input_images"][0],
                    keyframes=keyframes[index],
                    keyframes_mask=keyframe_masks[index],
                    enable_dynamicfps=True,
                    enable_skip_layer=True,
                    sigma_shift=input_args["flow_shift"],
                    num_inference_steps=input_args["num_inference_steps"],
                )
            except (OSError, subprocess.CalledProcessError):
                if get_world_group().rank != get_world_group().first_rank:
                    raise
                if not os.path.isfile(segment_path):
                    raise
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        segment_path,
                        "-i",
                        music_path,
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-shortest",
                        music_segment_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            torch.distributed.barrier()
            segment_paths.append(music_segment_path)

        torch.distributed.barrier()
        if get_world_group().rank == get_world_group().first_rank:
            if not segment_paths:
                raise RuntimeError("Wan-Dancer local refinement produced no segments.")
            concat_path = os.path.join(work_directory, "segments.txt")
            with open(concat_path, "w", encoding="utf-8") as file:
                for path in segment_paths:
                    escaped_path = path.replace("'", "'\\''")
                    file.write(f"file '{escaped_path}'\n")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    concat_path,
                    "-i",
                    input_args["input_audio"],
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    output_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            shutil.rmtree(work_directory)
            log(f"Wan-Dancer refined video saved to {output_path}")

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output_name = self.get_output_name(input_args)
        output_path = os.path.join(
            self.config.output_directory,
            f"{output_name}.mp4",
        )
        if self.config.task == "global":
            self._run_global(input_args, output_path)
        else:
            self._run_local(input_args, output_path)
        return DiffusionOutput(videos=[], pipe_args=input_args)

    def save_output(self, output: DiffusionOutput) -> None:
        return
