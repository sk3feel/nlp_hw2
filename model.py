from pathlib import Path
import torch
from transformers import BartConfig, BartForConditionalGeneration
from safetensors.torch import load_file, save_file


def model_config():
    config = BartConfig(
        vocab_size=8192, d_model=768,
        encoder_layers=6, decoder_layers=6,
        encoder_attention_heads=12, decoder_attention_heads=12,
        encoder_ffn_dim=3072, decoder_ffn_dim=3072,
        activation_function="gelu", dropout=0.1,
        attention_dropout=0.0, activation_dropout=0.0,
        encoder_layerdrop=0.0, decoder_layerdrop=0.0, classifier_dropout=0.0,
        max_position_embeddings=512, init_std=0.02, scale_embedding=False,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1,
        forced_eos_token_id=None, use_cache=True, num_labels=3,
        torch_dtype="float32", architectures=["BartForConditionalGeneration"])
    config._attn_implementation = "sdpa"
    return config


def create_model():
    model = BartForConditionalGeneration(model_config())
    assert sum(p.numel() for p in model.parameters()) == 106321920
    return model


def average_checkpoints(folders, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Не перезаписываем готовые веса: " + str(output))
    average = None
    config = None
    dtypes = None
    for folder in map(Path, folders):
        current_config = BartConfig.from_pretrained(folder, local_files_only=True)
        if config is None:
            config = current_config
        else:
            left, right = config.to_dict(), current_config.to_dict()
            left.pop("_name_or_path", None)
            right.pop("_name_or_path", None)
            assert left == right
        weights = load_file(str(folder / "model.safetensors"), device="cpu")
        if average is None:
            average = {k: v.float().clone() if v.is_floating_point() else v.clone()
                       for k, v in weights.items()}
            dtypes = {k: v.dtype for k, v in weights.items()}
        else:
            assert average.keys() == weights.keys()
            for name, value in weights.items():
                assert value.shape == average[name].shape and value.dtype == dtypes[name]
                if value.is_floating_point():
                    average[name].add_(value.float())
                else:
                    assert torch.equal(average[name], value)
    for value in average.values():
        if value.is_floating_point():
            value.div_(len(folders))
        assert bool(torch.isfinite(value).all())
    for alias in ("model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "lm_head.weight"):
        if alias in average:
            assert torch.equal(average[alias], average["model.shared.weight"])
    output.mkdir(parents=True)
    save_file(average, str(output / "model.safetensors"), metadata={"format": "pt"})
    config.save_pretrained(output)
    return output
