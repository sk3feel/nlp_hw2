import json
import math
import time
from pathlib import Path

import torch
from transformers import BartForConditionalGeneration
from tokenizers import Tokenizer

from data import sha256


def generate(model, tokenizer, questions, batch_size=32, max_new_tokens=256):
    device = next(model.parameters()).device
    result = []
    for start in range(0, len(questions), batch_size):
        texts = questions[start:start + batch_size]
        encoded = [[1] + row.ids + [2] for row in tokenizer.encode_batch(texts)]
        if max(map(len, encoded)) > 192:
            raise ValueError("Вопрос длиннее192 токенов; не обрезаем молча")
        width = math.ceil(max(map(len, encoded)) / 8) * 8
        ids = torch.zeros((len(texts), width), dtype=torch.long, device=device)
        for i, row in enumerate(encoded):
            ids[i, :len(row)] = torch.tensor(row, device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            outputs = model.generate(
                input_ids=ids, attention_mask=ids.ne(0), do_sample=False, num_beams=1,
                use_cache=True, max_new_tokens=max_new_tokens, forced_eos_token_id=None,
                pad_token_id=0, eos_token_id=2, decoder_start_token_id=1)
        for row in outputs.cpu().tolist():
            row = row[1:]
            if 2 in row:
                row = row[:row.index(2) + 1]
            result.append(tokenizer.decode(row, skip_special_tokens=True))
    return result


def run_inference(weights, assets, output, started=None, historical=True):
    started = time.perf_counter() if started is None else started
    weights, assets, output = map(Path, (weights, assets, output))
    identity = json.loads((assets / "identity.json").read_text())
    weights_sha = sha256(weights / "model.safetensors")
    if historical and weights_sha != identity["weights_sha256"]:
        raise ValueError("Неверный hash avg3_mid")
    for path, expected in [(assets / "tokenizer.json", identity["tokenizer_sha256"]),
                           (assets / "test_questions.jsonl", identity["test_sha256"])]:
        if sha256(path) != expected:
            raise ValueError("Нужен проверенный avg3_mid/tokenizer/test: " + str(path))
    rows = [json.loads(line) for line in (assets / "test_questions.jsonl").read_text().splitlines()]
    assert len(rows) == 1919 and all(row["index"] == i for i, row in enumerate(rows))
    questions = [row["question"] for row in rows]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Нужна одна видимая CUDA GPU")
    t = time.perf_counter()
    tokenizer = Tokenizer.from_file(str(assets / "tokenizer.json"))
    model, loading = BartForConditionalGeneration.from_pretrained(
        weights, local_files_only=True, attn_implementation="sdpa", output_loading_info=True)
    if loading["missing_keys"] or loading["unexpected_keys"]:
        raise ValueError("Неполные/несовместимые веса")
    model = model.cuda().eval()
    assert sum(p.numel() for p in model.parameters()) == 106321920
    assert model.model.shared.weight.data_ptr() == model.lm_head.weight.data_ptr()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - t
    t = time.perf_counter()
    generate(model, tokenizer, questions[:8], batch_size=8)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - t
    t = time.perf_counter()
    programs = generate(model, tokenizer, questions)
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - t
    output.mkdir(parents=True, exist_ok=True)
    submission = output / "submission.jsonl"
    submission.write_text("".join(json.dumps({"code": code}, ensure_ascii=True) + "\n" for code in programs))
    check = [json.loads(line) for line in submission.read_text().splitlines()]
    assert len(check) == 1919 and [row["code"] for row in check] == programs
    assert all(set(row) == {"code"} and isinstance(row["code"], str) for row in check)
    timing = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  load_seconds=load_seconds, warmup_seconds=warmup_seconds,
                  generation_seconds=generation_seconds, full_inference_seconds=time.perf_counter() - started,
                  scope="inference cell: imports if needed, identity checks, load, warmup,1919 generation,submission write; excludes pip/data preparation",
                  rows=1919, weights_sha256=weights_sha, historical_weights=historical,
                  test_sha256=identity["test_sha256"], programs_executed=False)
    (output / "timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2))
    return submission, timing
