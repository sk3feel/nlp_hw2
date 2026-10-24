from pathlib import Path
import contextlib
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from data import collate
from model import create_model, model_config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


class UpdateOrder:
    def __init__(self, dataset, seed, epoch, effective=256):
        self.effective = effective
        rng = np.random.default_rng(seed + epoch)
        self.order = rng.permutation(len(dataset))
        width = effective * 50
        for start in range(0, len(dataset), width):
            chunk = self.order[start:start + width]
            self.order[start:start + width] = chunk[
                np.argsort(dataset.lengths[chunk], kind="stable")]
                                                                       
        self.blocks = rng.permutation(math.ceil(len(dataset) / effective))

    def __len__(self):
        return len(self.blocks)

    def block(self, position):
        start = int(self.blocks[position]) * self.effective
        return self.order[start:start + self.effective]


def lr_factor(step, config):
    warmup, horizon = config["warmup_steps"], config["planned_steps"]
    if step < warmup:
        return max(1, step) / warmup
    progress = min(1.0, (step - warmup) / max(1, horizon - warmup))
    floor = config["floor_lr"] / config["peak_lr"]
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizer(model, config, device):
    return torch.optim.AdamW(
        model.parameters(), lr=config["peak_lr"], betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.01, fused=device.type == "cuda")


def make_scheduler(optimizer, config):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, config))


def backward_block(model, dataset, indices, physical, device, dtype=torch.bfloat16):
    total_tokens = int((dataset.target_offsets[indices + 1] - dataset.target_offsets[indices]).sum())
    total_loss, source_tokens = 0.0, 0
    model.train()
    for start in range(0, len(indices), physical):
        batch = collate([dataset[int(i)] for i in indices[start:start + physical]])
        tokens = int(batch["labels"].ne(-100).sum())
        source_tokens += int(batch["attention_mask"].sum())
        batch = {key: value.to(device) for key, value in batch.items()}
        amp = torch.autocast(device.type, dtype=dtype) if dtype is not None else contextlib.nullcontext()
        with amp:
            loss = model(**batch, use_cache=False).loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite loss")
                                                                 
        (loss * (tokens / total_tokens)).backward()
        total_loss += float(loss.detach()) * tokens
    return dict(loss=total_loss / total_tokens, target_tokens=total_tokens,
                source_tokens=source_tokens, examples=len(indices))


def validation_ce(model, dataset, device, dtype=torch.bfloat16):
    saved_rng = rng_state()
    total_loss, total_tokens = 0.0, 0
    try:
        model.eval()
        for batch in DataLoader(dataset, batch_size=32, collate_fn=collate):
            count = int(batch["labels"].ne(-100).sum())
            batch = {key: value.to(device) for key, value in batch.items()}
            amp = torch.autocast(device.type, dtype=dtype) if dtype is not None else contextlib.nullcontext()
            with torch.inference_mode(), amp:
                loss = model(**batch, use_cache=False).loss
            assert bool(torch.isfinite(loss))
            total_loss += float(loss) * count
            total_tokens += count
        return total_loss / total_tokens
    finally:
                                                                                   
        restore_rng(saved_rng)


def compatibility(config, identity):
    keys = ("seed", "physical_batch", "effective_batch", "precision", "peak_lr",
            "warmup_steps", "planned_steps", "floor_lr", "clip_grad")
    return dict(training={key: config[key] for key in keys},
                model=model_config().to_diff_dict(),
                tokenizer=identity["tokenizer_sha256"], data=identity["cache_sha256"])


def save_full(path, model, optimizer, scheduler, state, config, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(dict(format="student_bart_v2", model=model.state_dict(),
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    rng=rng_state(), state=dict(state),
                    compatibility=compatibility(config, identity)), temporary)
    temporary.replace(path)


def load_full(path, model, optimizer, scheduler, config, identity):
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if saved["format"] != "student_bart_v2" or saved["compatibility"] != compatibility(config, identity):
        raise ValueError("Несовместимые checkpoint, данные или параметры обучения")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    state = saved["state"]
    expected = config["peak_lr"] * lr_factor(state["updates"], config)
    if scheduler.last_epoch != state["updates"] or not all(
            math.isclose(group["lr"], expected, rel_tol=1e-12) for group in optimizer.param_groups):
        raise ValueError("Scheduler и счётчик updates не согласованы")
    restore_rng(saved["rng"])
    return state


def training_device(config):
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна")
    precision = config["precision"]
    if precision not in ("bf16", "fp32"):
        raise ValueError("Для обучения выберите bf16 или fp32")
    if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Для BF16 нужна поддерживающая его CUDA GPU; иначе выберите fp32")
    return device, torch.bfloat16 if precision == "bf16" else None


def train_step(model, dataset, order, state, optimizer, scheduler, config, device, dtype):
    started = time.perf_counter()
    indices = order.block(state["update_in_epoch"])
    optimizer.zero_grad(set_to_none=True)
    metrics = backward_block(model, dataset, indices, config["physical_batch"], device, dtype)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("Nonfinite gradients")
    lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    state["updates"] += 1
    state["update_in_epoch"] += 1
    for key in ("examples", "target_tokens", "source_tokens"):
        state[key] += metrics[key]
    if state["update_in_epoch"] == len(order):
        state["sampler_epoch"] += 1
        state["update_in_epoch"] = 0
        order = UpdateOrder(dataset, config["seed"], state["sampler_epoch"], config["effective_batch"])
    metrics.update(updates=state["updates"], lr=lr, grad_norm=float(norm),
                   examples_per_second=len(indices) / (time.perf_counter() - started))
    return order, metrics


def fit(train, validation, config, identity, output, resume=None):
    device, dtype = training_device(config)
    if config["effective_batch"] != 256 or 256 % config["physical_batch"]:
        raise ValueError("Нужен effective batch256 и делящий его physical batch")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(config["seed"])
    model = create_model().to(device)
    optimizer = make_optimizer(model, config, device)
    scheduler = make_scheduler(optimizer, config)
    state = dict(updates=0, examples=0, target_tokens=0, source_tokens=0,
                 sampler_epoch=0, update_in_epoch=0)
    if resume is not None:
        state = load_full(resume, model, optimizer, scheduler, config, identity)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    order = UpdateOrder(train, config["seed"], state["sampler_epoch"], config["effective_batch"])
    assert state["updates"] == state["sampler_epoch"] * len(order) + state["update_in_epoch"]
    print("Старт:", state, "LR:", optimizer.param_groups[0]["lr"])
    while state["updates"] < config["stop_step"]:
        order, metrics = train_step(model, train, order, state, optimizer, scheduler,
                                    config, device, dtype)
        if state["updates"] % config.get("log_every", 1000) == 0:
            print(metrics, flush=True)
        if state["updates"] in config["save_steps"]:
            folder = output / f'step{state["updates"]:09d}'
            if folder.exists():
                raise FileExistsError("Не перезаписываем сохранённый snapshot")
            saved_rng = rng_state()
            model.save_pretrained(folder, safe_serialization=True)
            state["last_ce"] = validation_ce(model, validation, device, dtype)
            restore_rng(saved_rng)
            print("Сохранён", folder, "CE:", state["last_ce"])
        if state["updates"] % config["save_every"] == 0 or state["updates"] in config["save_steps"]:
            save_full(output / "last.pt", model, optimizer, scheduler, state, config, identity)
    save_full(output / "last.pt", model, optimizer, scheduler, state, config, identity)
    return state
