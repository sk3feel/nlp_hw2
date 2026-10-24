from array import array
from pathlib import Path
import ast
import hashlib
import json
import math
import random
import shutil
import sqlite3
import tempfile
import zipfile

import numpy as np
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PAD, BOS, EOS = 0, 1, 2


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def question_group(question):
    return hashlib.sha256(" ".join(question.split()).encode()).digest()


def held_out(group):
    return int.from_bytes(group, "big") % 200 == 0


def valid_program_pair(question, code):
    if not isinstance(question, str) or not isinstance(code, str):
        return False
    if not question.strip() or not code.strip():
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return False
    functions = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == "solve"]
    if not functions or isinstance(functions[-1], ast.AsyncFunctionDef):
        return False
    args = functions[-1].args
    return (len(args.posonlyargs) + len(args.args) <= len(args.defaults)
            and all(default is not None for default in args.kw_defaults))


def extend_selection(previous, size, target, shard):
    target = min(target, size)
    previous = set(previous)
    selected = set(previous)
    rng = random.Random(90042 + shard)
    while len(selected) < target:
        selected.update(rng.sample(range(size), min(2 * (target - len(selected)), size)))
    extras = random.Random(91042 + shard).sample(
        sorted(selected - previous), target - len(previous))
    return np.array(sorted(previous | set(extras)), dtype=np.int32)


def select_indices(raw_dir, assets):
    initial, expanded = {}, {}
    saved_initial = np.load(Path(assets) / "tokenizer_sample_indices.npz")
    saved_expanded = np.load(Path(assets) / "train_indices.npz")
    for shard in range(60):
        name = f"part-{shard:05d}.parquet"
        size = pq.ParquetFile(Path(raw_dir) / name).metadata.num_rows
        first = np.array(sorted(random.Random(42 + shard).sample(
            range(size), min(5000, size))), dtype=np.int32)
        middle = extend_selection(first, size, 40000, shard)
        last = extend_selection(middle, size, 100000, shard)
        if not np.array_equal(first, saved_initial[name]) or not np.array_equal(last, saved_expanded[name]):
            raise ValueError("Исходные строки/порядок не совпали с сохранённой выборкой: " + name)
        initial[name], expanded[name] = first, last
    return initial, expanded


def initial_rows(raw_dir, selection):
    rows, seen = [], set()
    for name, indices in selection.items():
        table = pq.read_table(Path(raw_dir) / name, columns=["question", "code"])
        for row in table.take(indices).to_pylist():
            q, c = row["question"], row["code"]
            if not valid_program_pair(q, c):
                continue
            pair = hashlib.sha256((q + "\0" + c).encode()).digest()
            if pair not in seen:
                seen.add(pair)
                rows.append((q, c, question_group(q)))
    return rows


def train_tokenizer(rows, output):
    pool = [row for row in rows if not held_out(row[2])]
    chosen = random.Random(42).sample(pool, min(100000, len(pool)))
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=8192, min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator((text for q, c, _ in chosen for text in (q, c)), trainer)
    tokenizer.save(str(output))
    return tokenizer


def encoded_rows(rows, tokenizer):
    chunk = []
    for row in rows:
        chunk.append(row)
        if len(chunk) == 2000:
            yield from encode_chunk(chunk, tokenizer)
            chunk = []
    yield from encode_chunk(chunk, tokenizer)


def encode_chunk(rows, tokenizer):
    if not rows:
        return
    sources = tokenizer.encode_batch([q for q, _, _ in rows])
    targets = tokenizer.encode_batch([c for _, c, _ in rows])
    for row, source, target in zip(rows, sources, targets):
        src, tgt = [BOS] + source.ids + [EOS], target.ids + [EOS]
        if len(src) <= 192 and len(tgt) <= 384:
            yield row, src, tgt


def write_cache(rows, output, split, database=None):
    cache = Path(output) / "cache"
    offsets = {key: array("Q", [0]) for key in ("source", "target")}
    count = 0
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        tmp = Path(tmp)
        streams = {key: (tmp / key).open("wb") for key in offsets}
        groups = (tmp / "groups").open("wb")
        try:
            for (q, c, group), src, tgt in rows:
                if database is not None:
                    pair = hashlib.sha256((q + "\0" + c).encode()).digest()
                    if not database.execute("INSERT OR IGNORE INTO pairs VALUES (?)", (pair,)).rowcount:
                        continue
                for key, ids in (("source", src), ("target", tgt)):
                    array("H", ids).tofile(streams[key])
                    offsets[key].append(offsets[key][-1] + len(ids))
                groups.write(group)
                count += 1
        finally:
            for stream in streams.values():
                stream.close()
            groups.close()
        for key in offsets:
            flat = np.memmap(tmp / key, mode="r", dtype=np.uint16)
            dest = np.lib.format.open_memmap(cache / f"{split}_{key}.npy",
                                           mode="w+", dtype=np.uint16, shape=flat.shape)
            for start in range(0, len(flat), 10_000_000):
                dest[start:start + 10_000_000] = flat[start:start + 10_000_000]
            dest.flush()
            del flat, dest
            np.save(cache / f"{split}_{key}_offsets.npy", np.asarray(offsets[key], dtype=np.int64))
        if split == "train":
            flat = np.memmap(tmp / "groups", mode="r", dtype=np.uint8, shape=(count, 32))
            dest = np.lib.format.open_memmap(Path(output) / "train_groups.npy",
                                           mode="w+", dtype=np.uint8, shape=flat.shape)
            for start in range(0, count, 65536):
                dest[start:start + 65536] = flat[start:start + 65536]
            dest.flush()
            del flat, dest
    return count


def prepare_data(raw_dir, output, assets, retrain_tokenizer=False):
    raw_dir, output, assets = map(Path, (raw_dir, output, assets))
    if output.exists():
        raise FileExistsError("Выберите новый каталог, чтобы не менять готовые данные")
    for name, digest in json.loads((assets / "raw_sha256.json").read_text()).items():
        if sha256(raw_dir / name) != digest:
            raise ValueError("Не тот исходный parquet: " + name)
    initial, expanded = select_indices(raw_dir, assets)
    rows = initial_rows(raw_dir, initial)
    output.mkdir(parents=True)
    (output / "cache").mkdir()
    if retrain_tokenizer:
        train_tokenizer(rows, output / "retrained_tokenizer.json")
    tokenizer = Tokenizer.from_file(str(assets / "tokenizer.json"))
    validation = {}
    for row in rows:
        if held_out(row[2]):
            validation.setdefault(row[2].hex(), row)
    selected = [validation[key] for key in sorted(validation)[:1500]]
    if sorted(row[2].hex() for row in selected) != json.loads((assets / "validation_groups.json").read_text()):
        raise ValueError("Состав validation изменился")
    write_cache(encoded_rows(iter(selected), tokenizer), output, "validation")
    del rows

    def training_rows():
        for name, indices in expanded.items():
            table = pq.read_table(raw_dir / name, columns=["question", "code"]).take(indices)
            for batch in table.to_batches(max_chunksize=2000):
                for row in batch.to_pylist():
                    q, c = row["question"], row["code"]
                    if valid_program_pair(q, c):
                        group = question_group(q)
                        if not held_out(group):
                            yield q, c, group
            print("Подготовлен", name, flush=True)
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        with sqlite3.connect(Path(tmp) / "pairs.sqlite") as db:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("PRAGMA cache_size=-32768")
            db.execute("CREATE TABLE pairs(pair BLOB PRIMARY KEY) WITHOUT ROWID")
            write_cache(encoded_rows(training_rows(), tokenizer), output, "train", db)
    verify_data(output, assets)
    return output


def unpack_prepared(assets):
    assets = Path(assets)
    output = assets / "colab_training_data"
    if not output.exists():
        identity = json.loads((assets / "identity.json").read_text())
        if sha256(assets / "prepared_data.zip") != identity["prepared_zip_sha256"]:
            raise ValueError("Повреждён prepared_data.zip")
        output.mkdir()
        names = list(identity["cache_sha256"])
        with zipfile.ZipFile(assets / "prepared_data.zip") as archive:
            for name in names:
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open("colab_training_data/" + name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    return output


def verify_data(prepared, assets):
    prepared, assets = Path(prepared), Path(assets)
    identity = json.loads((assets / "identity.json").read_text())
    if sha256(assets / "tokenizer.json") != identity["tokenizer_sha256"]:
        raise ValueError("Нужен исходный tokenizer")
    for name, digest in identity["cache_sha256"].items():
        if sha256(prepared / name) != digest:
            raise ValueError("Состав/порядок/token IDs кэша изменились: " + name)
    train, val = TokenDataset(prepared / "cache"), TokenDataset(prepared / "cache", "validation")
    assert len(train) == identity["train_rows"] and len(val) == identity["validation_rows"]
    reserved = json.loads((assets / "validation_groups.json").read_text())
    assert len(reserved) == 1500 and all(int(group, 16) % 200 == 0 for group in reserved)
    groups = np.load(prepared / "train_groups.npy", mmap_mode="r")
    assert groups.shape == (len(train), 32)
    for start in range(0, len(groups), 65536):
        remainder = np.zeros(len(groups[start:start + 65536]), dtype=np.uint32)
        for col in range(32):
            remainder = (remainder * 256 + groups[start:start + 65536, col]) % 200
        assert not np.any(remainder == 0), "Validation-группа попала вtrain"
    return train, val


class TokenDataset:
    def __init__(self, directory, split="train"):
        self.source = np.load(Path(directory) / f"{split}_source.npy", mmap_mode="r")
        self.target = np.load(Path(directory) / f"{split}_target.npy", mmap_mode="r")
        self.source_offsets = np.load(Path(directory) / f"{split}_source_offsets.npy", mmap_mode="r")
        self.target_offsets = np.load(Path(directory) / f"{split}_target_offsets.npy", mmap_mode="r")
        self.lengths = np.diff(self.source_offsets) + np.diff(self.target_offsets)

    def __len__(self):
        return len(self.source_offsets) - 1

    def __getitem__(self, i):
        src = self.source[self.source_offsets[i]:self.source_offsets[i + 1]]
        tgt = self.target[self.target_offsets[i]:self.target_offsets[i + 1]]
        return src.astype(np.int64), tgt.astype(np.int64), i


def collate(rows):
    sw = math.ceil(max(len(r[0]) for r in rows) / 8) * 8
    tw = math.ceil(max(len(r[1]) for r in rows) / 8) * 8
    source = torch.full((len(rows), sw), PAD, dtype=torch.long)
    labels = torch.full((len(rows), tw), -100, dtype=torch.long)
    for i, (src, tgt, _) in enumerate(rows):
        source[i, :len(src)] = torch.from_numpy(src)
        labels[i, :len(tgt)] = torch.from_numpy(tgt)
    decoder = torch.full_like(labels, PAD)
    decoder[:, 0] = BOS
    decoder[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1] == -100, PAD)
    return dict(input_ids=source, attention_mask=source.ne(PAD), labels=labels,
                decoder_input_ids=decoder, decoder_attention_mask=decoder.ne(PAD))
