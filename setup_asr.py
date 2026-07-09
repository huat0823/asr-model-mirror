#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_asr.py — 受限网络环境一键安装中文 ASR(sherpa-onnx + Paraformer)

只依赖: python3 + pypi.org + raw.githubusercontent.com(其余域名全被挡也能跑)

用法:
    python3 setup_asr.py              # 装小模型(82MB,识别中文+中英夹杂,推荐先跑通)
    python3 setup_asr.py --large      # 同时装大模型(243MB,精度更高)
    python3 setup_asr.py --dir DIR    # 安装目录(默认 ./asr-models)

错误全自动处理:
  - 429/503 限流 → 按 Retry-After / 指数退避等待后重试
  - 连接中断(curl exit 56 那种)→ 已下载字节保留,HTTP Range 断点续传
  - 每个文件下载后强制 SHA256 校验,不对就删掉重来
  - 大模型分块下载后自动拼接并校验整体哈希
  - 脚本幂等:随时 Ctrl-C,重跑会跳过已完成的文件、续传下到一半的文件
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get(
    "ASR_MIRROR_BASE",
    "https://raw.githubusercontent.com/huat0823/asr-model-mirror/main",
)
RETRY_CAP = int(os.environ.get("ASR_RETRY_CAP", "180"))  # 单次等待上限(秒)
MAX_TRIES = int(os.environ.get("ASR_MAX_TRIES", "15"))

SMALL = [
    {"path": "paraformer-zh-small/model.int8.onnx",
     "sha256": "3ef6c19369b912f7caf3cef8e545c5ccd1a33d9d7ec792a46668dc41c4b229ec",
     "size": 81828675},
    {"path": "paraformer-zh-small/tokens.txt",
     "sha256": "4b2d964e18b9cf139b473003b6698fb2ed9a2a5ec55b93daa677b28f578897aa",
     "size": 75352},
]
LARGE_PARTS = [
    {"path": "paraformer-zh-large/model.int8.onnx.part-aa",
     "sha256": "7a58e0b82289326626ff7bb8aab82902f003a0899fa9bc7c0efd6b2d09e632eb",
     "size": 99614720},
    {"path": "paraformer-zh-large/model.int8.onnx.part-ab",
     "sha256": "d7385e9a1b9396759c94ba119af6afa6e25df4462bd48b1eff9ff946c56f6258",
     "size": 99614720},
    {"path": "paraformer-zh-large/model.int8.onnx.part-ac",
     "sha256": "532cbaa0c38695a884edab170fb83e1778f846cd62f82dc2a80bd157028f500a",
     "size": 44141778},
]
LARGE_ASSEMBLED = {
    "sha256": "f36a0433bcf096bd6d6f11b80a3ac8bed110bdca632fe0d731df8d1a84475945",
    "size": 243371218,
}
LARGE_TOKENS = {"path": "paraformer-zh-large/tokens.txt",
                "sha256": "59aba8873a2ed1e122c25fee421e25f283b63290efbde85c1f01a853d83cb6e6",
                "size": 75756}
VAD = {"path": "vad/silero_vad.onnx",
       "sha256": "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
       "size": 643854}
TEST_WAV = {"path": "test/2-zh-en.wav",
            "sha256": "eddf384a906bd6d905c9d9d652d614def1857608b88c2eee663ceeccbb31f7a3",
            "size": 259278}


def log(msg):
    print(msg, flush=True)


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(item, dest_dir):
    """下载单个文件:限流退避 + 断点续传 + SHA256 校验,幂等。"""
    rel, want_sha, size = item["path"], item["sha256"], item["size"]
    name = os.path.basename(rel)
    dest = os.path.join(dest_dir, name)
    if os.path.exists(dest) and os.path.getsize(dest) == size and sha256_file(dest) == want_sha:
        log("  = %s 已存在且校验通过,跳过" % name)
        return dest

    url = BASE + "/" + rel
    tmp = dest + ".part"
    tries = 0
    while True:
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if have >= size:
            break
        tries += 1
        if tries > MAX_TRIES:
            raise SystemExit(
                "  ✗ %s: 重试 %d 次仍未完成(已收 %d/%d 字节)。"
                "已下载部分已保留,稍等几分钟重跑本脚本即可自动续传。" % (name, MAX_TRIES, have, size))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "asr-setup/1.0"})
            if have:
                req.add_header("Range", "bytes=%d-" % have)
            with urllib.request.urlopen(req, timeout=120) as r:
                code = r.getcode()
                if have and code != 206:
                    log("    %s: 服务器未按 Range 续传(HTTP %d),重头下载" % (name, code))
                    have = 0
                with open(tmp, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
        except urllib.error.HTTPError as e:
            ra = e.headers.get("Retry-After") if e.headers else None
            wait = int(ra) if (ra or "").isdigit() else min(RETRY_CAP, 20 * tries)
            log("    %s: HTTP %d(限流/暂不可用),等 %ds 后重试(%d/%d)"
                % (name, e.code, wait, tries, MAX_TRIES))
            time.sleep(wait)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            wait = min(RETRY_CAP, 10 * tries)
            log("    %s: 传输中断(%s: %s),已收 %d/%d 字节,等 %ds 后断点续传(%d/%d)"
                % (name, type(e).__name__, e, got, size, wait, tries, MAX_TRIES))
            time.sleep(wait)

    if os.path.getsize(tmp) != size:
        os.remove(tmp)
        raise SystemExit("  ✗ %s: 文件大小不符,已删除,请重跑脚本" % name)
    if sha256_file(tmp) != want_sha:
        os.remove(tmp)
        raise SystemExit("  ✗ %s: SHA256 校验失败,已删除,请重跑脚本" % name)
    os.replace(tmp, dest)
    log("  ✓ %s 下载完成,SHA256 校验通过(%.1f MB)" % (name, size / 1e6))
    return dest


def assemble_large(dest_dir):
    out = os.path.join(dest_dir, "model.int8.onnx")
    if (os.path.exists(out) and os.path.getsize(out) == LARGE_ASSEMBLED["size"]
            and sha256_file(out) == LARGE_ASSEMBLED["sha256"]):
        log("  = 大模型已拼接且校验通过,跳过")
        return out
    parts = [fetch(p, dest_dir) for p in LARGE_PARTS]
    log("  拼接 3 个分块 ...")
    with open(out + ".tmp", "wb") as w:
        for p in parts:
            with open(p, "rb") as r:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    w.write(chunk)
    if sha256_file(out + ".tmp") != LARGE_ASSEMBLED["sha256"]:
        os.remove(out + ".tmp")
        raise SystemExit("  ✗ 拼接后的大模型 SHA256 校验失败,请重跑脚本")
    os.replace(out + ".tmp", out)
    for p in parts:
        os.remove(p)
    log("  ✓ 大模型拼接完成,SHA256 校验通过(243.4 MB),分块已清理")
    return out


def ensure_deps():
    try:
        import sherpa_onnx  # noqa: F401
        import numpy  # noqa: F401
        log("  = sherpa-onnx / numpy 已安装,跳过")
        return
    except ImportError:
        pass
    log("  安装 sherpa-onnx + numpy(走 PyPI)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "sherpa-onnx", "numpy"])
        log("  ✓ 依赖安装完成")
    except subprocess.CalledProcessError:
        raise SystemExit("  ✗ pip 安装失败 — 检查 files.pythonhosted.org 是否被放行"
                         "(pip 下载 wheel 走这个域名,不是 pypi.org)")


def smoke_test(model_dir, wav_path, label):
    import wave
    import numpy as np
    import sherpa_onnx
    rec = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=os.path.join(model_dir, "model.int8.onnx"),
        tokens=os.path.join(model_dir, "tokens.txt"),
        num_threads=2,
        decoding_method="greedy_search",
    )
    with wave.open(wav_path) as f:
        sr = f.getframerate()
        samples = np.frombuffer(f.readframes(f.getnframes()),
                                dtype=np.int16).astype(np.float32) / 32768.0
    s = rec.create_stream()
    s.accept_waveform(sr, samples)
    rec.decode_stream(s)
    text = s.result.text
    ok = "tuesday" in text.lower() and "星期" in text
    log("  %s 识别结果: %s" % (label, text))
    if not ok:
        raise SystemExit("  ✗ 识别结果与预期不符(预期含 'tuesday' 和 '星期'),请检查")
    log("  ✓ %s 中英夹杂识别正常" % label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true", help="同时下载 243MB 大模型")
    ap.add_argument("--dir", default="./asr-models", help="安装目录")
    ap.add_argument("--skip-test", action="store_true", help="跳过识别自测")
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.dir))
    small_dir = os.path.join(root, "paraformer-zh-small")
    large_dir = os.path.join(root, "paraformer-zh-large")
    vad_dir = os.path.join(root, "vad")
    for d in (small_dir, vad_dir) + ((large_dir,) if args.large else ()):
        os.makedirs(d, exist_ok=True)

    log("[1/4] 下载小模型(82MB)+ VAD + 测试音频 → %s" % root)
    for item in SMALL:
        fetch(item, small_dir)
    fetch(VAD, vad_dir)
    wav = fetch(TEST_WAV, root)

    if args.large:
        log("[2/4] 下载大模型(243MB,3 个分块)")
        assemble_large(large_dir)
        fetch(LARGE_TOKENS, large_dir)
    else:
        log("[2/4] 跳过大模型(需要时加 --large 重跑)")

    log("[3/4] 检查/安装依赖")
    ensure_deps()

    if args.skip_test:
        log("[4/4] 跳过识别自测")
    else:
        log("[4/4] 识别自测(中英夹杂测试音频)")
        smoke_test(small_dir, wav, "小模型")
        if args.large:
            smoke_test(large_dir, wav, "大模型")

    log("")
    log("全部完成 ✓  模型目录:")
    log("  小模型: %s" % small_dir)
    if args.large:
        log("  大模型: %s" % large_dir)
    log("  VAD:    %s" % os.path.join(vad_dir, "silero_vad.onnx"))
    log("")
    log("用法示例:")
    log("  import sherpa_onnx")
    log("  rec = sherpa_onnx.OfflineRecognizer.from_paraformer(")
    log("      paraformer=r'%s/model.int8.onnx'," % (large_dir if args.large else small_dir))
    log("      tokens=r'%s/tokens.txt', num_threads=2)" % (large_dir if args.large else small_dir))


if __name__ == "__main__":
    main()
