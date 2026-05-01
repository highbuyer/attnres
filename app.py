#!/usr/bin/env python3
"""训练可视化 Web 控制台。启动: python app.py → http://localhost:8080"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread

import paramiko

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
DATA_DIR = ROOT / "datasets"
SCRIPTS_DIR = ROOT / "scripts"

# -------- 训练配置 --------
PROFILES_FILE = ROOT / "profiles.json"
BUILTIN_PROFILES = {
    "local": {"name": "本地 4090", "host": None, "port": None, "user": None, "password": None,
              "work_dir": str(ROOT), "python": sys.executable},
    "remote": {"name": "远程服务器", "host": "", "port": 22, "user": "root", "password": "",
               "work_dir": "/root/autodl-tmp/attnres", "python": "/root/miniconda3/bin/python"},
}

def _load_profiles():
    if PROFILES_FILE.exists():
        try:
            saved = json.loads(PROFILES_FILE.read_text())
            merged = dict(BUILTIN_PROFILES)
            for k, v in saved.items():
                merged[k] = {**merged.get(k, {}), **v}
            return merged
        except Exception:
            pass
    return dict(BUILTIN_PROFILES)

PROFILES = _load_profiles()
active_profile = "local"

# -------- 全局状态 --------
proc: subprocess.Popen | None = None
log_queue: Queue = Queue()
loss_queue: Queue = Queue()
job_status = "idle"
job_type = ""

# -------- SSH 工具 --------
def _ssh():
    p = PROFILES[active_profile]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(p["host"], port=p["port"], username=p["user"], password=p["password"], timeout=10)
    return c

def _remote_run(cmd: str) -> str:
    c = _ssh()
    try:
        _, stdout, _ = c.exec_command(cmd)
        return stdout.read().decode().strip()
    finally:
        c.close()

def _remote_popen(cmd: str) -> subprocess.Popen:
    """通过 SSH 启动远程进程，返回本地 Popen 包装"""
    p = PROFILES[active_profile]
    ssh_cmd = [
        "sshpass", "-p", p["password"],
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-p", str(p["port"]), f"{p['user']}@{p['host']}",
        f"cd {p['work_dir']} && {cmd}"
    ]
    return subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

# -------- FastAPI --------
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

app = FastAPI(title="Training Console")

# -------- 工具函数 --------
def get_checkpoints():
    p = PROFILES[active_profile]
    if p["host"]:
        try:
            out = _remote_run(f"ls -1 {p['work_dir']}/checkpoints/*.pt 2>/dev/null || echo ''")
            if not out: return []
            return [{"name": line.split("/")[-1], "size_mb": 0} for line in out.splitlines() if line]
        except Exception:
            return []
    pts = sorted(CHECKPOINT_DIR.glob("*.pt"))
    return [{"name": p.name, "size_mb": round(p.stat().st_size / 1024 / 1024, 1)} for p in pts]

def get_datasets():
    p = PROFILES[active_profile]
    if p["host"]:
        try:
            out = _remote_run(f"find {p['work_dir']}/datasets -name '*.jsonl' -type f 2>/dev/null | sort || echo ''")
            if not out: return []
            results = []
            wd = p["work_dir"]
            for line in out.splitlines():
                if not line: continue
                rel = line.replace(wd + "/", "")
                name = line.split("/")[-1]
                results.append({"path": rel, "name": name, "size_mb": 0})
            return results
        except Exception:
            return []
    files = []
    for p_ in sorted(DATA_DIR.rglob("*.jsonl")):
        if p_.is_file():
            files.append({
                "path": str(p_.relative_to(ROOT)),
                "name": p_.name,
                "size_mb": round(p_.stat().st_size / 1024 / 1024, 1)
            })
    return files

def get_gpu_info():
    p = PROFILES[active_profile]
    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
           "--format=csv,noheader,nounits"]
    try:
        if p["host"]:
            out = _remote_run(" ".join(cmd))
        else:
            out = subprocess.check_output(cmd, timeout=5, text=True).strip()
        parts = [x.strip() for x in out.split(",")]
        return {
            "gpu_util": float(parts[0]) if len(parts) > 0 else 0,
            "mem_used_mb": float(parts[1]) if len(parts) > 1 else 0,
            "mem_total_mb": float(parts[2]) if len(parts) > 2 else 0,
            "temp": float(parts[3]) if len(parts) > 3 else 0,
            "power_w": float(parts[4]) if len(parts) > 4 else 0,
        }
    except Exception as e:
        return {"gpu_util": 0, "mem_used_mb": 0, "mem_total_mb": 0, "temp": 0, "power_w": 0, "error": str(e)[:100]}

LOSS_RE = re.compile(r"step\s+(\d+)/\d+\s+\|\s+loss\s+([\d.]+)")

def run_subprocess(cmd: list[str]):
    global proc, job_status
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, preexec_fn=os.setsid,
        )
        job_status = "running"
        for line in proc.stdout:
            line = line.rstrip()
            log_queue.put(line)
            m = LOSS_RE.search(line)
            if m:
                loss_queue.put({"step": int(m.group(1)), "loss": float(m.group(2))})
        proc.wait()
    except Exception as e:
        log_queue.put(f"[app] error: {e}")
    finally:
        job_status = "stopped"
        proc = None

def remote_train_runner(remote_cmd: str):
    global proc, job_status
    p = PROFILES[active_profile]
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(p["host"], port=p["port"], username=p["user"], password=p["password"], timeout=10)
        chan = c.get_transport().open_session()
        chan.exec_command(f"cd {p['work_dir']} && {remote_cmd}")
        job_status = "running"
        for line in iter(chan.recv, 1024):
            text = line.decode("utf-8", errors="replace")
            for txt_line in text.splitlines():
                txt_line = txt_line.rstrip()
                log_queue.put(txt_line)
                m = LOSS_RE.search(txt_line)
                if m:
                    loss_queue.put({"step": int(m.group(1)), "loss": float(m.group(2))})
        chan.recv_exit_status()
    except Exception as e:
        log_queue.put(f"[remote] error: {e}")
    finally:
        job_status = "stopped"
        proc = None

def stop_proc():
    global proc, job_status
    if PROFILES[active_profile]["host"]:
        # Remote: kill sft_v2.py on server
        try:
            _remote_run("pkill -f sft_v2.py 2>/dev/null; echo done")
        except Exception:
            pass
    elif proc is not None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()
    job_status = "stopped"
    log_queue.put("[app] 已手动停止")

# -------- API --------
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(PAGE_HTML)

@app.get("/api/status")
async def status():
    return {
        "job_status": job_status, "job_type": job_type,
        "active_profile": active_profile,
        "profiles": {k: {"name": v["name"]} for k, v in PROFILES.items()},
        "gpu": get_gpu_info(),
        "checkpoints": get_checkpoints(),
        "datasets": get_datasets(),
    }

@app.post("/api/profile/switch")
async def switch_profile(req: Request):
    global active_profile, job_status
    if job_status == "running":
        raise HTTPException(400, "请先停止当前训练")
    data = await req.json()
    name = data.get("profile", "local")
    if name not in PROFILES:
        raise HTTPException(400, f"未知配置: {name}")
    active_profile = name
    return {"ok": True, "profile": name}

@app.get("/api/profiles")
async def get_profiles():
    result = {}
    for k, v in PROFILES.items():
        p = dict(v)
        if p.get("password"):
            p["password"] = "***"  # mask in list view
        result[k] = p
    return result

@app.post("/api/profile/save")
async def save_profile(req: Request):
    global PROFILES
    data = await req.json()
    name = data.get("name", "")
    if name not in PROFILES:
        raise HTTPException(400, f"未知配置: {name}")
    updates = {k: v for k, v in data.items() if k in ("host", "port", "user", "password", "work_dir", "python")}
    if updates.get("port"):
        updates["port"] = int(updates["port"])
    PROFILES[name].update(updates)
    # Persist
    saved = {}
    if PROFILES_FILE.exists():
        try: saved = json.loads(PROFILES_FILE.read_text())
        except Exception: pass
    saved[name] = {k: v for k, v in PROFILES[name].items()
                   if k in ("name", "host", "port", "user", "password", "work_dir", "python")}
    PROFILES_FILE.write_text(json.dumps(saved, ensure_ascii=False, indent=2))
    return {"ok": True}

@app.post("/api/train/start")
async def start_train(req: Request):
    global job_status, job_type
    if job_status == "running":
        raise HTTPException(400, "已有任务在运行")
    data = await req.json()
    cmd = [
        sys.executable, "-u", str(ROOT / "src/sft_v2.py"),
        "--ckpt", data.get("ckpt", "checkpoints/d36_v2_mla_best_slim.pt"),
        "--data", data.get("data", "datasets/sft_archive/sft_mixed_v8_50k.jsonl"),
        "--data-claude", data.get("data_claude", "datasets/sft_claude_trajectories.jsonl"),
        "--mix-ratio-claude", str(data.get("mix_ratio", 0.7)),
        "--max-samples", str(data.get("max_samples", 50000)),
        "--steps", str(data.get("steps", 500)),
        "--bsz", str(data.get("bsz", 6)),
        "--grad-accum", str(data.get("grad_accum", 12)),
        "--lr", str(data.get("lr", 2e-5)),
        "--log-every", str(data.get("log_every", 25)),
        "--save-every", str(data.get("save_every", 100)),
        "--out", data.get("out", f"checkpoints/sft_{int(time.time())}.pt"),
    ]
    job_type = "train"
    if PROFILES[active_profile]["host"]:
        p = PROFILES[active_profile]
        # Replace local python/exe with remote paths
        remote_cmd = f"{p['python']} -u src/sft_v2.py " + " ".join(
            f"{shlex.quote(str(c))}" for c in cmd[3:]  # skip local python -u path
        )
        log_queue.put(f"[remote] {remote_cmd}")
        Thread(target=remote_train_runner, args=(remote_cmd,), daemon=True).start()
    else:
        Thread(target=run_subprocess, args=(cmd,), daemon=True).start()
    return {"ok": True, "cmd": " ".join(cmd)}

@app.post("/api/train/stop")
async def stop_train():
    global job_type
    if job_status != "running":
        raise HTTPException(400, "没有在运行的任务")
    stop_proc()
    return {"ok": True}

@app.post("/api/script/run")
async def run_script(req: Request):
    global job_status, job_type
    if job_status == "running":
        raise HTTPException(400, "已有任务在运行")
    data = await req.json()
    script = data.get("script", "")
    args = data.get("args", {})
    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        raise HTTPException(404, f"脚本不存在: {script}")
    cmd = [sys.executable, "-u", str(script_path)]
    for k, v in args.items():
        if v is None or v == "":
            continue
        kebab = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                cmd.append(kebab)
        else:
            cmd.extend([kebab, str(v)])
    job_type = "script"
    Thread(target=run_subprocess, args=(cmd,), daemon=True).start()
    return {"ok": True, "cmd": " ".join(shlex.quote(c) for c in cmd)}

@app.get("/api/script/templates")
async def script_templates():
    return {
        "scripts": [
            {
                "id": "build_traj",
                "title": "提取 Claude 轨迹",
                "script": "build_claude_traj_sft.py",
                "desc": "从 ~/.claude/projects/ 原始会话日志提取训练数据",
                "args": [
                    {"key": "root", "label": "会话目录", "default": os.path.expanduser("~/.claude/projects")},
                    {"key": "out", "label": "输出文件", "default": "datasets/sft_claude_trajectories.jsonl"},
                    {"key": "min_tool_use", "label": "最少工具调用次数", "default": "1"},
                    {"key": "max_tool_result_chars", "label": "工具结果截断长度", "default": "4000"},
                ]
            },
            {
                "id": "synth",
                "title": "生成合成数据 v2.1",
                "script": "synthesize_cc_trajectories_v2_1.py",
                "desc": "从代码库 grep 自动生成工具调用轨迹",
                "args": [
                    {"key": "out", "label": "输出文件", "default": "data/cc_synth_v2_1.jsonl"},
                    {"key": "seed", "label": "随机种子", "default": "42"},
                    {"key": "n_const", "label": "常量查找样本数", "default": "200"},
                    {"key": "n_func", "label": "函数定位样本数", "default": "200"},
                    {"key": "n_arg", "label": "argparse样本数", "default": "150"},
                    {"key": "n_cuse", "label": "常量用法样本数", "default": "200"},
                    {"key": "n_fuse", "label": "函数用法样本数", "default": "150"},
                ]
            },
            {
                "id": "gold",
                "title": "提取 Gold 轨迹",
                "script": "extract_cc_gold_v2.py",
                "desc": "从 attnres 会话提取 Read/Grep 轨迹并验证结果正确性",
                "args": [
                    {"key": "write", "label": "写入模式", "default": "1", "type": "bool"},
                    {"key": "limit", "label": "数量上限(空=不限)", "default": ""},
                ]
            },
            {
                "id": "merge",
                "title": "合并数据集",
                "script": "merge_sft_v5_cc.py",
                "desc": "合并 base + gold + synth 数据并打乱",
                "args": [
                    {"key": "base", "label": "基础数据", "default": "data/sft_tool_summary_v5_clean.jsonl"},
                    {"key": "gold", "label": "Gold轨迹", "default": "data/cc_gold_trajectories_v1.jsonl"},
                    {"key": "synth", "label": "合成数据", "default": "data/cc_synth_v2.jsonl"},
                    {"key": "out", "label": "输出文件", "default": "data/sft_v5_clean_cc.jsonl"},
                    {"key": "cc_oversample", "label": "过采样倍数", "default": "1"},
                ]
            },
            {
                "id": "convert",
                "title": "格式转换 (inline → Qwen)",
                "script": "convert_inline_to_qwen.py",
                "desc": "老格式内联标记转 Qwen tool_calls 结构",
                "args": [
                    {"key": "in", "label": "输入文件", "default": ""},
                    {"key": "out", "label": "输出文件", "default": ""},
                ]
            },
            {
                "id": "verify",
                "title": "数据质量验证",
                "script": "verify_training_data.py",
                "desc": "统计样本分布、工具调用比例、数据一致性",
                "args": []
            },
        ]
    }

@app.get("/api/train/log/stream")
async def log_stream():
    async def generate():
        while True:
            items = []
            while not log_queue.empty():
                try: items.append(log_queue.get_nowait())
                except Exception: break
            if items:
                for line in items:
                    yield f"data: {json.dumps({'text': line})}\n\n"
            loss_items = []
            while not loss_queue.empty():
                try: loss_items.append(loss_queue.get_nowait())
                except Exception: break
            if loss_items:
                yield f"data: {json.dumps({'losses': loss_items})}\n\n"
            if job_status == "stopped" and proc is None and log_queue.empty():
                yield f"data: {json.dumps({'text': '[app] 任务结束', 'done': True})}\n\n"
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(generate(), media_type="text/event-stream")

# -------- HTML --------
PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8"><title>Training Console</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,monospace;background:#1a1a2e;color:#e0e0e0;height:100vh;display:flex;flex-direction:column;overflow:hidden}
/* ---- tabs ---- */
#tabs{display:flex;flex-shrink:0;background:#111;border-bottom:1px solid #333}
#tabs button{flex:1;padding:12px;background:none;border:none;color:#888;font-size:14px;cursor:pointer;border-bottom:2px solid transparent}
#tabs button.active{color:#00d4ff;border-bottom-color:#00d4ff}
/* ---- global gpu ---- */
#gpu-bar{display:flex;flex-wrap:wrap;gap:8px;padding:8px 12px;flex-shrink:0;border-bottom:1px solid #222}
.gpu-card{flex:1;min-width:70px;background:#16213e;border-radius:6px;padding:6px 8px;text-align:center}
.gpu-card .val{font-size:16px;font-weight:bold;color:#00d4ff;overflow:hidden;text-overflow:ellipsis}
.gpu-card .lbl{font-size:10px;color:#888;margin-top:1px}
/* ---- tab content ---- */
.tab-content{display:none;flex:1;overflow:hidden;min-height:0}
.tab-content.active{display:flex}
/* ---- train tab ---- */
#train-tab{min-height:0}
#left{width:300px;min-width:300px;flex-shrink:0;padding:12px;border-right:1px solid #333;overflow-y:auto;display:flex;flex-direction:column;gap:6px}
#train-right{flex:1;min-width:0;display:flex;flex-direction:column;padding:12px;overflow:hidden}
#log-box{flex:1;background:#0d1117;border-radius:6px;padding:10px;overflow-y:auto;overflow-x:hidden;font-family:"Menlo","Consolas",monospace;font-size:11px;line-height:1.5;min-height:0;white-space:pre-wrap;word-break:break-all;color:#c8d6e5}
#chart-box{height:220px;flex-shrink:0;background:#16213e;border-radius:6px;padding:10px;margin-top:6px;min-height:0;overflow:hidden}
canvas{max-width:100%;max-height:100%}
/* ---- data tab ---- */
#data-tab{flex-direction:column;overflow-y:auto;gap:0}
#data-tab.active{display:flex}
#data-header{padding:12px 16px 8px;flex-shrink:0;display:flex;align-items:center;justify-content:space-between}
#data-cards{flex:1;overflow-y:auto;padding:0 16px 8px;display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}
.script-card{background:#16213e;border-radius:8px;padding:14px;border:1px solid #333;overflow:hidden;display:flex;flex-direction:column}
.script-card h3{font-size:14px;margin-bottom:4px;color:#00d4ff;overflow:hidden;text-overflow:ellipsis}
.script-card .desc{font-size:11px;color:#888;margin-bottom:8px;word-break:break-all}
.script-card .arg-row{display:flex;gap:6px;align-items:center;margin-bottom:4px}
.script-card .arg-row label{font-size:10px;color:#aaa;min-width:85px;text-align:right;flex-shrink:0}
.script-card .arg-row input{flex:1;min-width:0;padding:3px 6px;background:#0d1117;border:1px solid #333;border-radius:4px;color:#e0e0e0;font-size:11px}
.script-card .btn-script{margin-top:auto;align-self:flex-start}
#data-log{height:180px;flex-shrink:0;background:#0d1117;border-top:1px solid #333;padding:8px 16px;overflow-y:auto;font-family:"Menlo","Consolas",monospace;font-size:11px;line-height:1.4;color:#c8d6e5;white-space:pre-wrap;word-break:break-all}
/* ---- shared ---- */
.status{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.status.idle{background:#555}.status.running{background:#00ff88;animation:pulse 1s infinite}.status.stopped{background:#ff5555}
@keyframes pulse{50%{opacity:.3}}
label{font-size:11px;color:#aaa}
input,select{width:100%;padding:5px 7px;background:#16213e;border:1px solid #333;border-radius:4px;color:#e0e0e0;font-size:12px;margin-top:2px;transition:border-color .2s}
input:focus,select:focus{outline:none;border-color:#00d4ff;box-shadow:0 0 0 2px rgba(0,212,255,.15)}
select{background:#0d1117;cursor:pointer;max-width:100%}
option{background:#16213e;color:#e0e0e0}
.row{display:flex;gap:8px}.row>*{flex:1}
.btn{padding:6px 12px;border:none;border-radius:4px;font-size:12px;cursor:pointer;font-weight:bold}
.btn-start{background:#00c853;color:#000}
.btn-stop{background:#ff1744;color:#fff}
.btn-script{background:#2962ff;color:#fff}
.btn:disabled{opacity:.4;cursor:default}
.btn-row{display:flex;gap:8px;margin-top:6px}
#top-bar{display:flex;align-items:center;justify-content:space-between;margin-bottom:4px}
#top-bar .status-line{display:flex;align-items:center;font-size:12px}
</style>
</head>
<body>

<div id="tabs">
  <button class="active" onclick="switchTab('train')">训练</button>
  <button onclick="switchTab('data')">数据处理</button>
</div>

<!-- GPU 全局置顶 -->
<div id="gpu-bar">
  <div class="gpu-card"><div class="val" id="gpu-util">--</div><div class="lbl">GPU %</div></div>
  <div class="gpu-card"><div class="val" id="gpu-mem">--</div><div class="lbl">显存 GB</div></div>
  <div class="gpu-card"><div class="val" id="gpu-temp">--</div><div class="lbl">温度 °C</div></div>
  <div class="gpu-card"><div class="val" id="gpu-power">--</div><div class="lbl">功率 W</div></div>
  <div class="gpu-card" style="display:flex;align-items:center;justify-content:center;gap:4px">
    <span id="status-dot" class="status idle"></span><span id="status-text" style="font-size:11px">空闲</span>
  </div>
</div>

<!-- ========== 训练 Tab ========== -->
<div id="train-tab" class="tab-content active">
  <div id="left">
    <div id="top-bar"><h2 style="font-size:16px">Training</h2></div>
    <div style="display:flex;gap:6px;align-items:flex-end">
      <div style="flex:1"><label for="sel-profile">训练位置</label>
      <select id="sel-profile" onchange="switchProfile(this.value)">
        <option value="local">本地 4090</option>
        <option value="remote">远程服务器</option>
      </select></div>
      <button class="btn btn-script" onclick="showProfileEditor()" style="white-space:nowrap;margin-top:14px">编辑</button>
    </div>
    <label for="sel-ckpt">Checkpoint</label><select id="sel-ckpt"></select>
    <label for="sel-data">训练数据</label><select id="sel-data"></select>
    <label for="sel-claude">Claude数据</label><select id="sel-claude"></select>
    <div class="row"><div><label for="inp-steps">Steps</label><input id="inp-steps" value="500"></div><div><label for="inp-bsz">batch size</label><input id="inp-bsz" value="6"></div></div>
    <div class="row"><div><label for="inp-ga">grad accum</label><input id="inp-ga" value="12"></div><div><label for="inp-lr">lr</label><input id="inp-lr" value="2e-5"></div></div>
    <div class="row"><div><label for="inp-mix">mix_claude</label><input id="inp-mix" value="0.7"></div><div><label for="inp-save">save_every</label><input id="inp-save" value="100"></div></div>
    <div class="row"><div><label for="inp-maxs">max_samples</label><input id="inp-maxs" value="50000"></div><div><label for="inp-log">log_every</label><input id="inp-log" value="25"></div></div>
    <label for="inp-out">输出文件</label><input id="inp-out" value="checkpoints/sft_test.pt">
    <div class="btn-row">
      <button class="btn btn-start" id="btn-start" onclick="startTrain()">启动训练</button>
      <button class="btn btn-stop" id="btn-stop" onclick="stopJob()">停止</button>
    </div>
  </div>
  <div id="train-right">
    <div id="log-box"></div>
    <div id="chart-box"><canvas id="loss-chart"></canvas></div>
  </div>
</div>

<!-- ========== 数据处理 Tab ========== -->
<div id="data-tab" class="tab-content">
  <div id="data-header">
    <h2 style="font-size:16px">Data Processing</h2>
    <button class="btn btn-stop" onclick="stopJob()">停止当前任务</button>
  </div>
  <div id="data-cards">加载中...</div>
  <div id="data-log"></div>
</div>

<script>
const trainLog = document.getElementById('log-box');
const dataLog = document.getElementById('data-log');
let lossData = [];
let activeTab = 'train';
function activeLog() { return activeTab === 'train' ? trainLog : dataLog; }

const chart = new Chart(document.getElementById('loss-chart'), {
  type: 'line',
  data: {labels: [], datasets: [{label: 'loss', data: [], borderColor: '#00d4ff', borderWidth: 1.5, pointRadius: 0, tension: 0.3}]},
  options: {
    responsive: true, maintainAspectRatio: false,
    scales: {x: {ticks: {color: '#888'}, grid: {color: '#222'}}, y: {ticks: {color: '#888'}, grid: {color: '#222'}}},
    plugins: {legend: {labels: {color: '#888'}}}
  }
});

let autoScroll = true;
trainLog.addEventListener('scroll', function() {
  autoScroll = (this.scrollTop + this.clientHeight + 20) >= this.scrollHeight;
});

function logLine(text) {
  const box = activeLog();
  const div = document.createElement('div');
  div.textContent = text;
  box.appendChild(div);
  if (activeTab === 'train') {
    if (autoScroll) box.scrollTop = box.scrollHeight;
    while (box.children.length > 500) box.firstChild.remove();
  } else {
    box.scrollTop = box.scrollHeight;
    while (box.children.length > 200) box.firstChild.remove();
  }
}

function addLoss(step, loss) {
  lossData.push({step, loss});
  chart.data.labels.push(step);
  chart.data.datasets[0].data.push(loss);
  if (chart.data.labels.length > 200) { chart.data.labels.shift(); chart.data.datasets[0].data.shift(); }
  chart.update();
}

function clearChart() { lossData=[]; chart.data.labels=[]; chart.data.datasets[0].data=[]; chart.update(); }

async function loadStatus() {
  const r = await fetch('/api/status'); const s = await r.json(); const g = s.gpu;
  document.getElementById('gpu-util').textContent = g.gpu_util.toFixed(0)+'%';
  document.getElementById('gpu-mem').textContent = (g.mem_used_mb/1024).toFixed(1)+'G';
  document.getElementById('gpu-temp').textContent = g.temp.toFixed(0)+'°';
  document.getElementById('gpu-power').textContent = g.power_w.toFixed(0)+'W';
  const dot = document.getElementById('status-dot');
  dot.className = 'status ' + s.job_status;
  document.getElementById('status-text').textContent = s.job_status==='running'?'运行中':s.job_status==='stopped'?'已停止':'空闲';
  document.getElementById('btn-start').disabled = s.job_status === 'running';
  // Update profile selector
  const profSel = document.getElementById('sel-profile');
  if (s.active_profile && profSel.value !== s.active_profile) {
    profSel.value = s.active_profile;
    currentProfile = s.active_profile;
  }
  const ckpt = document.getElementById('sel-ckpt');
  if (!ckpt.dataset.filled) {
    s.checkpoints.forEach(c => ckpt.add(new Option(c.name+' ('+c.size_mb+'MB)', c.name)));
    s.datasets.forEach(d => {
      const opt = new Option(d.path, d.path);
      document.getElementById('sel-data').add(opt);
      document.getElementById('sel-claude').add(opt.cloneNode(true));
    });
    ckpt.value='d36_v2_mla_best_slim.pt';
    document.getElementById('sel-data').value='datasets/sft_archive/sft_mixed_v8_50k.jsonl';
    document.getElementById('sel-claude').value='datasets/sft_claude_trajectories.jsonl';
    ckpt.dataset.filled='1';
  }
}

let currentProfile = 'local';

async function switchProfile(name) {
  if (currentProfile === name) return;
  await fetch('/api/profile/switch', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({profile: name})});
  currentProfile = name;
  // Refresh checkpoints/datasets dropdowns
  const ckpt = document.getElementById('sel-ckpt');
  const selData = document.getElementById('sel-data');
  const selClaude = document.getElementById('sel-claude');
  ckpt.innerHTML = ''; selData.innerHTML = ''; selClaude.innerHTML = '';
  ckpt.dataset.filled = '';
  await loadStatus();
}

async function showProfileEditor() {
  const r = await fetch('/api/profiles');
  const profiles = await r.json();
  const p = profiles[currentProfile] || {};
  document.getElementById('edit-host').value = p.host || '';
  document.getElementById('edit-port').value = p.port || 22;
  document.getElementById('edit-user').value = p.user || 'root';
  document.getElementById('edit-pass').value = p.password || '';
  document.getElementById('edit-dir').value = p.work_dir || '';
  document.getElementById('edit-py').value = p.python || '';
  document.getElementById('profile-modal').style.display = 'flex';
}

async function saveProfile() {
  const data = {
    name: currentProfile,
    host: document.getElementById('edit-host').value,
    port: document.getElementById('edit-port').value,
    user: document.getElementById('edit-user').value,
    password: document.getElementById('edit-pass').value,
    work_dir: document.getElementById('edit-dir').value,
    python: document.getElementById('edit-py').value,
  };
  await fetch('/api/profile/save', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  document.getElementById('profile-modal').style.display = 'none';
  // Refresh dropdowns
  const ckpt = document.getElementById('sel-ckpt');
  ckpt.innerHTML = ''; document.getElementById('sel-data').innerHTML = ''; document.getElementById('sel-claude').innerHTML = '';
  ckpt.dataset.filled = '';
  await loadStatus();
}

async function startTrain() {
  const body = {
    ckpt: (currentProfile === 'remote-a800' ? '' : 'checkpoints/') + document.getElementById('sel-ckpt').value,
    data: document.getElementById('sel-data').value,
    data_claude: document.getElementById('sel-claude').value,
    steps: document.getElementById('inp-steps').value,
    bsz: document.getElementById('inp-bsz').value,
    grad_accum: document.getElementById('inp-ga').value,
    lr: document.getElementById('inp-lr').value,
    mix_ratio: document.getElementById('inp-mix').value,
    save_every: document.getElementById('inp-save').value,
    log_every: document.getElementById('inp-log').value,
    max_samples: document.getElementById('inp-maxs').value,
    out: document.getElementById('inp-out').value,
  };
  clearChart();
  await fetch('/api/train/start', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
}

async function stopJob() { await fetch('/api/train/stop', {method:'POST'}); }

// ---- Tabs ----
function switchTab(name) {
  activeTab = name;
  const fn = () => {
    document.querySelectorAll('#tabs button').forEach((b,i) => {
      b.classList.toggle('active', (i===0 && name==='train') || (i===1 && name==='data'));
    });
    document.getElementById('train-tab').classList.toggle('active', name==='train');
    document.getElementById('data-tab').classList.toggle('active', name==='data');
  };
  if (document.startViewTransition) {
    document.startViewTransition(fn);
  } else {
    fn();
  }
}

// ---- Script Cards ----
async function loadScripts() {
  const r = await fetch('/api/script/templates');
  const {scripts} = await r.json();
  const container = document.getElementById('data-cards');
  container.innerHTML = '';
  scripts.forEach(s => {
    const card = document.createElement('div');
    card.className = 'script-card';
    let argsHtml = '';
    const argKeys = s.args.map(a => a.key);
    s.args.forEach(a => {
      if (a.type === 'bool') {
        argsHtml += `<div class="arg-row"><label for="arg_${s.id}_${a.key}">${a.label}</label><input type="checkbox" id="arg_${s.id}_${a.key}" ${a.default==='1'?'checked':''}></div>`;
      } else {
        argsHtml += `<div class="arg-row"><label for="arg_${s.id}_${a.key}">${a.label}</label><input id="arg_${s.id}_${a.key}" value="${a.default||''}" placeholder="可选"></div>`;
      }
    });
    card.innerHTML = `
      <h3>${s.title}</h3>
      <div class="desc">${s.desc}<br><code>${s.script}</code></div>
      ${argsHtml}
      <button class="btn btn-script" style="margin-top:6px">运行</button>
    `;
    const btn = card.querySelector('.btn-script');
    btn.addEventListener('click', () => {
      const args = {};
      argKeys.forEach(k => {
        const el = document.getElementById('arg_'+s.id+'_'+k);
        if (!el) return;
        if (el.type === 'checkbox') args[k] = el.checked;
        else args[k] = el.value;
      });
      switchTab('data');
      dataLog.innerHTML = '';
      fetch('/api/script/run', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({script: s.script, args})});
    });
    container.appendChild(card);
  });
}

// ---- SSE ----
const es = new EventSource('/api/train/log/stream');
es.onmessage = (e) => {
  const d = JSON.parse(e.data);
  if (d.text) logLine(d.text);
  if (d.losses) d.losses.forEach(l => addLoss(l.step, l.loss));
  if (d.done) es.close();
};

setInterval(loadStatus, 2000);
loadStatus();
loadScripts();
</script>

<!-- ========== Profile Editor Modal ========== -->
<div id="profile-modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.7);z-index:99;align-items:center;justify-content:center">
<div style="background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:24px;width:420px;max-height:90vh;overflow-y:auto">
  <h3 style="margin-bottom:16px;color:#00d4ff">编辑服务器配置</h3>
  <label for="edit-host">主机地址</label><input id="edit-host" placeholder="connect.example.com">
  <label for="edit-port">端口</label><input id="edit-port" value="22">
  <label for="edit-user">用户名</label><input id="edit-user" value="root">
  <label for="edit-pass">密码</label><input id="edit-pass" type="password" placeholder="输入密码">
  <label for="edit-dir">工作目录</label><input id="edit-dir" placeholder="/root/autodl-tmp/attnres">
  <label for="edit-py">Python路径</label><input id="edit-py" placeholder="/root/miniconda3/bin/python">
  <div class="btn-row" style="margin-top:16px">
    <button class="btn btn-start" onclick="saveProfile()">保存</button>
    <button class="btn btn-stop" onclick="document.getElementById('profile-modal').style.display='none'">取消</button>
  </div>
</div></div>

</body></html>"""

if __name__ == "__main__":
    print(f"打开 http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
