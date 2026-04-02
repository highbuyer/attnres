#!/usr/bin/env python3
"""批量生成工具调用 SFT 训练样本。

生成三类样本：
1. search_code 工具调用（~200 条）
2. read_file 工具调用（~150 条）
3. 不需要工具的直接回答（~150 条）

输出：JSONL 文件，每行一个 {"messages": [...]} 样本。
"""
import json
import random

random.seed(42)

# ---------------------------------------------------------------------------
# 模板数据
# ---------------------------------------------------------------------------

# 通用的代码搜索场景（不限于 attnres 项目）
SEARCH_TEMPLATES = [
    # 查找函数定义
    ("函数 {func} 在哪里定义？", "{func}", "{file}:{line}     def {func}({args}):", "`{func}` 定义在 `{file}:{line}`。"),
    ("帮我找到 {func} 的实现。", "{func}", "{file}:{line}     def {func}({args}):", "`{func}` 的实现在 `{file}:{line}`。"),
    ("搜索一下 {func} 这个函数。", "{func}", "{file}:{line}     def {func}({args}):", "找到了，`{func}` 在 `{file}:{line}`。"),
    # 查找类定义
    ("类 {cls} 定义在哪？", "class {cls}", "{file}:{line} class {cls}:", "`{cls}` 类定义在 `{file}:{line}`。"),
    ("{cls} 这个类在哪个文件？", "class {cls}", "{file}:{line} class {cls}({base}):", "`{cls}` 在 `{file}:{line}`，继承自 `{base}`。"),
    # 查找变量/常量
    ("{var} 的值是多少？", "{var} =", "{file}:{line} {var} = {val}", "`{var}` 的值是 `{val}`，定义在 `{file}:{line}`。"),
    ("{var} 在哪里配置的？", "{var}", "{file}:{line} {var} = {val}", "`{var}` 配置在 `{file}:{line}`，当前值为 `{val}`。"),
    # 查找 import
    ("{module} 在哪里被 import？", "import {module}", "{file}:{line} import {module}", "`{module}` 在 `{file}:{line}` 被导入。"),
    ("项目里有没有用到 {module}？", "import {module}", "{file}:{line} from {module} import {item}", "有，`{file}:{line}` 导入了 `{module}` 的 `{item}`。"),
    # 查找字符串/注释
    ("代码里有没有提到 {keyword}？", "{keyword}", "{file}:{line} # {comment}", "有，在 `{file}:{line}` 的注释里提到了。"),
    ("搜索 {keyword} 相关的代码。", "{keyword}", "{file}:{line}     {code}", "找到了，在 `{file}:{line}`。"),
    # 查找配置参数
    ("{param} 参数默认值是什么？", "{param}", "{file}:{line}     {param}: {type} = {default}", "`{param}` 默认值是 `{default}`，类型 `{type}`，定义在 `{file}:{line}`。"),
    ("帮我查一下 {param} 的默认配置。", "{param} =", "{file}:{line} {param} = {default}", "`{param}` 默认是 `{default}`。"),
    # 查找错误处理
    ("代码里哪里抛出了 {exc}？", "raise {exc}", "{file}:{line}     raise {exc}(\"{msg}\")", "`{file}:{line}` 抛出了 `{exc}`，消息是 `{msg}`。"),
    # 多结果
    ("{func} 在哪些文件里出现了？", "{func}", "{file1}:{line1}     {func}(...)\n{file2}:{line2}     {func}(...)", "`{func}` 出现在 `{file1}:{line1}` 和 `{file2}:{line2}`。"),
]

READ_TEMPLATES = [
    # 读取文件开头
    ("请读取 {file} 的前 {n} 行。", "{file}", 1, "{n}", "{content}", "这是 `{file}` 的前 {n} 行。{summary}"),
    ("把 {file} 开头给我看一下。", "{file}", 1, "15", "{content}", "{summary}"),
    # 读取特定段落
    ("读取 {file} 里 {what} 附近的代码。", "{file}", "{offset}", "{limit}", "{content}", "{summary}"),
    ("给我看 {file} 第 {offset} 行附近。", "{file}", "{offset}", "{limit}", "{content}", "{summary}"),
    # 读取配置文件
    ("请读取 {file} 的配置部分。", "{file}", "{offset}", "{limit}", "{content}", "{summary}"),
    # 读取文档
    ("把 {file} 的内容给我看看。", "{file}", 1, "20", "{content}", "{summary}"),
]

DIRECT_ANSWER_TEMPLATES = [
    # 身份相关
    ("你是谁？", "我是微研，一个技术助手。"),
    ("你叫什么名字？", "我叫微研。"),
    ("你是 ChatGPT 吗？", "不是，我是微研。"),
    ("你是 Claude 吗？", "不是，我是微研，一个技术助手。"),
    ("你是谁开发的？", "我是微研，一个专注于技术问题的助手。"),
    ("你是 GPT-4 吗？", "不是，我是微研。"),
    ("你是人工智能吗？", "是的，我是微研，一个技术助手。"),
    ("你是真人吗？", "不是，我是微研，一个 AI 技术助手。"),
    ("请介绍一下你自己。", "我是微研，一个中文技术助手。我可以帮你搜索代码、阅读文件、解释技术概念。"),
    ("你有什么能力？", "我可以搜索代码、阅读文件、解释技术概念，帮你快速定位和理解代码。"),
    # 简单知识
    ("1+1 等于几？", "2。"),
    ("Python 的创始人是谁？", "Guido van Rossum。"),
    ("HTTP 200 表示什么？", "请求成功。"),
    ("Git 中 HEAD 是什么？", "HEAD 是当前分支的最新 commit 的指针。"),
    ("什么是 JSON？", "JSON 是一种轻量级的数据交换格式，使用键值对结构。"),
    ("TCP 和 UDP 的区别？", "TCP 面向连接、可靠传输；UDP 无连接、速度快但不保证可靠性。"),
    ("Python 里 list 和 tuple 的区别？", "`list` 可变，`tuple` 不可变。"),
    ("什么是递归？", "递归是函数调用自身的编程技巧，需要有终止条件。"),
    ("什么是梯度下降？", "梯度下降是一种优化算法，沿损失函数梯度反方向更新参数以最小化损失。"),
    ("什么是过拟合？", "模型在训练集上表现好但在新数据上泛化差，通常因模型过于复杂或训练数据不足。"),
    ("什么是 API？", "Application Programming Interface，应用程序之间交互的接口规范。"),
    ("Linux 里 chmod 755 是什么意思？", "文件所有者有读写执行权限，组用户和其他用户有读和执行权限。"),
    ("Docker 和虚拟机有什么区别？", "Docker 是容器化，共享宿主内核，轻量快速；虚拟机有独立内核，隔离更彻底但更重。"),
    ("什么是哈希表？", "一种通过哈希函数将键映射到桶的数据结构，平均查找时间 O(1)。"),
    ("HTTP GET 和 POST 的区别？", "GET 用于获取资源，参数在 URL 中；POST 用于提交数据，参数在请求体中。"),
    ("什么是 SQL 注入？", "攻击者通过在输入中嵌入 SQL 代码来操纵数据库查询，是常见的安全漏洞。"),
    ("什么是 REST API？", "基于 HTTP 协议的无状态 API 架构风格，使用标准 HTTP 方法（GET/POST/PUT/DELETE）操作资源。"),
    ("Python 中 `__init__` 是什么？", "Python 类的构造方法，在创建对象时自动调用。"),
    ("什么是正则表达式？", "用特殊语法定义的字符串匹配模式，用于搜索、替换、验证文本。"),
    ("什么是 LRU 缓存？", "Least Recently Used，淘汰最久未使用的缓存项，常用 `functools.lru_cache` 实现。"),
]

# ---------------------------------------------------------------------------
# 填充数据（模拟不同项目的代码搜索场景）
# ---------------------------------------------------------------------------

FUNCS = [
    ("load_model", "model.py", "45", "config, device='cuda'"),
    ("train_step", "trainer.py", "112", "batch, optimizer"),
    ("evaluate", "eval.py", "28", "model, dataloader"),
    ("save_checkpoint", "utils.py", "67", "model, path"),
    ("load_config", "config.py", "15", "path='config.yaml'"),
    ("preprocess", "data.py", "89", "text, tokenizer"),
    ("tokenize_batch", "data.py", "134", "texts, max_len=512"),
    ("compute_loss", "model.py", "201", "logits, labels"),
    ("init_weights", "model.py", "178", ""),
    ("get_lr_schedule", "optimizer.py", "34", "total_steps, warmup"),
    ("parse_args", "main.py", "12", ""),
    ("setup_logging", "utils.py", "8", "level='INFO'"),
    ("create_dataloader", "data.py", "56", "dataset, batch_size"),
    ("generate_text", "inference.py", "42", "prompt, max_tokens=256"),
    ("apply_rope", "attention.py", "88", "x, cos, sin"),
    ("build_model", "model.py", "15", "config"),
    ("run_evaluation", "eval.py", "95", "checkpoint_path"),
    ("export_onnx", "export.py", "22", "model, output_path"),
    ("merge_weights", "utils.py", "145", "base, adapter"),
    ("quantize_model", "quantize.py", "31", "model, bits=4"),
    ("load_dataset", "data.py", "12", "path, split='train'"),
    ("forward", "model.py", "250", "self, input_ids, targets=None"),
    ("backward", "trainer.py", "88", "loss"),
    ("clip_gradients", "trainer.py", "102", "model, max_norm=1.0"),
    ("log_metrics", "logger.py", "45", "metrics, step"),
]

CLASSES = [
    ("GPTModel", "model.py", "50", "nn.Module"),
    ("Transformer", "model.py", "25", "nn.Module"),
    ("DataProcessor", "data.py", "18", ""),
    ("TrainingConfig", "config.py", "8", ""),
    ("Tokenizer", "tokenizer.py", "12", ""),
    ("Attention", "attention.py", "35", "nn.Module"),
    ("MLP", "model.py", "120", "nn.Module"),
    ("Optimizer", "optimizer.py", "15", ""),
    ("LRScheduler", "scheduler.py", "22", ""),
    ("Evaluator", "eval.py", "10", ""),
    ("CheckpointManager", "utils.py", "88", ""),
    ("TextDataset", "data.py", "45", "Dataset"),
    ("Logger", "logger.py", "8", ""),
    ("ModelConfig", "config.py", "30", ""),
    ("InferenceEngine", "inference.py", "18", ""),
]

VARS = [
    ("LEARNING_RATE", "config.py", "5", "3e-4"),
    ("BATCH_SIZE", "config.py", "6", "32"),
    ("MAX_SEQ_LEN", "config.py", "7", "2048"),
    ("VOCAB_SIZE", "config.py", "8", "32768"),
    ("NUM_LAYERS", "config.py", "9", "12"),
    ("HIDDEN_DIM", "config.py", "10", "768"),
    ("NUM_HEADS", "config.py", "11", "12"),
    ("DROPOUT", "config.py", "12", "0.0"),
    ("WEIGHT_DECAY", "config.py", "13", "0.01"),
    ("WARMUP_STEPS", "config.py", "14", "100"),
    ("TOTAL_STEPS", "config.py", "15", "10000"),
    ("EVAL_INTERVAL", "config.py", "16", "500"),
    ("CHECKPOINT_DIR", "config.py", "17", "'./checkpoints'"),
    ("LOG_DIR", "config.py", "18", "'./logs'"),
    ("DATA_PATH", "config.py", "19", "'./data/train.jsonl'"),
    ("SEED", "config.py", "20", "42"),
    ("DEVICE", "train.py", "25", "'cuda'"),
    ("GRAD_ACCUM", "train.py", "30", "8"),
    ("ROPE_THETA", "model.py", "55", "10000.0"),
    ("SOFTCAP", "model.py", "60", "15"),
]

MODULES = [
    ("torch", "model.py", "1", "nn"),
    ("numpy", "data.py", "3", "np"),
    ("json", "utils.py", "2", ""),
    ("os", "main.py", "1", ""),
    ("argparse", "main.py", "2", ""),
    ("math", "model.py", "4", ""),
    ("tiktoken", "tokenizer.py", "5", ""),
    ("transformers", "model.py", "8", "AutoTokenizer"),
    ("torch.nn.functional", "model.py", "2", "F"),
    ("pathlib", "utils.py", "3", "Path"),
]

KEYWORDS = [
    ("gradient clipping", "trainer.py", "102", "# 梯度裁剪防止爆炸", "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)"),
    ("early stopping", "trainer.py", "180", "# 提前终止", "if no_improve > patience: break"),
    ("mixed precision", "train.py", "45", "# 混合精度训练", "autocast_ctx = torch.amp.autocast(...)"),
    ("learning rate", "scheduler.py", "30", "# 学习率调度", "lr = base_lr * warmup_factor"),
    ("data augmentation", "data.py", "120", "# 数据增强", "augmented = random_crop(text, max_len)"),
    ("attention mask", "model.py", "180", "# 注意力遮罩", "mask = torch.triu(torch.ones(T, T), diagonal=1)"),
    ("weight initialization", "model.py", "165", "# 权重初始化", "nn.init.normal_(self.weight, std=0.02)"),
    ("checkpoint", "utils.py", "60", "# 保存检查点", "torch.save(state, path)"),
    ("tokenization", "tokenizer.py", "40", "# 分词", "ids = self.enc.encode(text)"),
    ("embedding", "model.py", "90", "# 嵌入层", "x = self.wte(input_ids)"),
]

PARAMS = [
    ("learning_rate", "float", "3e-4"),
    ("batch_size", "int", "32"),
    ("max_seq_len", "int", "2048"),
    ("vocab_size", "int", "32768"),
    ("n_layer", "int", "12"),
    ("n_embd", "int", "768"),
    ("n_head", "int", "12"),
    ("dropout", "float", "0.0"),
    ("rope_theta", "float", "10000.0"),
    ("weight_decay", "float", "0.01"),
]

EXCEPTIONS = [
    ("ValueError", "config.py", "invalid config"),
    ("FileNotFoundError", "data.py", "data file not found"),
    ("RuntimeError", "model.py", "CUDA out of memory"),
    ("AssertionError", "train.py", "batch size mismatch"),
    ("KeyError", "utils.py", "missing key in checkpoint"),
]

FILES_FOR_READ = [
    ("config.py", "1", "15", "1 # 项目配置\n2 LEARNING_RATE = 3e-4\n3 BATCH_SIZE = 32\n4 MAX_SEQ_LEN = 2048\n5 VOCAB_SIZE = 32768", "配置文件包含了学习率、批大小等基本参数。"),
    ("model.py", "50", "12", "50 @dataclass\n51 class GPTConfig:\n52     n_layer: int = 12\n53     n_embd: int = 768\n54     n_head: int = 12", "这是模型配置类，定义了层数、维度和注意力头数。"),
    ("train.py", "1", "10", "1 #!/usr/bin/env python3\n2 \"\"\"Training script.\"\"\"\n3 import os\n4 import torch", "训练脚本开头导入了基本依赖。"),
    ("README.md", "1", "20", "1 # Project\n2 \n3 A language model training framework.\n5 ## Setup\n7 pip install -r requirements.txt", "README 介绍了项目概况和安装方法。"),
    ("data.py", "80", "10", "80 def preprocess(text):\n81     text = text.strip()\n82     if len(text) < 10:\n83         return None\n84     return tokenizer.encode(text)", "数据预处理函数会过滤过短文本。"),
    ("eval.py", "25", "12", "25 def evaluate(model, dataloader):\n26     model.eval()\n27     total_loss = 0\n28     with torch.no_grad():\n29         for batch in dataloader:\n30             loss = model(batch)", "评估函数遍历数据集计算平均 loss。"),
    ("requirements.txt", "1", "8", "1 torch>=2.0\n2 tiktoken\n3 numpy\n4 pyarrow", "依赖文件列出了 torch、tiktoken 等核心包。"),
    ("Makefile", "1", "10", "1 train:\n2 \tpython train.py\n3 eval:\n4 \tpython eval.py", "Makefile 定义了 train 和 eval 两个目标。"),
    ("utils.py", "60", "8", "60 def save_checkpoint(model, path):\n61     state = model.state_dict()\n62     torch.save(state, path)\n63     print(f'Saved to {path}')", "checkpoint 保存函数将模型权重写入磁盘。"),
    ("tokenizer.py", "10", "10", "10 class Tokenizer:\n11     def __init__(self, vocab_path):\n12         self.enc = tiktoken.Encoding(...)\n13     def encode(self, text):\n14         return self.enc.encode(text)", "Tokenizer 类封装了 tiktoken 编码。"),
]

# ---------------------------------------------------------------------------
# 生成函数
# ---------------------------------------------------------------------------

def make_search_sample(question, query, result, summary):
    return {"messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": f'<|tool_call_start|><|tool_name_search_code|>{{"query":"{query}"}}<|tool_call_end|>'},
        {"role": "assistant", "content": f"<|tool_result_start|>{result}<|tool_result_end|>"},
        {"role": "assistant", "content": summary},
    ]}

def make_read_sample(question, path, offset, limit, content, summary):
    return {"messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": f'<|tool_call_start|><|tool_name_read_file|>{{"path":"{path}","offset":{offset},"limit":{limit}}}<|tool_call_end|>'},
        {"role": "assistant", "content": f"<|tool_result_start|>{content}<|tool_result_end|>"},
        {"role": "assistant", "content": summary},
    ]}

def make_direct_sample(question, answer):
    return {"messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]}


def generate_all():
    samples = []

    # --- search_code 样本 ---
    # 函数搜索
    for func, file, line, args in FUNCS:
        templates = random.sample(SEARCH_TEMPLATES[:3], min(2, len(SEARCH_TEMPLATES[:3])))
        for tpl in templates:
            q = tpl[0].format(func=func)
            query = tpl[1].format(func=func)
            result = tpl[2].format(func=func, file=file, line=line, args=args)
            summary = tpl[3].format(func=func, file=file, line=line)
            samples.append(make_search_sample(q, query, result, summary))

    # 类搜索
    for cls, file, line, base in CLASSES:
        tpl = random.choice(SEARCH_TEMPLATES[3:5])
        q = tpl[0].format(cls=cls)
        query = tpl[1].format(cls=cls)
        base_str = base if base else "object"
        result = tpl[2].format(cls=cls, file=file, line=line, base=base_str)
        summary = tpl[3].format(cls=cls, file=file, line=line, base=base_str)
        samples.append(make_search_sample(q, query, result, summary))

    # 变量搜索
    for var, file, line, val in VARS:
        tpl = random.choice(SEARCH_TEMPLATES[5:7])
        q = tpl[0].format(var=var)
        query = tpl[1].format(var=var)
        result = tpl[2].format(var=var, file=file, line=line, val=val)
        summary = tpl[3].format(var=var, file=file, line=line, val=val)
        samples.append(make_search_sample(q, query, result, summary))

    # import 搜索
    for module, file, line, item in MODULES:
        tpl = random.choice(SEARCH_TEMPLATES[7:9])
        q = tpl[0].format(module=module)
        query = tpl[1].format(module=module)
        item_str = item if item else module
        result = tpl[2].format(module=module, file=file, line=line, item=item_str)
        summary = tpl[3].format(module=module, file=file, line=line, item=item_str)
        samples.append(make_search_sample(q, query, result, summary))

    # 关键词搜索
    for keyword, file, line, comment, code in KEYWORDS:
        tpl = random.choice(SEARCH_TEMPLATES[9:11])
        q = tpl[0].format(keyword=keyword)
        query = keyword
        result = tpl[2].format(keyword=keyword, file=file, line=line, comment=comment, code=code)
        summary = tpl[3].format(file=file, line=line)
        samples.append(make_search_sample(q, query, result, summary))

    # 参数搜索
    for param, type_, default in PARAMS:
        tpl = random.choice(SEARCH_TEMPLATES[11:13])
        q = tpl[0].format(param=param)
        query = tpl[1].format(param=param)
        result = tpl[2].format(param=param, file="config.py", line=str(random.randint(5, 30)), type=type_, default=default)
        summary = tpl[3].format(param=param, default=default, type=type_, file="config.py", line=str(random.randint(5, 30)))
        samples.append(make_search_sample(q, query, result, summary))

    # 异常搜索
    for exc, file, msg in EXCEPTIONS:
        tpl = SEARCH_TEMPLATES[13]
        q = tpl[0].format(exc=exc)
        query = f"raise {exc}"
        result = tpl[2].format(exc=exc, file=file, line=str(random.randint(20, 100)), msg=msg)
        summary = tpl[3].format(file=file, line=str(random.randint(20, 100)), exc=exc, msg=msg)
        samples.append(make_search_sample(q, query, result, summary))

    # --- read_file 样本 ---
    for file, offset, limit, content, summary in FILES_FOR_READ:
        # 每个文件生成 2-3 个不同角度的读取请求
        read_questions = [
            f"请读取 {file} 的前 {limit} 行。",
            f"把 {file} 里第 {offset} 行开始的内容给我看看。",
            f"读取 {file}。",
        ]
        for q in random.sample(read_questions, 2):
            samples.append(make_read_sample(q, file, offset, limit, content, summary))

    # 更多 read_file 变体
    extra_reads = [
        ("请读取 setup.py 的内容。", "setup.py", 1, 10, "1 from setuptools import setup\n2 setup(\n3     name='myproject',\n4     version='0.1.0',\n5 )", "这是项目的 setup.py，包名为 myproject。"),
        ("给我看 .gitignore 文件。", ".gitignore", 1, 8, "1 __pycache__/\n2 *.pyc\n3 .env\n4 checkpoints/\n5 *.log", ".gitignore 排除了缓存、环境变量和日志文件。"),
        ("读取 pyproject.toml。", "pyproject.toml", 1, 10, "1 [project]\n2 name = \"attnres\"\n3 version = \"0.1.0\"\n4 requires-python = \">=3.10\"", "项目使用 pyproject.toml 管理，要求 Python 3.10+。"),
        ("请读取 Dockerfile。", "Dockerfile", 1, 8, "1 FROM python:3.10-slim\n2 WORKDIR /app\n3 COPY . .\n4 RUN pip install -r requirements.txt", "Dockerfile 基于 Python 3.10，安装依赖后复制代码。"),
        ("给我看日志文件最新的内容。", "train.log", 990, 10, "990 step 4990 | loss: 0.542 | lr: 0.0001\n991 step 4995 | loss: 0.538\n992 step 5000 | loss: 0.535\n993 --- eval ---\n994 val_bpb: 0.573", "日志显示训练到 step 5000，val_bpb=0.573。"),
    ]
    for q, path, offset, limit, content, summary in extra_reads:
        samples.append(make_read_sample(q, path, offset, limit, content, summary))

    # --- 直接回答样本 ---
    for q, a in DIRECT_ANSWER_TEMPLATES:
        samples.append(make_direct_sample(q, a))

    # 更多直接回答变体（中文技术问答）
    extra_direct = [
        ("什么是 Transformer？", "Transformer 是一种基于自注意力机制的神经网络架构，广泛用于 NLP 和 CV。"),
        ("PyTorch 和 TensorFlow 哪个好？", "PyTorch 更灵活，适合研究；TensorFlow 部署生态更成熟。看具体需求选择。"),
        ("什么是 BPE？", "Byte Pair Encoding，一种子词分词算法，通过迭代合并最频繁的字节对来构建词表。"),
        ("什么是 RoPE？", "Rotary Position Embedding，旋转位置编码，通过旋转矩阵将位置信息编码到注意力的 Q/K 中。"),
        ("什么是 KV Cache？", "在自回归生成时缓存已计算的 Key 和 Value，避免重复计算，加速推理。"),
        ("什么是 Flash Attention？", "一种 IO-aware 的注意力实现，通过分块计算减少 HBM 访问，提速 2-4 倍。"),
        ("Adam 和 SGD 的区别？", "Adam 自适应学习率，收敛快但可能泛化差；SGD 更简单，配合学习率调度泛化更好。"),
        ("什么是 bfloat16？", "Brain Floating Point 16，Google 设计的 16 位浮点格式，指数位数与 float32 相同，精度略低但数值范围不变。"),
        ("什么是 MoE？", "Mixture of Experts，混合专家模型，每个 token 只激活部分专家，实现更大参数量但不增加计算量。"),
        ("什么是 LoRA？", "Low-Rank Adaptation，通过低秩分解只训练少量参数来微调大模型，节省显存。"),
        ("什么是 DPO？", "Direct Preference Optimization，直接用偏好数据优化模型，不需要单独训练奖励模型。"),
        ("什么是 RLHF？", "Reinforcement Learning from Human Feedback，用人类反馈训练奖励模型，再用 PPO 优化策略模型。"),
        ("GPU 显存不够怎么办？", "可以尝试：减小 batch size、用梯度累积、用混合精度训练、用 gradient checkpointing。"),
        ("什么是 beam search？", "一种解码策略，每步保留 top-k 个候选序列，最终选得分最高的。比贪心更好但比采样更确定。"),
        ("loss 突然变成 NaN 是什么原因？", "通常是学习率过大、梯度爆炸或数值溢出。尝试降低学习率、加梯度裁剪或检查数据。"),
        ("什么是 warmup？", "训练初期逐步增大学习率的策略，避免模型在随机初始化状态下因大学习率导致不稳定。"),
        ("什么是 cosine annealing？", "余弦退火学习率调度，学习率按余弦曲线从初始值下降到最小值。"),
        ("Python 中 yield 是什么？", "`yield` 让函数变成生成器，每次调用返回一个值并暂停，适合处理大数据流。"),
        ("什么是 GIL？", "Global Interpreter Lock，CPython 的全局解释器锁，限制同一时刻只有一个线程执行 Python 字节码。"),
        ("pip install -e . 是什么意思？", "以可编辑模式安装当前包，代码修改后不需要重新安装。"),
    ]
    for q, a in extra_direct:
        samples.append(make_direct_sample(q, a))

    # 过采样工具调用样本（确保比例够高）
    tool_samples = [s for s in samples if len(s["messages"]) > 2]
    direct_samples = [s for s in samples if len(s["messages"]) == 2]

    # 工具调用样本再重复一次增加权重
    all_samples = tool_samples * 2 + direct_samples
    random.shuffle(all_samples)
    return all_samples


if __name__ == "__main__":
    samples = generate_all()
    out_path = "docs/tool_call_samples_v3.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    tool_count = sum(1 for s in samples if len(s["messages"]) > 2)
    direct_count = sum(1 for s in samples if len(s["messages"]) == 2)
    print(f"总样本: {len(samples)}")
    print(f"  工具调用: {tool_count}")
    print(f"  直接回答: {direct_count}")
    print(f"输出: {out_path}")
