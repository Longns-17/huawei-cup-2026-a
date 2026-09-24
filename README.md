# 华为杯 A 题：FINAL 独立实现

本仓库保存实验代码、官方输入、实际生成的实验结果与复现入口。当前完成 **47 项测试、6 个算例的三问 1—5 核共 90 个配置、90 份共同池增强提交**；主搜索及补充实验合计 529 次官方评价（不含测试中的调用）。**100 例正式全量尚未启动**。详细证据见 [验证报告](checks/验证报告.md)。

本机 RTX 5070 Laptop GPU 的驻留扫描实测加速 83.29 倍，与 CPU 输出完全一致；此比值仅对应扫描模块。仓库不包含虚拟环境和缓存，依赖版本见 `requirements-lock.txt`。以下绝对路径及实验来源记录保留原运行电脑的信息；克隆后的求解入口按仓库自身位置解析路径。

正式工作区：`D:\研一上学期\华为杯\_code`。用 [工作区文件](华为杯_code.code-workspace) 打开；环境固定为本目录的 `.venv`，全部代码、输入、实验记录和图表都放在此处。

依据 [FINAL 快照](specification/FINAL_IDEA.md) 实现三问。快照 SHA-256 为 `405ddd925ef27eaf4010e7ff557768b290e9fea27b6795ed2c8980a0798174c1`。原文中的历史成绩是方案背景，本工程只报告自己实际运行产生的成绩。

这里使用此前独立编写的 `src/npu_schedule` 实现，重新建立环境并从原附件解压 100 个官方输入，再从头生成实验结果。旧 `_code` 完整保存在同级 `_code_旧版归档_20260924_181720`。本次部署没有导入旧实验数据；[来源记录](specification/implementation_origin.json) 保留了独立源码来源与文件哈希。

## 直接使用

在本目录打开 PowerShell，执行以下入口。每次计算都创建新批次，默认名称带时间；指定名称时使用尚不存在的名称。

```powershell
# 单元测试、GPU 实测、六例三问五核、补充对照、生成验证报告
.\开始验证.ps1

# 检查现有最新验证批次并更新报告，不重新求解
.\更新验证报告.ps1

# 100 例 × 三问 × 1—5 核，全量主求解
.\开始全量实验.ps1
```

[验证报告](checks/验证报告.md) 由真实运行记录自动生成。`experiments/LATEST.json` 指向最近一次完整验证的各个批次；全量入口会在结束时输出自身目录，不会把未完成全量冒充为已完成验证。

验证选择 006、019、064、069、078、090，覆盖单分量、多分量、共享输入和首读错峰案例；这六例不能代表 100 例总体成绩。GPU 基准另用 014 大图。默认同时处理两个算例，`-Workers 1` 可明确改为串行处理。官方大图评价可能耗时较长；程序没有强制超时或时间截断。

## 单例与全量后的分析

```powershell
# 指定算例；显式使用 GPU
.\.venv\Scripts\python.exe -X utf8 -B run.py solve --name my_case019 --cases 019 --scenes 1 2 3 --cores 5 --device cuda --workers 1

# 以下假设主求解批次叫 my_full；替换为实际目录名称
.\.venv\Scripts\python.exe -X utf8 -B run.py report experiments/my_full
.\.venv\Scripts\python.exe -X utf8 -B run.py paired experiments/my_full --name my_full_pairs

# 预先按结构和规模选取 20 例，固定问题三四核方案进行 9 档资源扫描
.\.venv\Scripts\python.exe -X utf8 -B run.py inventory
.\.venv\Scripts\python.exe -X utf8 -B run.py resources experiments/my_full --name my_full_resources --cases sample20

# 全部单分量图：仅替换 10% 预算候选的拓扑顺序
.\.venv\Scripts\python.exe -X utf8 -B run.py topology experiments/my_full --name my_full_topology --cases single --device cuda

# 固定种子单元，比较三种定价；同时包含 Kahn/DFS 预算候选对照
.\.venv\Scripts\python.exe -X utf8 -B run.py ablation --name my_ablation --cases 064 069 078 090 --cores 4 --device cuda
```

`paired` 同时生成同方案开关 L2 的对照，以及问题二、三共同候选池补评后的增强提交；保留主搜索结果。资源扫描固定提交，不能解释为资源变化后重新优化。拓扑实验只替换指定候选，其余候选和原有低核数继承项固定。

## 实现与 FINAL 的对应关系

| 模块 | 实现内容 |
|---|---|
| `problem.py`、`construction.py` | 原图抽象、分量/链、局部合法合并、10%/5% 任务预算、HEFT 插空分核 |
| `calendar.py` | 用整数代理时钟维护空闲区间，提高大图插空速度 |
| `residency.py` | CPU 与 CUDA 整核串行驻留扫描，实际调用本机 GPU |
| `pricing.py` | 源端写出释放、逐 Pipe 完成估计、目标核搬运定价、首读时序与缓存路径定价 |
| `ordering.py` | 保持归属的释放排序与首读错峰 |
| `candidates.py`、`search.py` | 固定尝试顺序，结构去重，4/4/7 准入预算，整图/低核数继承，严格字典序择优 |
| `official.py`、`metrics.py` | 调用未修改的官方程序；汇总周期、搬运、缓存、下界与内存峰值 |
| `analysis.py`、`statistics.py` | 共同池、公平对照、机制入选、结构分层、固定方案资源扫描 |
| `verification.py` | 重新验证及基于证据生成中文报告 |

GPU 加速的是候选构造中的驻留扫描，**官方评分仍在 CPU 运行**。扫描是串行访问代理，不等于官方并发执行的真实内存峰值。测试分别验证两者；GPU 基准不会被写成整个求解过程的速度比。

遵照用户要求，代码不吞错、不自动重试、不跳过失败候选、不自动切换 CPU、不生成失败兜底解。模型要求的整图候选和低核数继承正常保留。官方合法性检查和实验验证断言会直接暴露问题。

## 数据与结果位置

- `data/raw/`：从官方附件独立解压的 100 个输入及配置；[哈希记录](specification/input_provenance.json)。不改造正式测试输入。
- `data/derived/`：重新计算的图结构描述和预先确定的 20 例样本。
- `official/`：未改动的官方代码和说明。
- `experiments/<批次>/<算例>/q<问号>/k<核数>/`：两字段正式提交与 `metrics.json`。
- `experiments/<批次>/<算例>/q<问号>/evaluations/`：每次实际评价的提交和压缩官方结果。
- `experiments/<批次>/`：结果表、统计、完成状态、源码快照和输入哈希；`figures/` 存放性能图。
- `experiments/<配对批次>/best/`：共同池产生的增强提交；原主结果不覆盖。
- `experiments/<校准批次>/`：明确标注的人工机制例和新实验数据，与正式输入分开。
- `checks/`：测试记录、自动生成的验证报告和机器可读核验结果。
- `reports/`：GPU 实测、真实执行时序图和本工作区 CUDA 缓存。

## 环境复现

当前已安装独立 Python 3.12 环境。迁移到另一台配置兼容的 NVIDIA GPU 电脑时，在项目目录用已安装的 `uv` 创建环境并安装锁定版本：

```powershell
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe --index-url https://pypi.tuna.tsinghua.edu.cn/simple --link-mode copy -r requirements-lock.txt
.\开始验证.ps1
```

需要匹配 CUDA 13 的驱动。仅 CPU 环境可在单例或主求解命令明确写 `--device cpu`；完整验证入口专门检查本机 CUDA，不会自动切换设备。
