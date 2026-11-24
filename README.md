# 共享视图上的训练记忆写入与稀疏归因

离线模型记忆由可稀疏读取的物理页和几种张量视图组成。 数据为本地合成。

- `data/capture.json` 保存采集原始内容。records 是交换记录，tensors 字段分开描述名称、类型、形状和值。整数索引从零开始，未指定闭合的区间均为左闭右开。
- `data/capacity.json` 保存制造端的容量、规模或分片条件。大尺寸是接口需考虑的使用规模。工作区计入新分配的解包、转换、归约、版本暂存及交接缓冲。调用方已有只读输入和已消费的输出不计入工作区，额外物化副本计入。

## 交换约定

索引从零开始。视图字段 `base` 为基址，`strides` 为按元素计的有符号步幅，`shape` 定义合法逻辑索引。每页地址范围按 `page_width` 连续延伸，未列页是无归属。数组按视图定义的逻辑坐标提交，梯度以 float64 累加。
写操作 `add`、`set` 分别表示累加、替换，值和读写操作均有样本 ID 与序号。回传报告按读身份或物理地址归因，回传组可以混用视图读描述，但需与已引用的原始读身份匹配。若现有存储少一个物理地址的上游梯度，可通过声明该地址为零完成覆盖。

## 读取资料

使用 Python 3.11。首次创建本地环境时安装锁定依赖。只读查看显示原始字段和张量尺寸。

```sh
python -m pip install -r requirements.txt
python caseio.py data/capture.json
python -m unittest discover -s tests -v
```

现有检查核对采集格式和所附数据的结构。

## 批次读写算子 `memop.py`

`MemoryStore` 把一次训练步的 Gather 与 Scatter 合并为一次原子提交，直接消费采集记录中的视图、操作与上游回传格式。

- `submit_step(step_id, *, base, operations, views)`：提交带代次的基存储声明。读操作全部读到步开始时的页快照；视图经 `base + Σ strides[d]*coords[d]` 换算物理地址，目标身份只按物理地址判定，支持转置与负步幅。写操作按 `(样本序号, 操作序号)` 定全序，`add` 追加贡献，`set` 淘汰该地址此前的基值与贡献、其后的累加仍有效。返回读取批次、写后值、新代次与回传凭据 `token`。
- `backward(step_id, reports, *, token)`：按读身份与物理地址归因。读归因始终回到该步基版；同一读身份的重复递交按回传组 ID 去重，不同有效读命中同一物理位置则相加；被 `set` 淘汰的写操作梯度为零，只有仍决定新值的贡献与被保留的基值能收到写路径梯度。未覆盖全部写目标（未声明零值策略时）抛 `IncompleteBackwardError`，不能交给更新器。
- `apply_gradients(step_id)`：只接收已完成回传的步骤，把基版地址梯度以 float64 累加进更新器缓冲。
- 整步拒绝条件（无归属地址、基代次过期、同序号冲突、重复操作身份、凭据错误、基版已回收）均抛出带 `code` 与 `details` 的 `OperatorError` 子类，消息可用于修正后重新提交；被拒步骤不落写、代次保留，同一步身份重放不会再次落写。
- 稀疏版本管理：只复制被触及页的旧代次副本；开放步超过 `open_steps`（对应容量条件 `open_forward_steps`）后最旧步骤的基版过期，回传得到 `ExpiredVersionError`；`scratch_bytes` 暴露版本暂存与交接缓冲占用，容量测试核对其不超过 `peak_scratch_bytes`。

```sh
python run_demo.py data/capture.json
python -m unittest discover -s tests -v
```

隔离模式（`-I` 不读取 cwd/PYTHONPATH）下运行需显式登记路径：

```sh
python -I -c "import sys; sys.path[:0]=['.', 'tests']; import unittest, importlib; \
  suite=unittest.TestSuite(); loader=unittest.TestLoader(); \
  [suite.addTests(loader.loadTestsFromModule(importlib.import_module(m))) for m in ('test_capture','test_memop')]; \
  raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)"
```
