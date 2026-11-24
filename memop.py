"""共享视图上的批次读写算子（带代次的稀疏页存储）。

算子把一次训练步的 Gather 与 Scatter 合并为一次原子提交：

* 读操作一律看到步开始时的基版镜像，写操作按 ``(样本序号, 操作序号)``
  定全序，``add`` 追加贡献，``set`` 淘汰该物理地址上此前的基值与贡献；
* 目标身份一律按物理地址判定，视图名只用于逻辑坐标到物理地址的换算；
* 无归属地址、同序号冲突、基代次过期等问题整步拒绝，存储保留此前代次；
* 同一步身份重放不会再次落写；
* 回传按读身份去重、按物理地址聚合，被 ``set`` 淘汰的写操作收不到写路径
  梯度，未完成覆盖的回传不能交给更新器；
* 只暂存被触及页的旧代次副本，开放步超过预算后最旧的代次过期，绝不复制
  整个记忆。

典型用法（与 ``data/capture.json`` 的记录格式互通）::

    store = build_store_from_records(records, tensors)
    result = store.submit_step("s11", base=10, operations=ops, views=views)
    report = store.backward("s11", upstream, token=result.token)
    store.apply_gradients("s11")
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "View",
    "Operation",
    "GradientReport",
    "StepResult",
    "BackwardResult",
    "MemoryStore",
    "OperatorError",
    "StaleGenerationError",
    "UnownedAddressError",
    "OrderingConflictError",
    "CredentialError",
    "IncompleteBackwardError",
    "ExpiredVersionError",
    "build_store_from_records",
    "view_from_record",
    "operation_from_record",
    "report_from_record",
]

_READ = "read"
_ADD = "add"
_SET = "set"
_KINDS = (_READ, _ADD, _SET)


# --------------------------------------------------------------------------- #
# 可用于重新提交的错误：全部带机器可读 code 与中文解释
# --------------------------------------------------------------------------- #
class OperatorError(Exception):
    """算子拒绝请求的基类，携带可用于重新提交的解释。"""

    code = "operator_error"

    def __init__(self, message: str, **details):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:  # pragma: no cover - 简单委托
        return self.message


class StaleGenerationError(OperatorError):
    """提交声明的基代次与存储当前代次不一致。"""

    code = "stale_generation"


class UnownedAddressError(OperatorError):
    """逻辑坐标换算出的物理地址落在无归属页上。"""

    code = "unowned_address"


class OrderingConflictError(OperatorError):
    """同一步内出现无法排序的同序号请求或重复操作身份。"""

    code = "ordering_conflict"


class CredentialError(OperatorError):
    """回传凭据、读身份、物理地址或重复递交内容对不上。"""

    code = "bad_credential"


class IncompleteBackwardError(OperatorError):
    """上游导数没有覆盖本步全部写目标，回传未完成，不能交给更新器。"""

    code = "incomplete_backward"


class ExpiredVersionError(OperatorError):
    """步骤的基版已超出开放步预算被回收，无法再按旧代次回查。"""

    code = "expired_version"


# --------------------------------------------------------------------------- #
# 交换对象
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class View:
    """逻辑张量视图：``address(coords) = base + Σ strides[d]*coords[d]``。"""

    name: str
    base: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]

    def __post_init__(self):
        if len(self.shape) != len(self.strides):
            raise ValueError(f"视图 {self.name} 的 shape 与 strides 维数不一致")
        if any(size <= 0 for size in self.shape):
            raise ValueError(f"视图 {self.name} 的 shape 必须为正")

    def address(self, coords: Sequence[int]) -> int:
        coords = tuple(coords)
        if len(coords) != len(self.shape):
            raise CredentialError(
                f"视图 {self.name} 期望 {len(self.shape)} 维坐标，收到 {len(coords)} 维",
                view=self.name,
                coords=coords,
            )
        for axis, (index, size) in enumerate(zip(coords, self.shape)):
            if not (0 <= index < size):
                raise CredentialError(
                    f"视图 {self.name} 第 {axis} 维逻辑索引 {index} 越界（合法区间 [0,{size})）",
                    view=self.name,
                    axis=axis,
                    index=index,
                    bound=size,
                )
        return int(self.base + sum(s * c for s, c in zip(self.strides, coords)))


@dataclass(frozen=True)
class Operation:
    """一条读/写请求。``sample`` 与 ``seq`` 共同决定写全序。"""

    id: str
    sample: int
    seq: int
    kind: str
    view: str
    coords: tuple[int, ...]
    value: float | None = None

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"未知操作类型 {self.kind!r}，合法值为 {_KINDS}")
        if self.kind != _READ and self.value is None:
            raise ValueError(f"写操作 {self.id} 缺少 value")


@dataclass(frozen=True)
class GradientReport:
    """一组上游回传：读身份导数 + 新代次物理地址导数。"""

    id: str
    reads: tuple[tuple[str, float], ...] = ()
    new_values: tuple[tuple[int, float], ...] = ()


@dataclass
class StepResult:
    """前向提交结果。``token`` 是回传时必须出示的凭据。"""

    step_id: str
    base_generation: int
    new_generation: int
    reads: dict[str, tuple[int, float]]
    written_addresses: tuple[int, ...]
    values: dict[int, float]
    token: str
    replayed: bool = False

    def read_batch(self) -> dict[str, float]:
        """按读操作身份返回读到的步开始镜像值。"""
        return {op_id: value for op_id, (_, value) in self.reads.items()}


@dataclass
class BackwardResult:
    """一次完整回传的归因结果。"""

    step_id: str
    base_generation: int
    new_generation: int
    read_grads: dict[str, float]
    write_grads: dict[str, float]
    base_grads: dict[int, float]
    base_values: dict[int, float]
    complete: bool = True


# --------------------------------------------------------------------------- #
# 记录与配方（内部状态）
# --------------------------------------------------------------------------- #
@dataclass
class _Recipe:
    """某个物理地址在本步内的写配方。"""

    address: int
    base_retained: bool = True
    # 仍决定新值的写操作（有序），元素为 (sample, seq, op_id, kind, value)
    surviving: list[tuple] = field(default_factory=list)
    eliminated: set[str] = field(default_factory=set)


@dataclass
class _Record:
    step_id: str
    base_generation: int
    new_generation: int
    token: str
    read_ops: list[tuple[str, int]]               # (op_id, 物理地址)，保持提交序
    targets: set[int]                              # 写目标物理地址
    recipes: dict[int, _Recipe]
    snapshots: dict[int, np.ndarray]               # 仅被触及页的基版副本
    read_values: dict[str, tuple[int, float]]
    written_values: dict[int, float]
    status: str = "open"                           # open / complete / applied / expired
    result: BackwardResult | None = None
    seen_reports: dict[str, tuple] = field(default_factory=dict)


class MemoryStore:
    """按稀疏页管理代次的训练记忆存储。

    Parameters
    ----------
    page_width:
        每页连续元素数，物理地址 ``addr`` 属于页 ``addr // page_width``。
    pages:
        已物化的页数据（页号 -> float64 一维数组），调用方传入的只读输入
        不计入算子工作区，算子在首次触及时才复制。
    generation:
        当前代次。
    open_steps:
        同时保留基版快照、等待回传的最大步数，对应容量条件中的
        ``open_forward_steps``；超出后最旧的步骤代次过期。
    owned_pages:
        归属声明。``None`` 表示以 ``pages`` 的键为准；整数 N 表示页号
        ``[0, N)`` 全部归属、惰性零初始化；可迭代对象表示显式页号集合。
    unlisted_upstream:
        ``"zero"`` 时回传未列出的写目标地址按零导数完成覆盖；否则缺地址
        判定为回传未完成。
    """

    def __init__(
        self,
        page_width: int,
        pages: Mapping[int, np.ndarray] | None = None,
        *,
        generation: int = 0,
        open_steps: int = 8,
        owned_pages: int | Iterable[int] | None = None,
        unlisted_upstream: str | None = None,
    ):
        self.page_width = int(page_width)
        self._pages: dict[int, np.ndarray] = {}
        for page_id, data in (pages or {}).items():
            array = np.asarray(data, dtype=np.float64)
            if array.shape != (self.page_width,):
                raise ValueError(f"页 {page_id} 形状应为 ({self.page_width},)")
            self._pages[int(page_id)] = array
        if owned_pages is None:
            self._owned: set[int] | None = set(self._pages)
            self._lazy = False
        elif isinstance(owned_pages, int):
            self._owned = set(range(owned_pages))
            self._lazy = True
        else:
            self._owned = set(int(p) for p in owned_pages)
            self._lazy = True
        self._generation = int(generation)
        self._open_budget = int(open_steps)
        self._zero_policy = unlisted_upstream == "zero"
        self._records: dict[str, _Record] = {}
        self._order: list[str] = []
        self._rejected: dict[str, OperatorError] = {}
        self._grad: dict[int, np.float64] = {}

    # ------------------------------------------------------------------ #
    # 观测属性
    # ------------------------------------------------------------------ #
    @property
    def generation(self) -> int:
        return self._generation

    @property
    def open_step_ids(self) -> tuple[str, ...]:
        """仍保留基版、可回传的步骤身份（旧 -> 新）。"""
        return tuple(sid for sid in self._order if self._records[sid].status != "applied")

    @property
    def gradients(self) -> dict[int, float]:
        """更新器已累加的物理地址梯度（float64），返回副本。"""
        return {addr: float(g) for addr, g in self._grad.items()}

    @property
    def scratch_bytes(self) -> int:
        """版本暂存与待交接缓冲占用的字节数（不含调用方只读输入）。"""
        total = 0
        for record in self._records.values():
            total += int(sum(page.nbytes for page in record.snapshots.values()))
            if record.result is not None:
                total += int(sum(8 for _ in record.result.base_grads))
                total += int(sum(8 for _ in record.result.write_grads))
                total += int(sum(8 for _ in record.result.read_grads))
        return total

    def materialized_pages(self) -> tuple[int, ...]:
        return tuple(sorted(self._pages))

    def get(self, address: int) -> float:
        """读取当前代次下某物理地址的值（无归属地址报错）。"""
        page_id, offset = self._split(address)
        self._require_owned(page_id, address)
        return float(self._materialize(page_id)[offset])

    # ------------------------------------------------------------------ #
    # 前向：提交一个训练步
    # ------------------------------------------------------------------ #
    def submit_step(
        self,
        step_id: str,
        *,
        base: int,
        operations: Sequence[Operation | Sequence],
        views: Mapping[str, View | Mapping] | None = None,
    ) -> StepResult:
        """原子提交一批读写操作，返回读批次与新代次。

        同一步身份重放时直接返回首次结果，绝不会再次落写；被拒绝过的
        步骤重放会再次给出同一条解释。
        """
        if step_id in self._records:
            record = self._records[step_id]
            if record.status == "expired":
                raise ExpiredVersionError(
                    f"步骤身份 {step_id} 的基版代次 {record.base_generation} 已超出开放步预算 "
                    f"{self._open_budget} 被回收；请更换步骤身份并基于当前代次 "
                    f"{self._generation} 重新提交",
                    step_id=step_id,
                    base_generation=record.base_generation,
                    current_generation=self._generation,
                )
            return self._replay_accepted(record)
        if step_id in self._rejected:
            raise self._rejected[step_id]

        view_map = self._coerce_views(views)
        ops = [self._coerce_operation(item) for item in operations]

        # 整步校验：任何一条不合法都不落写。
        self._check_base_generation(step_id, base)
        self._check_identities(step_id, ops)
        resolved = self._resolve_addresses(step_id, ops, view_map)

        # 仅快照被触及的页（含只读页，回传可能需要按旧代次查基值）。
        touched_pages = {addr // self.page_width for addr in resolved.values()}
        snapshots = {pid: self._materialize(pid).copy() for pid in touched_pages}

        read_values: dict[str, tuple[int, float]] = {}
        read_ops: list[tuple[str, int]] = []
        for op in ops:
            if op.kind != _READ:
                continue
            addr = resolved[op.id]
            page_id, offset = self._split(addr)
            read_values[op.id] = (addr, float(snapshots[page_id][offset]))
            read_ops.append((op.id, addr))

        recipes = self._build_recipes(ops, resolved)
        self._apply_writes(ops, resolved, recipes)

        written_values = {
            addr: float(self._pages[addr // self.page_width][addr % self.page_width])
            for addr in recipes
        }
        self._generation += 1
        record = _Record(
            step_id=step_id,
            base_generation=base,
            new_generation=self._generation,
            token=secrets.token_hex(16),
            read_ops=read_ops,
            targets=set(recipes),
            recipes=recipes,
            snapshots=snapshots,
            read_values=read_values,
            written_values=written_values,
        )
        self._records[step_id] = record
        self._order.append(step_id)
        self._retire_oldest()
        return self._replay_accepted(record, replayed=False)

    # ------------------------------------------------------------------ #
    # 反向：提交上游导数
    # ------------------------------------------------------------------ #
    def backward(
        self,
        step_id: str,
        reports: Sequence[GradientReport | Mapping],
        *,
        token: str,
    ) -> BackwardResult:
        """按读身份与物理地址完成回传归因。

        重复递交同一回传组且内容一致时去重；凭据错误、身份不匹配、覆盖
        不完整或基版过期都会抛出带解释的异常，可据此重新提交。
        """
        record = self._require_open_record(step_id)
        if not token or token != record.token:
            raise CredentialError(
                f"步骤 {step_id} 的回传凭据错误：请使用提交步时返回的 token 重新提交",
                step_id=step_id,
            )
        coerced = [self._coerce_report(item) for item in reports]

        if record.status == "complete":
            self._assert_same_reports(record, coerced)
            return record.result  # type: ignore[return-value]

        known_reads = {op_id for op_id, _ in record.read_ops}
        read_addr = dict(record.read_ops)

        read_grads: dict[str, float] = {op_id: 0.0 for op_id in known_reads}
        value_grads: dict[int, float] = {addr: 0.0 for addr in record.targets}
        covered: set[int] = set()
        incoming: dict[str, tuple] = {}

        for report in coerced:
            payload = (tuple(report.reads), tuple(report.new_values))
            previous = incoming.get(report.id)
            if previous is not None and previous != payload:
                raise CredentialError(
                    f"回传组 {report.id} 曾以不同内容递交过，凭据冲突，请核对后重新提交",
                    report_id=report.id,
                    step_id=step_id,
                )
            if previous is not None:
                continue  # 同内容重复递交：不重复记账
            incoming[report.id] = payload

            for op_id, grad in report.reads:
                if op_id not in known_reads:
                    raise CredentialError(
                        f"回传组 {report.id} 引用的读身份 {op_id} 不属于步骤 {step_id}",
                        report_id=report.id,
                        read_id=op_id,
                    )
                read_grads[op_id] += float(grad)
            for addr, grad in report.new_values:
                if addr not in record.targets:
                    raise CredentialError(
                        f"回传组 {report.id} 给出的地址 {addr} 不是步骤 {step_id} 的写目标",
                        report_id=report.id,
                        address=addr,
                    )
                value_grads[addr] += float(grad)
                covered.add(addr)

        missing = sorted(record.targets - covered)
        if missing and not self._zero_policy:
            raise IncompleteBackwardError(
                f"步骤 {step_id} 的回传未覆盖写目标地址 {missing}；"
                "可显式声明零导数后重新提交，未完成的回传不能交给更新器",
                step_id=step_id,
                missing_addresses=missing,
            )

        # 读归因：不同有效读操作命中同一物理位置相加，始终回到基版。
        base_grads: dict[int, float] = {}
        for op_id, grad in read_grads.items():
            if grad == 0.0:
                continue
            addr = read_addr[op_id]
            base_grads[addr] = base_grads.get(addr, 0.0) + grad

        # 写归因：只有仍决定新值的贡献与被保留的基值能收到梯度。
        write_grads: dict[str, float] = {}
        for addr, recipe in record.recipes.items():
            upstream = value_grads.get(addr, 0.0)
            for _, _, op_id, kind, _value in recipe.surviving:
                write_grads[op_id] = float(upstream)
            for op_id in recipe.eliminated:
                write_grads[op_id] = 0.0
            if recipe.base_retained and upstream != 0.0:
                base_grads[addr] = base_grads.get(addr, 0.0) + upstream

        base_values = {
            addr: float(record.snapshots[addr // self.page_width][addr % self.page_width])
            for addr in base_grads
        }
        # 回查基值与梯度大小无关：读命中但梯度相消为零的地址也要能查到基版值。
        for addr in {a for _op, a in record.read_ops}:
            base_values.setdefault(
                addr,
                float(record.snapshots[addr // self.page_width][addr % self.page_width]),
            )

        result = BackwardResult(
            step_id=step_id,
            base_generation=record.base_generation,
            new_generation=record.new_generation,
            read_grads=dict(sorted(read_grads.items())),
            write_grads=dict(sorted(write_grads.items())),
            base_grads=dict(sorted(base_grads.items())),
            base_values=dict(sorted(base_values.items())),
        )
        record.status = "complete"
        record.result = result
        record.seen_reports = incoming
        record.snapshots = {}  # 基值已查完，释放旧代次页副本
        return result

    # ------------------------------------------------------------------ #
    # 更新器：只接收已完成回传的步骤
    # ------------------------------------------------------------------ #
    def apply_gradients(self, step_id: str) -> dict[int, float]:
        """把已完成回传的基版梯度累加进存储梯度缓冲（float64）。"""
        if step_id in self._rejected:
            raise CredentialError(
                f"步骤 {step_id} 在前向提交时已被拒绝，没有可交接的回传结果",
                step_id=step_id,
            )
        record = self._records.get(step_id)
        if record is None:
            raise CredentialError(f"未知步骤身份 {step_id}", step_id=step_id)
        if record.status == "expired":
            raise ExpiredVersionError(
                f"步骤 {step_id} 的基版代次 {record.base_generation} 已超出开放步预算 "
                f"{self._open_budget} 被回收，回传结果无法交接；请基于当前代次重新提交",
                step_id=step_id,
                base_generation=record.base_generation,
                current_generation=self._generation,
            )
        if record.status != "complete" or record.result is None:
            raise IncompleteBackwardError(
                f"步骤 {step_id} 的回传尚未完成，不能交给更新器",
                step_id=step_id,
            )
        delta = record.result.base_grads
        for addr, grad in delta.items():
            self._grad[addr] = np.float64(self._grad.get(addr, 0.0)) + np.float64(grad)
        record.status = "applied"
        self._order.remove(step_id)
        del self._records[step_id]
        return dict(delta)

    # ------------------------------------------------------------------ #
    # 内部：校验与换算
    # ------------------------------------------------------------------ #
    def _check_base_generation(self, step_id: str, base: int) -> None:
        if base != self._generation:
            raise StaleGenerationError(
                f"步骤 {step_id} 声明的基代次为 {base}，存储当前代次为 {self._generation}；"
                "请基于最新代次重新提交",
                step_id=step_id,
                declared=base,
                current=self._generation,
            )

    def _check_identities(self, step_id: str, ops: list[Operation]) -> None:
        seen_ids: set[str] = set()
        seen_order: set[tuple[int, int]] = set()
        for op in ops:
            if op.id in seen_ids:
                err = OrderingConflictError(
                    f"步骤 {step_id} 中操作身份 {op.id} 重复登记",
                    step_id=step_id,
                    op_id=op.id,
                )
                self._rejected[step_id] = err
                raise err
            seen_ids.add(op.id)
            order_key = (op.sample, op.seq)
            if order_key in seen_order:
                err = OrderingConflictError(
                    f"步骤 {step_id} 出现样本 {op.sample} 序号 {op.seq} 的同序号冲突请求，"
                    "写全序无法确定",
                    step_id=step_id,
                    sample=op.sample,
                    seq=op.seq,
                )
                self._rejected[step_id] = err
                raise err
            seen_order.add(order_key)

    def _resolve_addresses(
        self,
        step_id: str,
        ops: list[Operation],
        view_map: Mapping[str, View],
    ) -> dict[str, int]:
        resolved: dict[str, int] = {}
        for op in ops:
            view = view_map.get(op.view)
            if view is None:
                err = CredentialError(
                    f"步骤 {step_id} 的操作 {op.id} 引用了未知视图 {op.view}",
                    step_id=step_id,
                    op_id=op.id,
                    view=op.view,
                )
                self._rejected[step_id] = err
                raise err
            addr = view.address(op.coords)
            page_id, _offset = self._split(addr)
            if not self._is_owned(page_id):
                err = UnownedAddressError(
                    f"步骤 {step_id} 的操作 {op.id} 经视图 {op.view}{tuple(op.coords)} "
                    f"换算到物理地址 {addr}（页 {page_id}），该地址无归属；整步拒绝并保留代次 "
                    f"{self._generation}",
                    step_id=step_id,
                    op_id=op.id,
                    view=op.view,
                    coords=list(op.coords),
                    address=addr,
                    page=page_id,
                )
                self._rejected[step_id] = err
                raise err
            resolved[op.id] = addr
        return resolved

    def _build_recipes(
        self,
        ops: list[Operation],
        resolved: Mapping[str, int],
    ) -> dict[int, _Recipe]:
        recipes: dict[int, _Recipe] = {}
        writes = sorted(
            (op for op in ops if op.kind != _READ),
            key=lambda op: (op.sample, op.seq),
        )
        for op in writes:
            addr = resolved[op.id]
            recipe = recipes.get(addr)
            if recipe is None:
                recipe = _Recipe(address=addr)
                recipes[addr] = recipe
            entry = (op.sample, op.seq, op.id, op.kind, float(op.value))
            if op.kind == _SET:
                # 替换：淘汰此前保留的基值与全部贡献，之后的累加仍然有效。
                recipe.base_retained = False
                recipe.eliminated.update(op_id for *_, op_id, _k, _v in recipe.surviving)
                recipe.surviving = [entry]
            else:
                recipe.surviving.append(entry)
        return recipes

    def _apply_writes(
        self,
        ops: list[Operation],
        resolved: Mapping[str, int],
        recipes: Mapping[int, _Recipe],
    ) -> None:
        for op in sorted((o for o in ops if o.kind != _READ),
                         key=lambda o: (o.sample, o.seq)):
            addr = resolved[op.id]
            page_id, offset = self._split(addr)
            page = self._materialize(page_id)
            if op.kind == _SET:
                page[offset] = np.float64(op.value)
            else:
                page[offset] = page[offset] + np.float64(op.value)

    def _retire_oldest(self) -> None:
        while len(self._order) > self._open_budget:
            oldest_id = self._order.pop(0)
            record = self._records.pop(oldest_id)
            record.status = "expired"
            record.snapshots = {}
            # 保留轻量存根，使回传得到“已过期”而非“未知身份”的解释。
            self._records[oldest_id] = record
        # 被拒存根同样有界，防止无限增长。
        if len(self._rejected) > self._open_budget:
            for stale_id in list(self._rejected)[:-self._open_budget]:
                del self._rejected[stale_id]

    def _require_open_record(self, step_id: str) -> _Record:
        if step_id in self._rejected:
            raise CredentialError(
                f"步骤 {step_id} 在前向提交时已被拒绝，不存在可回传的代次",
                step_id=step_id,
            )
        record = self._records.get(step_id)
        if record is None:
            raise CredentialError(f"未知步骤身份 {step_id}，无回传凭据可核对",
                                  step_id=step_id)
        if record.status == "expired":
            raise ExpiredVersionError(
                f"步骤 {step_id} 的基版代次 {record.base_generation} 已超出开放步预算 "
                f"{self._open_budget} 被回收；请基于当前代次 {self._generation} 重新提交",
                step_id=step_id,
                base_generation=record.base_generation,
                current_generation=self._generation,
            )
        if record.status == "applied":
            raise CredentialError(
                f"步骤 {step_id} 的回传已交给更新器，不能重复交接",
                step_id=step_id,
            )
        return record

    def _replay_accepted(self, record: _Record, *, replayed: bool = True) -> StepResult:
        return StepResult(
            step_id=record.step_id,
            base_generation=record.base_generation,
            new_generation=record.new_generation,
            reads=dict(record.read_values),
            written_addresses=tuple(sorted(record.targets)),
            values=dict(record.written_values),
            token=record.token,
            replayed=replayed,
        )

    @staticmethod
    def _assert_same_reports(record: _Record, reports: list[GradientReport]) -> None:
        for report in reports:
            payload = (tuple(report.reads), tuple(report.new_values))
            previous = record.seen_reports.get(report.id)
            if previous is None:
                raise CredentialError(
                    f"步骤 {record.step_id} 已完成回传并封存，新回传组 {report.id} "
                    "无法再并入；请核对回传凭据或基于新代次重新提交",
                    step_id=record.step_id,
                    report_id=report.id,
                )
            if previous != payload:
                raise CredentialError(
                    f"步骤 {record.step_id} 已完成回传，回传组 {report.id} 的新内容与旧内容冲突",
                    step_id=record.step_id,
                    report_id=report.id,
                )

    # ------------------------------------------------------------------ #
    # 内部：页与地址
    # ------------------------------------------------------------------ #
    def _split(self, address: int) -> tuple[int, int]:
        if address < 0:
            page_id, offset = (address + 1) // self.page_width - 1, address % self.page_width
        else:
            page_id, offset = divmod(address, self.page_width)
        return page_id, offset

    def _is_owned(self, page_id: int) -> bool:
        return self._owned is None or page_id in self._owned

    def _require_owned(self, page_id: int, address: int) -> None:
        if not self._is_owned(page_id):
            raise UnownedAddressError(
                f"物理地址 {address}（页 {page_id}）无归属",
                address=address,
                page=page_id,
            )

    def _materialize(self, page_id: int) -> np.ndarray:
        page = self._pages.get(page_id)
        if page is None:
            if not self._is_owned(page_id):
                raise UnownedAddressError(
                    f"页 {page_id} 无归属，无法物化", page=page_id
                )
            page = np.zeros(self.page_width, dtype=np.float64)
            self._pages[page_id] = page
        return page

    # ------------------------------------------------------------------ #
    # 内部：输入兼容采集 JSON 的原生容器
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_views(views: Mapping[str, View | Mapping] | None) -> dict[str, View]:
        out: dict[str, View] = {}
        for name, item in (views or {}).items():
            if isinstance(item, View):
                out[name] = item
            else:
                out[name] = View(
                    name=name,
                    base=int(item["base"]),
                    shape=tuple(int(s) for s in item["shape"]),
                    strides=tuple(int(s) for s in item["strides"]),
                )
        return out

    @staticmethod
    def _coerce_operation(item: Operation | Sequence) -> Operation:
        if isinstance(item, Operation):
            return item
        return operation_from_record(item)

    @staticmethod
    def _coerce_report(item: GradientReport | Mapping) -> GradientReport:
        if isinstance(item, GradientReport):
            return item
        return report_from_record(item)


# --------------------------------------------------------------------------- #
# 采集记录适配
# --------------------------------------------------------------------------- #
def view_from_record(descriptor: Mapping, name: str) -> View:
    """从 capture.json 的 views 条目构造 :class:`View`。"""
    return View(
        name=name,
        base=int(descriptor["base"]),
        shape=tuple(int(s) for s in descriptor["shape"]),
        strides=tuple(int(s) for s in descriptor["strides"]),
    )


def operation_from_record(entry: Sequence) -> Operation:
    """从 ``[id, sample, seq, kind, view, coords, value]`` 构造操作。"""
    op_id, sample, seq, kind, view, coords, value = entry
    return Operation(
        id=str(op_id),
        sample=int(sample),
        seq=int(seq),
        kind=str(kind),
        view=str(view),
        coords=tuple(int(c) for c in coords),
        value=None if value is None else float(value),
    )


def report_from_record(entry: Mapping) -> GradientReport:
    """从 upstream 条目构造 :class:`GradientReport`。"""
    return GradientReport(
        id=str(entry["id"]),
        reads=tuple((str(rid), float(g)) for rid, g in entry.get("reads", ())),
        new_values=tuple((int(addr), float(g)) for addr, g in entry.get("new_value", ())),
    )


def build_store_from_records(
    records: Mapping,
    tensors: Mapping[str, np.ndarray],
    *,
    open_steps: int = 8,
) -> MemoryStore:
    """按 capture.json 的 records/tensors 构建存储（含页归属与零值策略）。"""
    page_width = int(records["page_width"])
    pages: dict[int, np.ndarray] = {}
    for page_id_text, tensor_name in records["pages"].items():
        pages[int(page_id_text)] = np.asarray(tensors[tensor_name], dtype=np.float64)
    return MemoryStore(
        page_width=page_width,
        pages=pages,
        generation=int(records["generation"]),
        open_steps=open_steps,
        unlisted_upstream=records.get("unlisted_upstream"),
    )
