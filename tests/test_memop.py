"""批次读写算子的语义测试（基于 data/capture.json 与容量条件）。"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from caseio import load_case
from memop import (
    BackwardResult,
    CredentialError,
    ExpiredVersionError,
    GradientReport,
    IncompleteBackwardError,
    MemoryStore,
    Operation,
    OrderingConflictError,
    StaleGenerationError,
    UnownedAddressError,
    View,
    build_store_from_records,
    operation_from_record,
    report_from_record,
    view_from_record,
)

DATA = Path(__file__).resolve().parents[1] / "data"


class CaptureCaseMixin:
    def build_capture(self, *, open_steps: int = 8):
        records, tensors = load_case(DATA / "capture.json")
        store = build_store_from_records(records, tensors, open_steps=open_steps)
        views = {name: view_from_record(desc, name) for name, desc in records["views"].items()}
        ops = [operation_from_record(entry) for entry in records["step"]["operations"]]
        rejected_ops = [operation_from_record(e) for e in records["rejected_step"]["operations"]]
        reports = [report_from_record(e) for e in records["upstream"]]
        return records, store, views, ops, rejected_ops, reports


class ForwardTests(CaptureCaseMixin, unittest.TestCase):
    def test_reads_see_base_snapshot_and_views_alias_physical_slot(self):
        _records, store, views, ops, _rej, _rep = self.build_capture()
        result = store.submit_step("s11", base=10, operations=ops, views=views)

        self.assertFalse(result.replayed)
        self.assertEqual(result.base_generation, 10)
        self.assertEqual(result.new_generation, 11)
        # a(0,0)=16 -> p3 偏移 1 -> 1；twist(2,0)=18-2=16，同一物理槽。
        self.assertEqual(result.read_batch(), {"read-a": 1.0, "read-b": 1.0})
        self.assertEqual(result.reads["read-a"][0], 16)
        self.assertEqual(result.reads["read-b"][0], 16)

    def test_write_ordering_add_set_add_and_base_value(self):
        _records, store, views, ops, _rej, _rep = self.build_capture()
        result = store.submit_step("s11", base=10, operations=ops, views=views)

        # addr16：1 + 1.5 -> set 8（淘汰基值与 1.5）-> +(-2) = 6
        self.assertEqual(result.values[16], 6.0)
        # addr22：p4 偏移 2 的基值 8 + 2.25 = 10.25
        self.assertEqual(result.values[22], 10.25)
        self.assertEqual(store.get(16), 6.0)
        self.assertEqual(store.get(22), 10.25)
        # 未触及位置保持基版值。
        self.assertEqual(store.get(17), 2.0)
        self.assertEqual(store.get(50), 11.0)

    def test_reads_are_isolated_from_writes_within_step(self):
        store = MemoryStore(
            page_width=4,
            pages={0: np.arange(4, dtype=np.float64)},
            generation=0,
        )
        view = View("v", base=0, shape=(4,), strides=(1,))
        ops = [
            Operation("r0", 1, 0, "read", "v", (0,), None),
            Operation("w0", 2, 0, "set", "v", (0,), 99.0),
            Operation("r1", 3, 0, "read", "v", (0,), None),
        ]
        result = store.submit_step("s", base=0, operations=ops, views={"v": view})
        # 即使 set 在两个读之间按写序执行，两个读都只见到步开始镜像 0。
        self.assertEqual(result.read_batch(), {"r0": 0.0, "r1": 0.0})
        self.assertEqual(store.get(0), 99.0)

    def test_unowned_address_rejects_whole_step_and_keeps_generation(self):
        _records, store, views, ops, rejected_ops, _rep = self.build_capture()
        with self.assertRaises(UnownedAddressError) as ctx:
            store.submit_step("s12", base=10, operations=rejected_ops, views=views)
        self.assertEqual(ctx.exception.code, "unowned_address")
        self.assertEqual(ctx.exception.details["address"], 30)
        # 拒绝后代次与存储不变。
        self.assertEqual(store.generation, 10)
        self.assertEqual(store.get(16), 1.0)
        # 重放被拒步骤给出同一解释，不落写。
        with self.assertRaises(UnownedAddressError):
            store.submit_step("s12", base=10, operations=rejected_ops, views=views)
        self.assertEqual(store.generation, 10)

    def test_stale_base_generation_is_rejected_with_resubmittable_detail(self):
        _records, store, views, ops, _rej, _rep = self.build_capture()
        with self.assertRaises(StaleGenerationError) as ctx:
            store.submit_step("late", base=9, operations=ops, views=views)
        self.assertEqual(ctx.exception.details, {"step_id": "late", "declared": 9, "current": 10})
        self.assertEqual(store.generation, 10)

    def test_replay_same_step_does_not_write_twice(self):
        _records, store, views, ops, _rej, _rep = self.build_capture()
        first = store.submit_step("s11", base=10, operations=ops, views=views)
        second = store.submit_step("s11", base=10, operations=ops, views=views)
        self.assertTrue(second.replayed)
        self.assertEqual(second.token, first.token)
        self.assertEqual(second.new_generation, 11)
        # 重放不推进代次、不再次累加。
        self.assertEqual(store.generation, 11)
        self.assertEqual(store.get(16), 6.0)
        self.assertEqual(store.get(22), 10.25)

    def test_same_sample_seq_conflict_and_duplicate_op_ids_rejected(self):
        store = MemoryStore(page_width=4, pages={0: np.zeros(4)}, generation=0)
        view = View("v", base=0, shape=(2,), strides=(1,))
        conflicting = [
            Operation("w1", 1, 0, "add", "v", (0,), 1.0),
            Operation("w2", 1, 0, "add", "v", (1,), 2.0),
        ]
        with self.assertRaises(OrderingConflictError):
            store.submit_step("c", base=0, operations=conflicting, views={"v": view})
        self.assertEqual(store.generation, 0)

        duplicate_ids = [
            Operation("w", 1, 0, "add", "v", (0,), 1.0),
            Operation("w", 1, 1, "add", "v", (1,), 2.0),
        ]
        with self.assertRaises(OrderingConflictError):
            store.submit_step("d", base=0, operations=duplicate_ids, views={"v": view})
        self.assertEqual(store.generation, 0)

    def test_negative_strides_and_transposed_view(self):
        # twist: base 18, strides (-1, 5)，逐点核对负步幅换算。
        twist = View("twist", 18, (3, 2), (-1, 5))
        expected = {(0, 0): 18, (0, 1): 23, (1, 0): 17, (1, 1): 22, (2, 0): 16, (2, 1): 21}
        for coords, addr in expected.items():
            self.assertEqual(twist.address(coords), addr)
        with self.assertRaises(CredentialError):
            twist.address((3, 0))


class BackwardTests(CaptureCaseMixin, unittest.TestCase):
    def test_capture_attribution_end_to_end(self):
        _records, store, views, ops, _rej, reports = self.build_capture()
        result = store.submit_step("s11", base=10, operations=ops, views=views)
        backward = store.backward("s11", reports, token=result.token)

        self.assertIsInstance(backward, BackwardResult)
        # 两份相同的 rA 是重复递交，去重后每个读身份只记一次。
        self.assertEqual(backward.read_grads, {"read-a": 7.0, "read-b": 3.0})
        # 两个有效读命中同一物理地址 16，相加；写路径 5 不回基值（set 未保留基值）。
        # addr22 基值保留，rB 的 -4 同时分给基值与 add-d。
        self.assertEqual(backward.base_grads, {16: 10.0, 22: -4.0})
        # 被 set 淘汰的 add-a 收不到写路径梯度；set-b 与后续 add-c 有效。
        self.assertEqual(backward.write_grads["add-a"], 0.0)
        self.assertEqual(backward.write_grads["set-b"], 5.0)
        self.assertEqual(backward.write_grads["add-c"], 5.0)
        self.assertEqual(backward.write_grads["add-d"], -4.0)
        # 回查的基值来自步开始代次。
        self.assertEqual(backward.base_values[16], 1.0)
        self.assertEqual(backward.base_values[22], 8.0)

    def test_backward_requires_matching_token(self):
        _records, store, views, ops, _rej, reports = self.build_capture()
        result = store.submit_step("s11", base=10, operations=ops, views=views)
        with self.assertRaises(CredentialError):
            store.backward("s11", reports, token="forged-token")

    def test_backward_is_idempotent_and_dedupes_shards(self):
        _records, store, views, ops, _rej, reports = self.build_capture()
        result = store.submit_step("s11", base=10, operations=ops, views=views)
        first = store.backward("s11", reports, token=result.token)
        # 同内容完整重放：结果一致，不重复记账。
        second = store.backward("s11", reports, token=result.token)
        self.assertEqual(second.base_grads, first.base_grads)
        self.assertEqual(second.write_grads, first.write_grads)

    def test_sharded_reports_merge_into_same_attribution(self):
        _records, store, views, ops, _rej, _reports = self.build_capture()
        result = store.submit_step("s11", base=10, operations=ops, views=views)
        shards = [
            GradientReport("shard-1", reads=(("read-a", 7.0),), new_values=((16, 5.0),)),
            GradientReport("shard-2", reads=(("read-b", 3.0),), new_values=((22, -4.0),)),
        ]
        backward = store.backward("s11", shards, token=result.token)
        self.assertEqual(backward.read_grads, {"read-a": 7.0, "read-b": 3.0})
        self.assertEqual(backward.base_grads, {16: 10.0, 22: -4.0})

    def test_report_group_id_is_idempotency_key_while_distinct_groups_sum(self):
        # 同组 ID 的再次递交是重放 -> 去重；不同组 ID 即便引用同一读身份也是
        # 不同有效递交 -> 在物理地址上相加。
        _records, store, views, ops, _rej, _reports = self.build_capture()
        result = store.submit_step("s11", base=10, operations=ops, views=views)
        backward = store.backward(
            "s11",
            [
                GradientReport("g1", reads=(("read-a", 7.0),),
                               new_values=((16, 0.0), (22, 0.0))),
                GradientReport("g1", reads=(("read-a", 7.0),),
                               new_values=((16, 0.0), (22, 0.0))),
                GradientReport("g2", reads=(("read-a", 2.0), ("read-b", 3.0),),
                               new_values=((16, 0.0), (22, 0.0))),
            ],
            token=result.token,
        )
        self.assertEqual(backward.read_grads, {"read-a": 9.0, "read-b": 3.0})
        self.assertEqual(backward.base_grads, {16: 12.0})

    def test_zero_policy_treats_unlisted_write_targets_as_zero(self):
        # capture.json 声明 unlisted_upstream=zero：只覆盖基值被保留的 addr22，
        # 未列出的 addr16 按零导数完成覆盖。
        _records, store, views, ops, _rej, _reports = self.build_capture()
        result = store.submit_step("s11", base=10, operations=ops, views=views)
        backward = store.backward(
            "s11",
            [GradientReport("partial", new_values=((22, -4.0),))],
            token=result.token,
        )
        self.assertTrue(backward.complete)
        # addr16 未列出按零：set-b/add-c 写梯度为零，基值已被 set 淘汰本就不回基版。
        self.assertEqual(backward.write_grads["set-b"], 0.0)
        self.assertEqual(backward.write_grads["add-c"], 0.0)
        self.assertEqual(backward.base_grads, {22: -4.0})
        self.assertEqual(backward.base_values[22], 8.0)

    def test_duplicate_report_group_with_changed_payload_rejected(self):
        _records, store, views, ops, _rej, _reports = self.build_capture()
        result = store.submit_step("s11", base=10, operations=ops, views=views)
        good = [GradientReport("g", reads=(("read-a", 7.0),), new_values=((16, 5.0), (22, -4.0),))]
        store.backward("s11", good, token=result.token)
        bad = [GradientReport("g", reads=(("read-a", 8.0),), new_values=((16, 5.0), (22, -4.0),))]
        with self.assertRaises(CredentialError):
            store.backward("s11", bad, token=result.token)

    def test_new_report_group_after_completion_rejected(self):
        _records, store, views, ops, _rej, _reports = self.build_capture()
        result = store.submit_step("s11", base=10, operations=ops, views=views)
        store.backward(
            "s11",
            [GradientReport("g", new_values=((16, 0.0), (22, 0.0)))],
            token=result.token,
        )
        with self.assertRaises(CredentialError):
            store.backward(
                "s11",
                [GradientReport("late-arrival", new_values=((16, 1.0),))],
                token=result.token,
            )

    def test_rejected_step_has_no_backward_or_updater_path(self):
        _records, store, views, _ops, rejected_ops, _reports = self.build_capture()
        with self.assertRaises(UnownedAddressError):
            store.submit_step("s12", base=10, operations=rejected_ops, views=views)
        with self.assertRaises(CredentialError):
            store.backward("s12", [], token="whatever")
        with self.assertRaises(CredentialError):
            store.apply_gradients("s12")

    def test_unknown_read_identity_and_foreign_address_rejected(self):
        _records, store, views, ops, _rej, _reports = self.build_capture()
        result = store.submit_step("s11", base=10, operations=ops, views=views)
        with self.assertRaises(CredentialError):
            store.backward(
                "s11",
                [GradientReport("x", reads=(("nope", 1.0),), new_values=((16, 0.0), (22, 0.0),))],
                token=result.token,
            )
        with self.assertRaises(CredentialError):
            store.backward(
                "s11",
                [GradientReport("x", reads=(), new_values=((17, 1.0),))],
                token=result.token,
            )

    def test_incomplete_backward_blocked_until_explicit_zero_coverage(self):
        store = MemoryStore(
            page_width=4,
            pages={0: np.ones(4, dtype=np.float64)},
            generation=0,
            unlisted_upstream=None,
        )
        view = View("v", base=0, shape=(2,), strides=(1,))
        ops = [
            Operation("w1", 1, 0, "set", "v", (0,), 5.0),
            Operation("w2", 2, 0, "set", "v", (1,), 6.0),
        ]
        result = store.submit_step("s", base=0, operations=ops, views={"v": view})
        with self.assertRaises(IncompleteBackwardError) as ctx:
            store.backward("s", [GradientReport("p", new_values=((0, 2.0),))], token=result.token)
        self.assertEqual(ctx.exception.details["missing_addresses"], [1])
        # 未完成的回传不能交给更新器。
        with self.assertRaises(IncompleteBackwardError):
            store.apply_gradients("s")
        # 显式声明零导数后完成覆盖。
        backward = store.backward(
            "s",
            [GradientReport("p", new_values=((0, 2.0), (1, 0.0)))],
            token=result.token,
        )
        self.assertTrue(backward.complete)

    def test_set_eliminates_base_gradient_but_later_adds_still_receive_it(self):
        store = MemoryStore(page_width=4, pages={0: np.full(4, 3.0)}, generation=0)
        view = View("v", base=0, shape=(1,), strides=(1,))
        ops = [
            Operation("a1", 1, 0, "add", "v", (0,), 1.0),
            Operation("s1", 2, 0, "set", "v", (0,), 9.0),
            Operation("a2", 3, 0, "add", "v", (0,), 2.0),
        ]
        result = store.submit_step("s", base=0, operations=ops, views={"v": view})
        self.assertEqual(result.values[0], 11.0)  # set 9 + add 2
        backward = store.backward(
            "s", [GradientReport("g", new_values=((0, 4.0),))], token=result.token
        )
        self.assertEqual(backward.write_grads, {"a1": 0.0, "s1": 4.0, "a2": 4.0})
        # 基值被 set 淘汰：写路径梯度不回基版。
        self.assertEqual(backward.base_grads, {})

    def test_gradients_accumulate_in_float64_for_updater(self):
        store = MemoryStore(page_width=4, pages={0: np.zeros(4)}, generation=0)
        view = View("v", base=0, shape=(1,), strides=(1,))
        results = []
        for step_id in ("s1", "s2"):
            ops = [Operation("w", 1, 0, "add", "v", (0,), 1e-10)]
            results.append((step_id, store.submit_step(
                step_id, base=store.generation, operations=ops, views={"v": view})))
        for step_id, result in results:
            store.backward(step_id,
                           [GradientReport("g", new_values=((0, 1e-10),))],
                           token=result.token)
        store.apply_gradients("s1")
        store.apply_gradients("s2")
        self.assertEqual(store.gradients[0], 2e-10)
        self.assertIsInstance(store.gradients[0], float)
        # 已交接的步骤不能重复交接。
        with self.assertRaises(CredentialError):
            store.apply_gradients("s1")


class VersioningAndCapacityTests(CaptureCaseMixin, unittest.TestCase):
    def test_old_generation_expires_but_storage_is_preserved(self):
        _records, store, views, ops, _rej, _rep = self.build_capture(open_steps=2)
        first = store.submit_step("s11", base=10, operations=ops, views=views)
        # 再提交两个空步把 s11 挤出开放窗口。
        for step_id in ("s13", "s14"):
            store.submit_step(step_id, base=store.generation, operations=[], views=views)
        self.assertNotIn("s11", store.open_step_ids)
        with self.assertRaises(ExpiredVersionError) as ctx:
            store.backward("s11", [], token=first.token)
        self.assertEqual(ctx.exception.details["base_generation"], 10)
        with self.assertRaises(ExpiredVersionError):
            store.submit_step("s11", base=10, operations=ops, views=views)
        with self.assertRaises(ExpiredVersionError):
            store.apply_gradients("s11")
        # 存储值不受过期影响。
        self.assertEqual(store.get(16), 6.0)
        self.assertEqual(store.generation, 13)

    def test_sparse_scratch_stays_within_capacity_budget(self):
        capacity = json.loads((DATA / "capacity.json").read_text(encoding="utf-8"))
        width = capacity["page_width"]
        store = MemoryStore(
            page_width=width,
            generation=0,
            open_steps=capacity["open_forward_steps"],
            owned_pages=capacity["owned_pages"],
        )
        # 每页一条写，逐步触及 touched_pages_per_step 个不同物理页。
        rng = np.random.default_rng(capacity["seed"])
        for step_index in range(capacity["open_forward_steps"]):
            page_ids = rng.choice(capacity["owned_pages"],
                                  size=capacity["touched_pages_per_step"], replace=False)
            # 每个操作一个单元素视图，逻辑坐标 0 落到抽中页的页首。
            per_op_views = {
                f"v{i}": View(f"v{i}", int(page_id) * width, (1,), (1,))
                for i, page_id in enumerate(page_ids)
            }
            ops = [
                Operation(f"w{i}", i, 0, "add", f"v{i}", (0,), 1.0)
                for i, page_id in enumerate(page_ids)
            ]
            store.submit_step(f"s{step_index}", base=step_index,
                              operations=ops, views=per_op_views)

        # 工作区只有被触及页的快照与待交接缓冲，远低于 peak_scratch_bytes。
        self.assertLessEqual(store.scratch_bytes, capacity["peak_scratch_bytes"])
        # 只物化被触及的页（跨步可能复用，故为上限），从未全量复制记忆。
        self.assertLessEqual(
            len(store.materialized_pages()),
            capacity["open_forward_steps"] * capacity["touched_pages_per_step"],
        )
        self.assertLess(len(store.materialized_pages()), capacity["owned_pages"])

    def test_lazy_owned_pages_read_as_zero_without_full_materialization(self):
        store = MemoryStore(page_width=8, generation=0, owned_pages=20000)
        self.assertEqual(store.materialized_pages(), ())
        view = View("v", base=8 * 1234 + 2, shape=(1,), strides=(1,))
        result = store.submit_step(
            "s", base=0,
            operations=[Operation("r", 1, 0, "read", "v", (0,), None),
                        Operation("w", 2, 0, "set", "v", (0,), 7.0)],
            views={"v": view},
        )
        self.assertEqual(result.read_batch(), {"r": 0.0})
        self.assertEqual(store.get(8 * 1234 + 2), 7.0)
        self.assertEqual(store.materialized_pages(), (1234,))


if __name__ == "__main__":
    unittest.main()
