"""端到端演示：按 data/capture.json 跑通提交、回传与更新器交接。

用法：python run_demo.py data/capture.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from caseio import load_case
from memop import (
    OperatorError,
    build_store_from_records,
    operation_from_record,
    report_from_record,
    view_from_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="批次读写算子端到端演示")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    records, tensors = load_case(args.path)
    store = build_store_from_records(records, tensors)
    views = {name: view_from_record(desc, name) for name, desc in records["views"].items()}

    print(f"起始代次: {store.generation}")

    # 1) 正常步：原子提交读取与写入。
    step = records["step"]
    ops = [operation_from_record(e) for e in step["operations"]]
    result = store.submit_step(step["id"], base=step["base"], operations=ops, views=views)
    print(f"\n前向步 {result.step_id}: 代次 {result.base_generation} -> {result.new_generation}")
    print("读取批次（步开始镜像）:")
    for op_id, (addr, value) in sorted(result.reads.items()):
        print(f"  {op_id:7s} 物理地址 {addr:3d} = {value}")
    print("写后物理值:")
    for addr, value in sorted(result.values.items()):
        print(f"  地址 {addr:3d} = {value}")

    # 2) 回传：重复 rA 去重，按物理地址归因。
    reports = [report_from_record(e) for e in records["upstream"]]
    backward = store.backward(step["id"], reports, token=result.token)
    print("\n回传归因:")
    print("  读身份梯度:", backward.read_grads)
    print("  写来源梯度:", backward.write_grads)
    print("  基版地址梯度:", backward.base_grads)
    print("  回查基值:", backward.base_values)

    # 3) 更新器只接收已完成的回传。
    applied = store.apply_gradients(step["id"])
    print("\n更新器接收梯度:", applied)

    # 4) 无归属地址整步拒绝，代次保留。
    bad = records["rejected_step"]
    bad_ops = [operation_from_record(e) for e in bad["operations"]]
    try:
        store.submit_step(bad["id"], base=bad["base"], operations=bad_ops, views=views)
    except OperatorError as exc:
        print(f"\n拒绝步 {bad['id']} [{exc.code}]: {exc.message}")
    print(f"拒绝后代次仍为: {store.generation}")


if __name__ == "__main__":
    main()
