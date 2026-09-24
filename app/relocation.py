"""已保存配平方案的失效槽位应急迁移建议(只读, 不改变方案与保存格式)。

问题模型
--------
运行中的转子某个启用槽位突然失效(杯位损伤/卡滞), 该槽位**不得再占用**。
为尽快恢复运行, 不重新称量、不重新求解, 而是基于已保存方案中的标称质量、
槽位角度与半径, 把**尽可能少**的试管搬到其他启用空槽, 使迁移后的一阶
不平衡量满足

    |U₁| = R·|Σ m_k·e^{iθ_k}| ≤ 限值              (g·mm)

约束
----
- 锁定试管不得移动(保持原槽);
- 失效槽位不得占用: 若失效槽位上原有试管, 该管必须搬走(计入移动数),
  若该管为锁定试管则任何移动上限内都无可行摆放;
- 禁用槽位与失效槽位不能作为目标槽, 每槽至多一支试管;
- 移动管数不超过用户给定上限(0~3)。

精确搜索(有界枚举)
------------------
为控制搜索规模, 仅接受至多 10 支已放置试管、16 个启用槽位、移动上限 0~3
的方案。设移动管集合为 M(|M| = m): 未移动的试管(锁定管 + 留守管)槽位
固定, 候选目标槽为"启用 ∧ 非失效 ∧ 未被留守占用"的槽位; M 中各管目标槽
互不相同且不等于其原槽(否则应计入更小的 m)。按 m = 0,1,… 递增枚举 M 的
全部组合与目标槽的全部单射指派, 第一个出现可行摆放的 m 即**最少移动数**;
在该 m 下取**最终 |U₁| 最小**者, 并列(相对容差 1e-9)时按**管号升序的
目标槽位序列**(字典序)裁决。组合数上界 C(10,3)·P(15,3) ≈ 3.3×10⁵, 每次
评估 O(1), 瞬时完成; 同时统计达到同一(最少移动数, 最小 |U₁|)的摆放种数。

无可行摆放
----------
移动上限内没有任何摆放满足限值时明确报告无可行摆放, 并给出移动上限内
可达到的最小 |U₁|(及其移动数)供参考; 失效槽位被锁定试管占用时直接判定
无可行摆放, 不进入搜索。

拒绝条件(HTTP 400)
------------------
- 方案没有已放置试管; 已放置试管数超过 10; 启用槽位数超过 16;
- 移动上限不是 0~3 的整数; 限值为负或非有限数;
- 失效槽位不存在, 或在方案中已是禁用槽位;
- 摆放明细中试管编号/槽位重复、质量非正、缺少槽位角度,
  或占用了禁用槽/几何之外的槽位。
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Optional

# 搜索规模上限: 已放置试管 ≤ 10, 启用槽位 ≤ 16, 移动上限 0~3
MAX_TUBES = 10
MAX_ENABLED_SLOTS = 16
MAX_MOVES = 3
# 与 app/compare.py 一致的数值容差
_NUM_TOL = 1e-9


def _u1_block(u: complex) -> dict[str, Any]:
    return {
        "u1_mag_g_mm": abs(u),
        "u1_angle_deg": math.degrees(math.atan2(u.imag, u.real)) % 360.0,
        "u1_x_g_mm": u.real,
        "u1_y_g_mm": u.imag,
    }


def plan_emergency_relocation(plan: dict[str, Any], failed_slot: int,
                              u1_limit_g_mm: float,
                              max_moves: int) -> dict[str, Any]:
    """对已保存方案生成失效槽位应急迁移建议(只读)。

    参数
    ----
    plan: GET /api/plan/{id} 返回结构的方案(含 id/name/request/result);
        仅读取, 不修改。
    failed_slot: 新失效槽位槽号; 必须是方案中的启用槽位, 迁移后不得占用。
    u1_limit_g_mm: 一阶不平衡量限值(g·mm), 迁移后 |U₁| 不得超过该值。
    max_moves: 最多可移动管数(0~3); 锁定试管不得移动, 也不计入移动数。

    返回
    ----
    含可行性、逐管原槽/目标槽、移动数、最终 |U₁| 与搜索统计的字典;
    无可行摆放时 feasible=False 并给出明确原因及上限内可达最小 |U₁|。
    """
    plan_id = plan.get("id")

    # ---- 输入校验 ----
    if isinstance(failed_slot, bool):
        raise ValueError("失效槽位必须为整数槽号。")
    try:
        fs = int(failed_slot)
    except (TypeError, ValueError):
        raise ValueError("失效槽位必须为整数槽号。") from None
    if fs != failed_slot:
        raise ValueError("失效槽位必须为整数槽号。")
    try:
        limit = float(u1_limit_g_mm)
    except (TypeError, ValueError):
        raise ValueError("一阶不平衡量限值必须为有限数值。") from None
    if not math.isfinite(limit):
        raise ValueError("一阶不平衡量限值必须为有限数值。")
    if limit < 0.0:
        raise ValueError("一阶不平衡量限值不能为负。")
    if isinstance(max_moves, bool):
        raise ValueError(f"最多可移动管数必须为 0~{MAX_MOVES} 的整数。")
    try:
        moves_cap = int(max_moves)
    except (TypeError, ValueError):
        raise ValueError(
            f"最多可移动管数必须为 0~{MAX_MOVES} 的整数。"
        ) from None
    if moves_cap != max_moves:
        raise ValueError(f"最多可移动管数必须为 0~{MAX_MOVES} 的整数。")
    if not (0 <= moves_cap <= MAX_MOVES):
        raise ValueError(
            f"最多可移动管数必须在 0~{MAX_MOVES} 之间(当前 {moves_cap})。"
        )

    result = plan.get("result") or {}
    cfg = result.get("config") or {}
    radius = cfg.get("radius_mm")
    if radius is None:
        radius = (plan.get("request") or {}).get("radius_mm", 100.0)
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("方案中的转子半径必须为正数。")

    # 槽位几何: 优先取求解结果中的 slot_state, 缺失时回退到请求中的 slots
    geo_src = (result.get("slot_state")
               or (plan.get("request") or {}).get("slots") or [])
    geometry: dict[int, dict[str, Any]] = {}
    for s in geo_src:
        raw_idx = s.get("slot", s.get("index"))
        if raw_idx is None:
            raise ValueError("方案的槽位几何信息缺少槽号。")
        idx = int(raw_idx)
        angle = s.get("angle")
        if angle is None:
            raise ValueError(f"槽位 {idx} 缺少角度, 无法计算不平衡量。")
        geometry[idx] = {"angle": float(angle),
                         "enabled": bool(s.get("enabled", True))}
    if not geometry:
        raise ValueError("方案缺少槽位几何信息。")

    enabled_slots = sorted(i for i, g in geometry.items() if g["enabled"])
    if len(enabled_slots) > MAX_ENABLED_SLOTS:
        raise ValueError(
            f"方案 {plan_id} 有 {len(enabled_slots)} 个启用槽位, 超过应急迁移"
            f"上限 {MAX_ENABLED_SLOTS} 个(为控制搜索规模), 明确拒绝。"
        )
    if fs not in geometry:
        raise ValueError(f"失效槽位 {fs} 在方案 {plan_id} 的槽位几何中不存在。")
    if not geometry[fs]["enabled"]:
        raise ValueError(
            f"失效槽位 {fs} 在方案 {plan_id} 中已是禁用槽位, 无需应急迁移。"
        )

    used = [p for p in (result.get("placements") or [])
            if p.get("slot") is not None]
    if not used:
        raise ValueError(
            f"方案 {plan_id} 没有已放置的试管, 无法进行应急迁移。"
        )
    if len(used) > MAX_TUBES:
        raise ValueError(
            f"方案 {plan_id} 有 {len(used)} 支已放置试管, 超过应急迁移上限 "
            f"{MAX_TUBES} 支(为控制搜索规模), 明确拒绝。"
        )

    tubes: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_slots: set[int] = set()
    for p in used:
        tid = int(p["tube_id"])
        if tid in seen_ids:
            raise ValueError(f"摆放明细中试管 {tid} 重复。")
        seen_ids.add(tid)
        slot = int(p["slot"])
        if slot in seen_slots:
            raise ValueError(f"摆放明细中槽位 {slot} 被重复占用。")
        seen_slots.add(slot)
        if slot not in geometry:
            raise ValueError(
                f"试管 {tid} 所在槽位 {slot} 不在方案槽位几何中。"
            )
        if not geometry[slot]["enabled"]:
            raise ValueError(
                f"试管 {tid} 所在槽位 {slot} 在方案中为禁用槽位, "
                "摆放明细与槽位几何矛盾。"
            )
        mass = float(p["mass"])
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"试管 {tid} 的标称质量必须为正数。")
        tubes.append({
            "tube_id": tid,
            "slot": slot,
            "mass": mass,
            "locked": bool(p.get("locked", False)),
        })
    # 按试管编号升序: 并列裁决的目标槽位序列以此为依据
    tubes.sort(key=lambda t: t["tube_id"])
    sorted_ids = [t["tube_id"] for t in tubes]
    orig_slot = {t["tube_id"]: t["slot"] for t in tubes}
    movable_ids = [t["tube_id"] for t in tubes if not t["locked"]]

    occupant = next((t for t in tubes if t["slot"] == fs), None)
    locked_occupant = occupant is not None and occupant["locked"]

    # 各管在各启用槽的质径积向量 v = R·m·e^{iθ} (g·mm)
    vec: dict[tuple[int, int], complex] = {}
    for t in tubes:
        tid, m = t["tube_id"], t["mass"]
        for s in enabled_slots:
            a = math.radians(geometry[s]["angle"])
            vec[(tid, s)] = complex(m * radius * math.cos(a),
                                    m * radius * math.sin(a))
    orig_u = sum((vec[(t["tube_id"], t["slot"])] for t in tubes), 0j)

    # ---- 有界精确枚举: 移动数递增, 同级取 |U₁| 最小, 并列按目标槽位序列 ----
    limit_tol = _NUM_TOL * max(1.0, limit)
    evaluated = 0
    best_any: Optional[tuple] = None   # (mag, moves, seq), 不限值的最优(参考)
    chosen: Optional[tuple] = None     # (seq, u, final_slots)
    chosen_moves = 0
    chosen_count = 0
    if not locked_occupant:
        occ_id = occupant["tube_id"] if occupant is not None else None
        max_m = min(moves_cap, len(movable_ids))
        for m in range(max_m + 1):
            best_mag: Optional[float] = None
            best_entry: Optional[tuple] = None  # (seq, u, final_slots)
            level_count = 0
            for combo in itertools.combinations(movable_ids, m):
                # 失效槽位上的试管必须搬走(占用移动名额)
                if occ_id is not None and occ_id not in combo:
                    continue
                stay_u = orig_u
                for tid in combo:
                    stay_u -= vec[(tid, orig_slot[tid])]
                stay_slots = {orig_slot[tid] for tid in orig_slot
                              if tid not in combo}
                targets = [s for s in enabled_slots
                           if s != fs and s not in stay_slots]
                if len(targets) < m:
                    continue
                for assign in itertools.permutations(targets, m):
                    # 移动的每支管必须真正换槽(否则应计入更小的 m)
                    if any(assign[i] == orig_slot[combo[i]]
                           for i in range(m)):
                        continue
                    evaluated += 1
                    u = stay_u
                    for i, tid in enumerate(combo):
                        u += vec[(tid, assign[i])]
                    mag = abs(u)
                    final_slots = dict(orig_slot)
                    final_slots.update(zip(combo, assign))
                    seq = tuple(final_slots[tid] for tid in sorted_ids)
                    if best_any is None or (mag, m, seq) < best_any:
                        best_any = (mag, m, seq)
                    if mag > limit + limit_tol:
                        continue
                    tol = _NUM_TOL * max(1.0, best_mag) if best_mag is not None else 0.0
                    if best_mag is None or mag < best_mag - tol:
                        best_mag = mag
                        best_entry = (seq, u, final_slots)
                        level_count = 1
                    elif mag <= best_mag + tol:
                        level_count += 1
                        if seq < best_entry[0]:
                            best_entry = (seq, u, final_slots)
            if best_entry is not None:
                chosen, chosen_moves, chosen_count = best_entry, m, level_count
                break

    # ---- 结果组装 ----
    tube_rows: list[dict[str, Any]] = []
    for t in tubes:
        tid = t["tube_id"]
        orig = t["slot"]
        tgt = chosen[2][tid] if chosen is not None else None
        tube_rows.append({
            "tube_id": tid,
            "mass_g": t["mass"],
            "locked": t["locked"],
            "original_slot": orig,
            "original_angle_deg": geometry[orig]["angle"],
            "target_slot": tgt,
            "target_angle_deg": (geometry[tgt]["angle"]
                                 if tgt is not None else None),
            "moved": tgt is not None and tgt != orig,
        })
    moves = [
        {
            "tube_id": r["tube_id"],
            "from_slot": r["original_slot"],
            "from_angle_deg": r["original_angle_deg"],
            "to_slot": r["target_slot"],
            "to_angle_deg": r["target_angle_deg"],
        }
        for r in tube_rows if r["moved"]
    ]

    out: dict[str, Any] = {
        "method": (
            "有界精确枚举: 按移动数 0..K 递增枚举移动管组合与目标槽单射指派"
            "(锁定试管不动、失效槽位禁用), 首个出现可行摆放的移动数即最少"
            "移动数; 该移动数下取最终 |U₁| 最小, 并列按管号升序的目标槽位"
            "序列裁决"
        ),
        "failed_slot": fs,
        "failed_slot_occupied": occupant is not None,
        "failed_slot_occupant": (occupant["tube_id"]
                                 if occupant is not None else None),
        "u1_limit_g_mm": limit,
        "max_moves": moves_cap,
        "original_u1": _u1_block(orig_u),
        "tube_count": len(tubes),
        "enabled_slot_count": len(enabled_slots),
        "radius_mm": radius,
        "assignments_evaluated": evaluated,
        "tubes": tube_rows,
        "plan": {
            "id": plan.get("id"),
            "name": plan.get("name"),
            "created_at": plan.get("created_at"),
        },
        "note": "失效槽位应急迁移建议为只读结果, 未修改任何已保存方案。",
    }

    if chosen is not None:
        u = chosen[1]
        mag = abs(u)
        out.update({
            "feasible": True,
            "reason": None,
            "move_count": chosen_moves,
            "moves": moves,
            "moved_tube_ids": [r["tube_id"] for r in tube_rows if r["moved"]],
            "final_u1": _u1_block(u),
            "within_limit": True,
            "margin_g_mm": max(0.0, limit - mag),
            "excess_g_mm": 0.0,
            "improvement_g_mm": abs(orig_u) - mag,
            "optimal_assignment_count": chosen_count,
            "tie_broken": chosen_count > 1,
            "best_achievable": None,
        })
        return out

    if locked_occupant:
        reason = (
            f"失效槽位 {fs} 被锁定试管 {occupant['tube_id']} 占用: "
            "锁定试管不得移动, 失效槽位不得占用, 任何移动上限内都无可行摆放。"
        )
    elif best_any is None:
        reason = (
            f"移动上限 {moves_cap} 支内不存在任何满足约束的摆放"
            f"(失效槽位 {fs} 必须腾空, 但没有可移动的试管或可用的目标槽), "
            "无可行摆放。"
        )
    else:
        reason = (
            f"移动上限 {moves_cap} 支内可达到的最小 |U₁| 为 "
            f"{best_any[0]:.6g} g·mm(移动 {best_any[1]} 支), 仍超过限值 "
            f"{limit:g} g·mm, 无可行摆放。"
        )
    out.update({
        "feasible": False,
        "reason": reason,
        "move_count": None,
        "moves": [],
        "moved_tube_ids": [],
        "final_u1": None,
        "within_limit": False,
        "margin_g_mm": None,
        "excess_g_mm": None,
        "improvement_g_mm": None,
        "optimal_assignment_count": 0,
        "tie_broken": False,
        "best_achievable": (
            {
                "u1_mag_g_mm": best_any[0],
                "move_count": best_any[1],
                "target_slot_sequence": list(best_any[2]),
            }
            if best_any is not None else None
        ),
    })
    return out
