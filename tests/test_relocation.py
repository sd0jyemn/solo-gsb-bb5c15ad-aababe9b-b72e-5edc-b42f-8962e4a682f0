"""失效槽位应急迁移测试: python -m unittest discover -s tests

核心断言: 有界精确枚举求得的最少移动数、最终 |U₁| 最小值与目标槽位序列,
必须与独立暴力枚举一致(含并列时按管号升序的目标槽位序列裁决); 锁定试管
不得移动、失效槽位不得占用、禁用槽位不能作为目标槽; 无可行摆放时明确
报告并给出上限内可达最小 |U₁|; 超过 10 支管、16 个启用槽位、移动上限
越界、非法失效槽位等必须拒绝; 规划过程不修改入参(只读)。
"""
import copy
import itertools
import math
import random
import unittest

from app.relocation import plan_emergency_relocation


def _plan(pid, placements, n_slots=12, disabled=(), radius=100.0, locked=(),
          name=None, with_slot_state=True):
    """placements: {槽号: (试管编号, 质量)}; locked: 锁定试管编号集合。"""
    step = 360.0 / n_slots
    slots = [{"index": i, "angle": i * step, "enabled": i not in disabled}
             for i in range(n_slots)]
    result = {
        "config": {"radius_mm": radius},
        "placements": [
            {"tube_id": t, "slot": s, "mass": m, "angle": s * step,
             "locked": t in locked}
            for s, (t, m) in placements.items()
        ],
    }
    if with_slot_state:
        occ = {s: (t, m) for s, (t, m) in placements.items()}
        result["slot_state"] = [
            {"slot": s["index"], "angle": s["angle"], "enabled": s["enabled"],
             "tube_id": occ.get(s["index"], (None,))[0],
             "locked": (occ[s["index"]][0] in locked
                        if s["index"] in occ else False)}
            for s in slots
        ]
    return {
        "id": pid,
        "name": name or f"方案{pid}",
        "created_at": "2026-09-24T00:00:00+00:00",
        "request": {"slot_count": n_slots, "slots": slots, "radius_mm": radius},
        "result": result,
    }


def _brute(plan, failed_slot, limit, max_moves):
    """独立暴力参考: (feasible, moves, mag, seq, count); U₁ 每次从零复算。"""
    res = plan["result"]
    req = plan["request"]
    radius = float(res["config"]["radius_mm"])
    geo = {int(s["index"]): (float(s["angle"]), bool(s["enabled"]))
           for s in req["slots"]}
    enabled = sorted(i for i, (_, e) in geo.items() if e)
    tubes = [p for p in res["placements"] if p["slot"] is not None]
    ids = sorted(int(p["tube_id"]) for p in tubes)
    mass = {int(p["tube_id"]): float(p["mass"]) for p in tubes}
    orig = {int(p["tube_id"]): int(p["slot"]) for p in tubes}
    locked = {int(p["tube_id"]) for p in tubes if p.get("locked")}
    movable = [i for i in ids if i not in locked]
    occ = next((i for i in ids if orig[i] == failed_slot), None)
    if occ is not None and occ in locked:
        return (False, None, None, None, 0)

    def mag_of(final):
        ux = uy = 0.0
        for tid, s in final.items():
            a = math.radians(geo[s][0])
            ux += mass[tid] * radius * math.cos(a)
            uy += mass[tid] * radius * math.sin(a)
        return math.hypot(ux, uy)

    tol = 1e-9 * max(1.0, limit)
    for m in range(0, min(max_moves, len(movable)) + 1):
        best_mag = None
        best_seq = None
        count = 0
        for combo in itertools.combinations(movable, m):
            if occ is not None and occ not in combo:
                continue
            stay = {t: s for t, s in orig.items() if t not in combo}
            targets = [s for s in enabled
                       if s != failed_slot and s not in stay.values()]
            for assign in itertools.permutations(targets, m):
                if any(assign[i] == orig[combo[i]] for i in range(m)):
                    continue
                final = dict(stay)
                final.update(zip(combo, assign))
                mg = mag_of(final)
                if mg > limit + tol:
                    continue
                seq = tuple(final[t] for t in ids)
                if best_mag is None or mg < best_mag - 1e-9 * max(1.0, best_mag):
                    best_mag, best_seq, count = mg, seq, 1
                elif mg <= best_mag + 1e-9 * max(1.0, best_mag):
                    count += 1
                    if seq < best_seq:
                        best_seq = seq
        if best_seq is not None:
            return (True, m, best_mag, best_seq, count)
    return (False, None, None, None, 0)


class TestEmergencyRelocation(unittest.TestCase):
    def test_no_move_needed(self):
        # 失效槽空置且当前已满足限值: 0 支移动
        p = _plan(1, {0: (1, 50.0), 6: (2, 50.0)})
        r = plan_emergency_relocation(p, failed_slot=3,
                                      u1_limit_g_mm=100.0, max_moves=3)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["move_count"], 0)
        self.assertEqual(r["moves"], [])
        self.assertEqual(r["moved_tube_ids"], [])
        self.assertAlmostEqual(r["final_u1"]["u1_mag_g_mm"], 0.0, places=9)
        self.assertAlmostEqual(r["original_u1"]["u1_mag_g_mm"], 0.0, places=9)
        self.assertTrue(r["within_limit"])
        self.assertAlmostEqual(r["margin_g_mm"], 100.0, places=9)
        self.assertEqual(r["optimal_assignment_count"], 1)
        self.assertFalse(r["tie_broken"])
        self.assertFalse(r["failed_slot_occupied"])
        self.assertIsNone(r["failed_slot_occupant"])
        self.assertEqual(r["assignments_evaluated"], 1)
        tg = {t["tube_id"]: t["target_slot"] for t in r["tubes"]}
        self.assertEqual(tg, {1: 0, 2: 6})
        self.assertTrue(all(not t["moved"] for t in r["tubes"]))

    def test_occupant_must_move_two_moves_tie_break(self):
        # 失效槽被非锁定管占用且限值为 0: 必须成对搬到对径空槽;
        # 达到 (2 支移动, |U₁|=0) 的摆放共 10 种, 按管号升序目标槽位序列
        # 裁决为 试管1→槽1, 试管2→槽7
        p = _plan(1, {0: (1, 50.0), 6: (2, 50.0)})
        r = plan_emergency_relocation(p, failed_slot=0,
                                      u1_limit_g_mm=0.0, max_moves=2)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["move_count"], 2)
        self.assertAlmostEqual(r["final_u1"]["u1_mag_g_mm"], 0.0, places=9)
        self.assertEqual(r["optimal_assignment_count"], 10)
        self.assertTrue(r["tie_broken"])
        tg = {t["tube_id"]: t["target_slot"] for t in r["tubes"]}
        self.assertEqual(tg, {1: 1, 2: 7})
        self.assertEqual(r["moved_tube_ids"], [1, 2])
        mv = {(m["tube_id"], m["from_slot"], m["to_slot"]) for m in r["moves"]}
        self.assertEqual(mv, {(1, 0, 1), (2, 6, 7)})
        self.assertTrue(r["failed_slot_occupied"])
        self.assertEqual(r["failed_slot_occupant"], 1)
        # 失效槽位不得占用
        self.assertNotIn(0, tg.values())

    def test_one_move_cap_infeasible_reports_best(self):
        # 同上但移动上限 1: 无可行摆放, 报告上限内可达最小 |U₁|
        p = _plan(1, {0: (1, 50.0), 6: (2, 50.0)})
        r = plan_emergency_relocation(p, failed_slot=0,
                                      u1_limit_g_mm=0.0, max_moves=1)
        self.assertFalse(r["feasible"])
        self.assertFalse(r["within_limit"])
        self.assertIsNone(r["move_count"])
        self.assertIsNone(r["final_u1"])
        self.assertIn("超过限值", r["reason"])
        self.assertIn("无可行摆放", r["reason"])
        # 单支 50 g 搬到离 0° 最近的空槽(±30°): |U₁| = 5000·2·sin15°
        want = 5000.0 * 2.0 * math.sin(math.radians(15.0))
        self.assertIsNotNone(r["best_achievable"])
        self.assertAlmostEqual(r["best_achievable"]["u1_mag_g_mm"], want,
                               places=6)
        self.assertEqual(r["best_achievable"]["move_count"], 1)

    def test_locked_occupant_infeasible(self):
        # 失效槽被锁定试管占用: 锁定不得移动, 任何上限内无可行摆放
        p = _plan(1, {0: (1, 50.0), 6: (2, 50.0)}, locked=(1,))
        r = plan_emergency_relocation(p, failed_slot=0,
                                      u1_limit_g_mm=1e9, max_moves=3)
        self.assertFalse(r["feasible"])
        self.assertIn("锁定试管", r["reason"])
        self.assertIn("无可行摆放", r["reason"])
        self.assertIsNone(r["best_achievable"])
        self.assertEqual(r["assignments_evaluated"], 0)
        self.assertEqual(r["moves"], [])
        self.assertTrue(all(t["target_slot"] is None for t in r["tubes"]))

    def test_zero_move_cap_with_occupied_failed_slot(self):
        # 移动上限 0 但失效槽被占用: 没有任何满足约束的摆放
        p = _plan(1, {0: (1, 50.0), 6: (2, 50.0)})
        r = plan_emergency_relocation(p, failed_slot=0,
                                      u1_limit_g_mm=1e9, max_moves=0)
        self.assertFalse(r["feasible"])
        self.assertIn("腾空", r["reason"])
        self.assertIsNone(r["best_achievable"])
        self.assertEqual(r["assignments_evaluated"], 0)

    def test_zero_move_cap_current_exceeds_limit(self):
        # 失效槽空置、移动上限 0、当前 |U₁| 超限: 无可行摆放,
        # 上限内可达最小 |U₁| 即当前值
        p = _plan(1, {0: (1, 50.0), 1: (2, 40.0)})
        cur = 100.0 * abs(50.0 + 40.0 * complex(
            math.cos(math.radians(30.0)), math.sin(math.radians(30.0))))
        r = plan_emergency_relocation(p, failed_slot=5,
                                      u1_limit_g_mm=1.0, max_moves=0)
        self.assertFalse(r["feasible"])
        self.assertAlmostEqual(r["original_u1"]["u1_mag_g_mm"], cur, places=9)
        self.assertIsNotNone(r["best_achievable"])
        self.assertEqual(r["best_achievable"]["move_count"], 0)
        self.assertAlmostEqual(r["best_achievable"]["u1_mag_g_mm"], cur,
                               places=6)

    def test_locked_tube_stays_and_others_relocate(self):
        # 锁定管不动; 失效槽腾空后由非锁定管补偿
        p = _plan(1, {0: (1, 50.0), 3: (2, 50.0), 6: (3, 40.0)}, locked=(3,))
        r = plan_emergency_relocation(p, failed_slot=3,
                                      u1_limit_g_mm=6000.0, max_moves=3)
        self.assertTrue(r["feasible"])
        tg = {t["tube_id"]: t["target_slot"] for t in r["tubes"]}
        self.assertEqual(tg[3], 6)           # 锁定管保持原槽
        self.assertNotIn(3, tg.values())     # 失效槽腾空
        self.assertLessEqual(r["move_count"], 3)
        self.assertLessEqual(r["final_u1"]["u1_mag_g_mm"], 6000.0 + 1e-6)
        # 与暴力参考一致
        bf, bm, bmag, bseq, bcnt = _brute(p, 3, 6000.0, 3)
        self.assertTrue(bf)
        self.assertEqual(r["move_count"], bm)
        self.assertAlmostEqual(r["final_u1"]["u1_mag_g_mm"], bmag,
                               delta=1e-6 * max(1.0, bmag))
        self.assertEqual(tuple(tg[t] for t in sorted(tg)), bseq)
        self.assertEqual(r["optimal_assignment_count"], bcnt)

    def test_matches_brute_force_random(self):
        rng = random.Random(20260924)
        for _ in range(120):
            n_slots = rng.choice([8, 10, 12])
            enabled = list(range(n_slots))
            disabled = set(rng.sample(enabled, rng.randint(0, 2)))
            free = [s for s in enabled if s not in disabled]
            n_tubes = rng.randint(1, min(6, len(free)))
            occ_slots = rng.sample(free, n_tubes)
            locked = set()
            placements = {}
            for i, s in enumerate(occ_slots):
                tid = i + 1
                if rng.random() < 0.25:
                    locked.add(tid)
                placements[s] = (tid, rng.uniform(5.0, 60.0))
            radius = rng.uniform(50.0, 200.0)
            plan = _plan(1, placements, n_slots=n_slots, disabled=disabled,
                         radius=radius, locked=locked)
            failed = rng.choice(free)
            limit = rng.choice([0.0, rng.uniform(0.0, 8000.0), 1e9])
            mm = rng.randint(0, 3)
            snap = copy.deepcopy(plan)
            r = plan_emergency_relocation(plan, failed, limit, mm)
            self.assertEqual(plan, snap)  # 只读
            bf, bm, bmag, bseq, bcnt = _brute(plan, failed, limit, mm)
            self.assertEqual(r["feasible"], bf)
            if not bf:
                self.assertIsNone(r["final_u1"])
                self.assertFalse(r["within_limit"])
                self.assertTrue(r["reason"])
                continue
            self.assertEqual(r["move_count"], bm)
            self.assertLessEqual(r["move_count"], mm)
            self.assertAlmostEqual(r["final_u1"]["u1_mag_g_mm"], bmag,
                                   delta=1e-6 * max(1.0, bmag))
            tg = {t["tube_id"]: t["target_slot"] for t in r["tubes"]}
            self.assertEqual(tuple(tg[t] for t in sorted(tg)), bseq)
            self.assertEqual(r["optimal_assignment_count"], bcnt)
            self.assertEqual(r["tie_broken"], bcnt > 1)
            # 失效槽腾空; 锁定管不动; 移动数与逐管标记一致
            self.assertNotIn(failed, tg.values())
            for t in r["tubes"]:
                if t["locked"]:
                    self.assertEqual(t["target_slot"], t["original_slot"])
                    self.assertFalse(t["moved"])
                self.assertEqual(t["moved"],
                                 t["target_slot"] != t["original_slot"])
            self.assertEqual(sum(1 for t in r["tubes"] if t["moved"]),
                             r["move_count"])
            self.assertEqual(len(r["moves"]), r["move_count"])
            self.assertLessEqual(r["final_u1"]["u1_mag_g_mm"],
                                 limit + 1e-6 * max(1.0, limit))
            # 目标槽互不相同且都在启用槽内(不含失效槽)
            self.assertEqual(len(set(tg.values())), len(tg))
            self.assertTrue(set(tg.values()) <= set(free) - {failed})
            # 最终 |U₁| 与逐管目标槽独立复算一致
            ux = uy = 0.0
            for t in r["tubes"]:
                a = math.radians(t["target_angle_deg"])
                ux += t["mass_g"] * radius * math.cos(a)
                uy += t["mass_g"] * radius * math.sin(a)
            self.assertAlmostEqual(r["final_u1"]["u1_mag_g_mm"],
                                   math.hypot(ux, uy), places=6)

    def test_geometry_fallback_without_slot_state(self):
        # 结果中缺少 slot_state 时回退到请求中的槽位几何
        p = _plan(1, {0: (1, 50.0), 6: (2, 50.0)}, with_slot_state=False)
        r = plan_emergency_relocation(p, failed_slot=0,
                                      u1_limit_g_mm=0.0, max_moves=2)
        self.assertTrue(r["feasible"])
        tg = {t["tube_id"]: t["target_slot"] for t in r["tubes"]}
        self.assertEqual(tg, {1: 1, 2: 7})

    def test_too_many_tubes_rejected(self):
        placements = {i: (i + 1, 10.0) for i in range(11)}
        p = _plan(1, placements, n_slots=12)
        with self.assertRaises(ValueError) as ctx:
            plan_emergency_relocation(p, 11, 100.0, 1)
        self.assertIn("10", str(ctx.exception))

    def test_too_many_enabled_slots_rejected(self):
        p = _plan(1, {0: (1, 10.0)}, n_slots=17)
        with self.assertRaises(ValueError) as ctx:
            plan_emergency_relocation(p, 1, 100.0, 1)
        self.assertIn("16", str(ctx.exception))

    def test_move_cap_validation(self):
        p = _plan(1, {0: (1, 50.0)})
        for bad in (-1, 4, 2.5, "3", None, True):
            with self.assertRaises(ValueError):
                plan_emergency_relocation(p, 1, 100.0, bad)

    def test_limit_validation(self):
        p = _plan(1, {0: (1, 50.0)})
        for bad in (-0.5, float("inf"), float("nan"), None, "x"):
            with self.assertRaises(ValueError):
                plan_emergency_relocation(p, 1, bad, 1)

    def test_failed_slot_validation(self):
        p = _plan(1, {0: (1, 50.0)}, disabled=(3,))
        with self.assertRaises(ValueError):
            plan_emergency_relocation(p, 99, 100.0, 1)   # 不存在
        with self.assertRaises(ValueError):
            plan_emergency_relocation(p, 3, 100.0, 1)    # 已禁用
        with self.assertRaises(ValueError):
            plan_emergency_relocation(p, 1.5, 100.0, 1)  # 非整数

    def test_empty_plan_rejected(self):
        with self.assertRaises(ValueError):
            plan_emergency_relocation(_plan(1, {}), 0, 100.0, 1)

    def test_invalid_placements_rejected(self):
        # 重复试管编号
        p1 = _plan(1, {0: (1, 50.0), 1: (2, 40.0)})
        p1["result"]["placements"][1]["tube_id"] = 1
        with self.assertRaises(ValueError):
            plan_emergency_relocation(p1, 5, 100.0, 1)
        # 重复槽位
        p2 = _plan(1, {0: (1, 50.0), 1: (2, 40.0)})
        p2["result"]["placements"][1]["slot"] = 0
        with self.assertRaises(ValueError):
            plan_emergency_relocation(p2, 5, 100.0, 1)
        # 质量非正
        p3 = _plan(1, {0: (1, 0.0)})
        with self.assertRaises(ValueError):
            plan_emergency_relocation(p3, 5, 100.0, 1)
        # 占用禁用槽
        p4 = _plan(1, {0: (1, 50.0)}, disabled=(1,))
        p4["result"]["placements"][0]["slot"] = 1
        with self.assertRaises(ValueError):
            plan_emergency_relocation(p4, 5, 100.0, 1)

    def test_read_only(self):
        p = _plan(1, {0: (1, 50.0), 6: (2, 50.0), 3: (3, 30.0)}, locked=(3,))
        snap = copy.deepcopy(p)
        plan_emergency_relocation(p, 0, 100.0, 3)
        plan_emergency_relocation(p, 5, 0.0, 0)
        self.assertEqual(p, snap)

    def test_plan_info_and_note(self):
        p = _plan(9, {0: (1, 50.0)}, name="迁移测试")
        r = plan_emergency_relocation(p, 4, 1e9, 0)
        self.assertEqual(r["plan"]["id"], 9)
        self.assertEqual(r["plan"]["name"], "迁移测试")
        self.assertIn("只读", r["note"])
        self.assertIn("枚举", r["method"])
        self.assertEqual(r["failed_slot"], 4)
        self.assertEqual(r["max_moves"], 0)
        self.assertEqual(r["tube_count"], 1)
        self.assertEqual(r["enabled_slot_count"], 12)
        self.assertEqual(r["radius_mm"], 100.0)
        self.assertEqual(r["u1_limit_g_mm"], 1e9)


if __name__ == "__main__":
    unittest.main()
