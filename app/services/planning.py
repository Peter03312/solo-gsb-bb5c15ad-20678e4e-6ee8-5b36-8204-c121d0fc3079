"""潮汐通航行程规划核心算法。

时间轴为绝对"分钟"，与潮位表 minute 列一致。船舶可在节点的任意整分钟
等待；航段一旦进入即连续占用 travel_time 个分钟（进入分钟 d 占用
d, d+1, ..., d+t-1，于 d+t 抵达下一节点）。

约束（占用的每一分钟均须满足，按修订的预测误差带取保守极值）：
  * 深度：静水深度 + 名义潮位 − 误差 ≥ 吃水 + 安全余量（真值可能偏低）
  * 桥梁：桥底净空 − 名义潮位 − 误差 ≥ 水面以上高度 + 安全余量（真值可能偏高，有桥时）
  * 独占：同一航段同一分钟已被其他**已确认**行程占用时不可再进入（进入
    分钟 d 至抵达前一分钟 d+t-1 独占该航段；节点等待不占用航段）
等待不占用航段，不参与校验。

择优：最早抵达；同一抵达时刻按航段编号序列字典序（小者优先）。
算法：窗口 <= 1440 分钟，分两阶段——
  1) 按时间展开的可达性 DP 求最早抵达分钟（泊位模式：首个泊位作业可行的
     新鲜抵达分钟）。可达性与历史无关，此阶段不含任何序列剪枝；
  2) 在「恰于该分钟抵达」的全部行程中求字典序最小的航段编号序列：先做
     反向可达 DP（终点为吸收态），再正向贪心——当前位置集合恒可压缩为
     「单一节点 + 分钟下界」，每步取可走的最小编号航段、以该航段最早的
     可行进入分钟进入。
注意：不能沿用「每个 (节点, 分钟) 只保留字典序最小序列」的剪枝——正航时
回路下，同一状态的两条序列可能互为前缀（短者字典序更小），但追加同一段
后续航段后大小关系会反转（如 (1,) 与 (1,2,3) 同刻位于 B，追加航段 4 后
(1,4) 反而大于 (1,2,3,4)），剪枝会丢失最终同刻字典序最小的航路。
"""
import json
from dataclasses import dataclass
from typing import Optional

from .. import config
from ..errors import ApiError, no_itinerary, tide_missing, unknown_node
from .berths import (
    berth_free_at,
    load_berth_occupancy,
    resolve_plan_berth,
    work_fits_calendar,
)
from .controls import effective_travel_time, load_controls


@dataclass(frozen=True)
class Segment:
    seg_id: int
    origin: str
    destination: str
    travel_time: int
    depth: float
    clearance: Optional[float]


def _load_segments(conn) -> dict[int, Segment]:
    rows = conn.execute(
        "SELECT seg_id, origin, destination, travel_time, depth, clearance"
        " FROM segments ORDER BY seg_id"
    ).fetchall()
    return {
        r["seg_id"]: Segment(
            r["seg_id"], r["origin"], r["destination"],
            r["travel_time"], r["depth"], r["clearance"],
        )
        for r in rows
    }


def _load_tide(conn, revision: int, lo: int, hi: int) -> dict[int, tuple[float, float]]:
    """返回修订 revision 在 [lo, hi] 闭区间内的 {minute: (level, error)}。"""
    rows = conn.execute(
        "SELECT minute, level, error FROM tide_values"
        " WHERE revision = ? AND minute BETWEEN ? AND ?",
        (revision, lo, hi),
    ).fetchall()
    return {r["minute"]: (r["level"], r["error"]) for r in rows}


def _minute_ok(seg: Segment, tide: float, error: float,
               draft: float, height: float, margin: float) -> bool:
    """航段在某一分钟是否满足深度/净空约束（按误差带取保守极值）。"""
    eps = config.EPS
    if seg.depth + tide - error + eps < draft + margin:
        return False
    if seg.clearance is not None:
        if seg.clearance - tide - error + eps < height + margin:
            return False
    return True


def load_occupancy(conn, lo: int, hi: int, exclude_trip_id: Optional[int] = None
                   ) -> dict[int, set[int]]:
    """读取 [lo, hi] 闭区间内全部已确认行程的航段占用（含狭窄水道组展开）。

    返回 {seg_id: {占用分钟, ...}}。每段从进入分钟 d 起占用
    d, d+1, …, d+travel_time-1（抵达分钟不占用）。既有快照即使彼此重叠也
    原样保留——它们就是后续规划必须避开的占用。
    exclude_trip_id 用于改期复核：建议须避开其他已确认快照，但排除自身。

    狭窄水道组：快照任一段若属于某组，该段原占用（按其固化航时）展开到
    组内全部成员航段的同一批分钟——船进入组内任一航段即独占整组。组建前
    的已确认快照同样按原航段与固化航时计入新组占用（快照不改写，组建前
    已有的重叠也不清理）。
    """
    from .groups import expand_group_occupancy, load_groups

    # 已作废行程的原确认快照仍保留在 confirmed_trips 中作为历史快照可查，
    # 但其按固化航时计算的航段/组占用不再阻挡后续调度，故 LEFT JOIN 排除。
    sql = (
        "SELECT c.itinerary AS itinerary FROM confirmed_trips c"
        " LEFT JOIN voided_trips v ON v.trip_id = c.trip_id"
        " WHERE v.trip_id IS NULL"
    )
    params: list = []
    if exclude_trip_id is not None:
        sql += " AND c.trip_id <> ?"
        params.append(exclude_trip_id)
    occupied: dict[int, set[int]] = {}
    for row in conn.execute(sql, params).fetchall():
        for leg in json.loads(row["itinerary"]):
            d, t = leg["start_minute"], leg["travel_time"]
            end = d + t - 1
            if end < lo or d > hi:
                continue
            occupied.setdefault(leg["seg_id"], set()).update(
                range(max(lo, d), min(hi, end) + 1)
            )
    seg_to_group, group_members = load_groups(conn)
    expand_group_occupancy(occupied, seg_to_group, group_members)
    # 展开可能引入 [lo,hi] 之外航段的键，但其分钟来自区间内原占用，裁剪口径不变
    return occupied


def _build_departure_feasible(
    segs: list[Segment],
    tide: dict[int, float],
    start: int,
    window: int,
    draft: float,
    height: float,
    margin: float,
    blocked: Optional[dict[int, set[int]]] = None,
    controls: Optional[dict[int, list[tuple[int, int, int]]]] = None,
) -> tuple[dict[int, list[bool]], dict[int, list[int]]]:
    """每条航段每个可能进入分钟（窗口内局部下标）是否可行，及其实际航时。

    不可行原因有三：该进入分钟选用的实际航时使航程越出窗口、航段在该航程
    的某一分钟不满足潮位误差带水深/净空，或该航程与其他已确认行程的占用
    （blocked：{seg_id: {绝对分钟}}）重叠。航时在**进入分钟**按分时管制
    （controls：{seg_id: [(起, 止, 通行分钟), ...]}，半开区间）选定，途中
    跨越管制边界也不变；未命中区间沿用航段基础航时。用"不可行分钟"前缀
    和，令一次区间查询为 O(1)。

    返回 (feasible, travel_at)：
      feasible[seg_id][i]：于局部分钟 i（绝对分钟 start+i）进入是否可行；
      travel_at[seg_id][i]：于该分钟进入时的实际通行分钟（含不可行情形）。
    """
    blocked = blocked or {}
    controls = controls or {}
    feasible: dict[int, list[bool]] = {}
    travel_at: dict[int, list[int]] = {}
    horizon = start + window

    for seg in segs:
        seg_blocked = blocked.get(seg.seg_id, ())
        seg_controls = controls.get(seg.seg_id, ())
        # bad[i] 对应绝对分钟 start+i
        bad = [0] * window
        for i in range(window):
            m = start + i
            val = tide.get(m)
            if (val is None
                    or not _minute_ok(seg, val[0], val[1], draft, height, margin)
                    or m in seg_blocked):
                bad[i] = 1
        pref = [0] * (window + 1)
        for i in range(window):
            pref[i + 1] = pref[i] + bad[i]

        ok = [False] * (window + 1)   # 局部下标 d-start，0..window
        tt = [0] * (window + 1)       # 各进入分钟实际选用的通行分钟
        base_t = seg.travel_time
        for i in range(0, window + 1):
            dep = start + i
            t = effective_travel_time(base_t, seg_controls, dep)
            tt[i] = t
            if i + t > window:
                continue
            if pref[i + t] - pref[i] == 0:
                ok[i] = True
        feasible[seg.seg_id] = ok
        travel_at[seg.seg_id] = tt
    return feasible, travel_at


def find_itinerary(
    conn,
    origin: str,
    destination: str,
    start_minute: int,
    window_minutes: int,
    draft: float,
    height: float,
    margin: float,
    revision: Optional[int] = None,
    blocked: Optional[dict[int, set[int]]] = None,
    berth_id: Optional[str] = None,
    work_minutes: Optional[int] = None,
    berth_blocked: Optional[dict[str, list[tuple[int, int]]]] = None,
) -> dict:
    """计算最早抵达行程。

    blocked 给出须避开的已确认航段占用 {seg_id: {绝对分钟}}；缺省为不避占
    （仅供不需要占用语义的内部调用）。

    泊位参数（berth_id 与 work_minutes 成对出现，二选一整体缺省）：船抵达
    终点即开工，作业区间 [arrival, arrival+work) 须完整落在该泊位同一开放
    区间、且不与 berth_blocked（{berth_id: [(起, 止), ...]}，半开占用）
    重叠。未指定泊位时求解与响应完全沿用旧语义。berth_blocked 缺省时由
    调用方语义决定：HTTP 单船/批次路径总是显式传入（含本批前船拟占用）。
    未知节点 -> NODE_NOT_FOUND (404)
    未知泊位 -> BERTH_NOT_FOUND (404)；泊位不属终点 -> BERTH_NOT_AT_DESTINATION (422)
    窗口内缺潮位 -> MISSING_TIDE_DATA (422)
    无可行行程 -> NO_FEASIBLE_ITINERARY (404)
    """
    # 1) 节点校验
    node_rows = conn.execute(
        "SELECT node_id FROM nodes WHERE node_id IN (?, ?)", (origin, destination)
    ).fetchall()
    known = {r["node_id"] for r in node_rows}
    if origin not in known:
        raise unknown_node(origin)
    if destination not in known:
        raise unknown_node(destination)

    # 2) 修订号（显式指定或取最新）
    if revision is None:
        row = conn.execute(
            "SELECT revision FROM tide_revisions ORDER BY revision DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise ApiError(
                404, "TIDE_REVISION_NOT_FOUND",
                "系统中不存在任何潮位修订，无法规划",
            )
        revision = row["revision"]
    else:
        row = conn.execute(
            "SELECT 1 FROM tide_revisions WHERE revision = ?", (revision,)
        ).fetchone()
        if row is None:
            raise ApiError(
                404, "TIDE_REVISION_NOT_FOUND", f"潮位修订不存在: {revision}",
                {"revision": revision},
            )

    horizon = start_minute + window_minutes

    # 2b) 泊位参数解析（未知泊位 / 不属终点 / 非法作业分钟在此拒绝）
    berth = resolve_plan_berth(conn, berth_id, work_minutes, destination)
    berth_mode = berth is not None
    if berth_mode:
        berth_occupied = (berth_blocked or {}).get(berth["berth_id"], [])

    # 3) 潮位覆盖校验：占用分钟最晚到 horizon-1；泊位模式作业区间同样须落在
    #    窗口内（arrival+work <= horizon），故潮位仍按整个窗口逐分钟校验
    tide = _load_tide(conn, revision, start_minute, horizon - 1)
    missing = [m for m in range(start_minute, horizon) if m not in tide]
    if missing:
        raise tide_missing(missing)

    # 4) 起终点相同：零航段，立即抵达（泊位模式下须在起点=终点即时开工）
    if origin == destination:
        if berth_mode and not _berth_window_ok(
            start_minute, berth, berth_occupied, horizon
        ):
            raise no_itinerary(origin, destination)
        result = _result(
            revision, origin, destination, start_minute, window_minutes,
            draft, height, margin, arrival=start_minute, legs=[],
        )
        _attach_berth(result, berth, start_minute)
        return result

    segs = _load_segments(conn)
    seg_list = list(segs.values())
    out_segs: dict[str, list[Segment]] = {}
    for s in seg_list:
        out_segs.setdefault(s.origin, []).append(s)

    # 航段分时航时管制：航时在进入分钟选定（未命中区间用基础航时）
    controls = load_controls(conn, [s.seg_id for s in seg_list])
    feasible, travel_at = _build_departure_feasible(
        seg_list, tide, start_minute, window_minutes, draft, height, margin,
        blocked=blocked, controls=controls,
    )

    # 5) 第一阶段：按分钟展开的可达性 DP，求最早抵达分钟（局部）。
    #    只记录「该分钟能到达哪些节点」，不记录航段序列——同一 (节点, 分钟)
    #    可能同时存在互为前缀的多条序列（正航时回路），任何单序列剪枝都可能
    #    丢失延续后的最优解；序列在第二阶段精确求出。
    #
    #    泊位模式下终点是"吸收态"：船一旦抵达终点即开工，不能在终点等待或再
    #    出发。故落入终点的航段只记入 fresh_dest（新鲜抵达分钟），终点从不
    #    进入 states，也不从终点展开任何航段。
    n = window_minutes
    states: list[set[str]] = [set() for _ in range(n + 1)]
    fresh_dest: set[int] = set()    # 泊位模式：新鲜抵达终点的局部分钟
    states[0].add(origin)

    arrival_local = None
    for i in range(n + 1):
        # 5.1 等待：从 i-1 原样延续到 i（泊位模式终点不在 states，天然不等待）。
        if i > 0:
            states[i] |= states[i - 1]

        # 5.2 非泊位模式：终点首次出现即为最早到达。泊位模式改为扫描全部
        #     新鲜抵达（见 5.4）。
        if not berth_mode and destination in states[i]:
            arrival_local = i
            break

        # 5.3 从本分钟出发的航段（船恰好在节点上即可出发，等待后亦可）。
        #     泊位模式不从终点出发；落入终点的航段写入 fresh_dest。
        for v in states[i]:
            for s in out_segs.get(v, []):
                t = travel_at[s.seg_id][i]
                j = i + t
                if j > n:
                    continue
                if not feasible[s.seg_id][i]:
                    continue
                if berth_mode and s.destination == destination:
                    fresh_dest.add(j)
                else:
                    states[j].add(s.destination)

        # 5.4 泊位模式：按分钟升序，首个泊位作业可行的新鲜抵达分钟即最早抵达。
        if berth_mode and i in fresh_dest:
            arrival_abs = start_minute + i
            if _berth_window_ok(arrival_abs, berth, berth_occupied, horizon):
                arrival_local = i
                break

    if arrival_local is None:
        raise no_itinerary(origin, destination)

    # 6) 第二阶段：在「恰于 arrival_local 抵达终点」的全部可行行程中，取航段
    #    编号序列字典序最小者及其各段进入分钟（详见 _lexicographic_legs）。
    legs = _lexicographic_legs(
        out_segs, feasible, travel_at, origin, destination, arrival_local
    )

    leg_dicts = [
        {
            "seg_id": sid,
            "origin": segs[sid].origin,
            "destination": segs[sid].destination,
            "start_minute": start_minute + pi,
            "end_minute": start_minute + ai,
            # 实际航时以进入分钟当时管制为准（可能与基础航时不同）
            "travel_time": travel_at[sid][pi],
        }
        for sid, pi, ai in legs
    ]
    result = _result(
        revision, origin, destination, start_minute, window_minutes,
        draft, height, margin, arrival=start_minute + arrival_local, legs=leg_dicts,
    )
    # 无论是否指定泊位都附带泊位字段（未指定时为 None，旧响应仅新增 null 字段）
    _attach_berth(result, berth if berth_mode else None,
                  start_minute + arrival_local)
    return result


def _lexicographic_legs(
    out_segs: dict[str, list[Segment]],
    feasible: dict[int, list[bool]],
    travel_at: dict[int, list[int]],
    origin: str,
    destination: str,
    arrival_local: int,
) -> list[tuple[int, int, int]]:
    """恰于局部抵达分钟 arrival_local 到达终点的全部可行行程中，求航段编号
    序列字典序最小者，返回 [(seg_id, 进入局部分钟, 抵达局部分钟), ...]。

    同一 (节点, 分钟) 可能同时存在互为前缀的多条航段序列（正航时回路：短
    序列是长序列的前缀、字典序更小，但两者追加同一段后续后大小关系可能
    反转），因此不能按状态剪枝保留单条序列。正确做法分两步：

    1) 反向可达 DP：can_reach[v][i] 表示「局部分钟 i 位于节点 v（可等待）
       时，能否恰于 arrival_local 抵达终点」。终点视为吸收态——早于
       arrival_local 抵达终点行程即已结束（泊位模式本就不能在终点等待/
       再出发；非泊位模式若更早抵达则与 arrival_local 的最早性矛盾），
       故 can_reach(终点, j) ⟺ j == arrival_local；
    2) 正向贪心：当前位置集合恒可压缩为「单一节点 v + 分钟下界 τ」（在 v
       可于 τ 之后任意整分钟等待）。每一步在所有「τ 之后仍存在可续进入
       分钟」的出航段中取最小编号者，并以该航段**最早**的可续进入分钟
       进入。进入分钟 i 可续当且仅当：航段于 i 进入可行（潮位误差带/
       占用/管制/窗口）、i 本身可续（can_reach[v][i]，即可由当前前缀经
       等待到达）且抵达后仍能恰于 arrival_local 到达终点。
    """
    m = arrival_local

    # 1) 反向可达性（只需有出航段的节点；终点吸收、无出航段的节点恒不可达）
    can_reach: dict[str, list[bool]] = {
        v: [False] * (m + 1) for v in out_segs if v != destination
    }

    def reach(node: str, j: int) -> bool:
        if node == destination:
            return j == m
        cr = can_reach.get(node)
        return cr is not None and cr[j]

    for i in range(m - 1, -1, -1):
        for v, cr in can_reach.items():
            ok = cr[i + 1]  # 在节点等待一分钟
            if not ok:
                for s in out_segs[v]:
                    t = travel_at[s.seg_id][i]
                    if (feasible[s.seg_id][i] and i + t <= m
                            and reach(s.destination, i + t)):
                        ok = True
                        break
            cr[i] = ok

    # 2) 各航段「不小于 i 的最早可续进入分钟」（无则 None），O(1) 查询
    next_good: dict[int, list[Optional[int]]] = {}
    for v, segs_out in out_segs.items():
        if v == destination:
            continue
        cr_v = can_reach[v]
        for s in segs_out:
            tt = travel_at[s.seg_id]
            fs = feasible[s.seg_id]
            nxt: list[Optional[int]] = [None] * (m + 2)  # m+1 为哨兵 None
            for i in range(m, -1, -1):
                t = tt[i]
                if (fs[i] and cr_v[i] and i + t <= m
                        and reach(s.destination, i + t)):
                    nxt[i] = i
                else:
                    nxt[i] = nxt[i + 1]
            next_good[s.seg_id] = nxt

    # 3) 正向贪心：不变式为 can_reach(v, τ) 为真，故每步必有可走航段
    legs: list[tuple[int, int, int]] = []
    v, tau = origin, 0
    while v != destination:
        best_s: Optional[Segment] = None
        best_i = 0
        for s in out_segs.get(v, []):
            nxt = next_good.get(s.seg_id)
            if nxt is None:
                continue
            i = nxt[tau]
            if i is None:
                continue
            if best_s is None or s.seg_id < best_s.seg_id:
                best_s, best_i = s, i
        if best_s is None:  # 不会到达：第一阶段已保证可达，此处仅为防御
            raise no_itinerary(origin, destination)
        t = travel_at[best_s.seg_id][best_i]
        legs.append((best_s.seg_id, best_i, best_i + t))
        v, tau = best_s.destination, best_i + t
    return legs


def _berth_window_ok(arrival: int, berth: dict,
                     occupied: list[tuple[int, int]], horizon: int) -> bool:
    """抵达分钟 arrival 时能否在该泊位完成作业：作业区间不越规划窗口
    （arrival+work <= horizon）、完整落在同一开放区间且不与占用重叠。"""
    if arrival + berth["work_minutes"] > horizon:
        return False
    return (
        work_fits_calendar(arrival, berth["work_minutes"], berth["calendar"])
        and berth_free_at(arrival, berth["work_minutes"], occupied)
    )


def _attach_berth(result: dict, berth: Optional[dict], arrival: int) -> None:
    """把泊位作业结果附加到规划响应：抵达即开工，作业区间固化为
    [arrival, arrival + work_minutes)。未指定泊位时字段为 None。"""
    if berth is None:
        result["berth_id"] = None
        result["work_minutes"] = None
        result["work_start_minute"] = None
        result["work_end_minute"] = None
        return
    result["berth_id"] = berth["berth_id"]
    result["work_minutes"] = berth["work_minutes"]
    result["work_start_minute"] = arrival
    result["work_end_minute"] = arrival + berth["work_minutes"]


def _result(revision, origin, destination, start_minute, window_minutes,
            draft, height, margin, arrival, legs) -> dict:
    sailing = sum(l["travel_time"] for l in legs) if legs else 0
    return {
        "revision": revision,
        "origin": origin,
        "destination": destination,
        "start_minute": start_minute,
        "window_minutes": window_minutes,
        "draft": draft,
        "height": height,
        "margin": margin,
        "arrival_minute": arrival,
        "total_wait_minutes": arrival - start_minute - sailing,
        "itinerary": legs,
    }
