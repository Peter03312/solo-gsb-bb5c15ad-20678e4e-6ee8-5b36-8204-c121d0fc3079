"""核心业务测试。"""
from app import config
from app.database import connect, init_db

from conftest import add_node, add_segment, seed_initial_revision


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _level(lo, hi, v):
    return {m: v for m in range(lo, hi + 1)}


def _linear_graph(client):
    """A -[edge]-> B，其余参数由测试指定。"""
    add_node(client, "A")
    add_node(client, "B")


# ---------------------------------------------------------------------------
# 1. 最早抵达：必须等待到潮位足够的整分钟
# ---------------------------------------------------------------------------

def test_earliest_arrival_with_waiting(client):
    _linear_graph(client)
    seg = add_segment(client, "A", "B", travel_time=3, depth=4.0)
    # 0..4 潮位 0（吃水不够），5..19 潮位 3（够）
    levels = {**_level(0, 4, 0.0), **_level(5, 19, 3.0)}
    rev = seed_initial_revision(levels)

    r = client.post("/api/trips/plan", json={
        "origin": "A", "destination": "B",
        "start_minute": 0, "window_minutes": 20,
        "draft": 6.0, "height": 1.0, "margin": 0.5,
        "revision": rev,
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["arrival_minute"] == 8          # 第 5 分钟进入，占用 5,6,7
    assert data["total_wait_minutes"] == 5
    assert len(data["itinerary"]) == 1
    leg = data["itinerary"][0]
    assert leg == {
        "seg_id": seg, "origin": "A", "destination": "B",
        "start_minute": 5, "end_minute": 8, "travel_time": 3,
    }


# ---------------------------------------------------------------------------
# 2. 同到达时刻：按航段编号序列字典序择优
# ---------------------------------------------------------------------------

def test_tie_break_single_edge_smaller_seg_id(client):
    _linear_graph(client)
    s1 = add_segment(client, "A", "B", travel_time=3, depth=99.0)
    s2 = add_segment(client, "A", "B", travel_time=3, depth=99.0)
    assert s1 < s2
    seed_initial_revision(_level(0, 19, 3.0))

    r = client.post("/api/trips/plan", json={
        "origin": "A", "destination": "B",
        "start_minute": 0, "window_minutes": 20,
        "draft": 1.0, "height": 0.0, "margin": 0.0,
    })
    assert r.status_code == 200
    assert r.json()["itinerary"][0]["seg_id"] == s1


def test_tie_break_multi_edge_sequence(client):
    # A->X 边1, X->B 边3；A->Y 边4, Y->B 边5；另有 A->X 边2 但字典序更差。
    # 全部航时 1 且全程可行：两条路径均于第 2 分钟到 B，应选 (1,3)。
    for n in ["A", "B", "X", "Y"]:
        add_node(client, n)
    s1 = add_segment(client, "A", "X", 1)
    s2 = add_segment(client, "A", "Y", 1)
    s3 = add_segment(client, "X", "B", 1)
    s4 = add_segment(client, "Y", "B", 1)
    assert (s1, s3) < (s2, s4)
    seed_initial_revision(_level(0, 19, 3.0))

    r = client.post("/api/trips/plan", json={
        "origin": "A", "destination": "B",
        "start_minute": 0, "window_minutes": 20,
        "draft": 1.0, "height": 0.0, "margin": 0.0,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["arrival_minute"] == 2
    assert [l["seg_id"] for l in data["itinerary"]] == [s1, s3]
    assert [(l["start_minute"], l["end_minute"]) for l in data["itinerary"]] == [(0, 1), (1, 2)]


# ---------------------------------------------------------------------------
# 3. 桥底净空约束
# ---------------------------------------------------------------------------

def test_bridge_clearance_constraint(client):
    _linear_graph(client)
    # 净空 5m；船高 3m + 余量 1m => 潮位必须 <= 1m
    seg = add_segment(client, "A", "B", travel_time=1, depth=99.0, clearance=5.0)
    levels = {**_level(0, 9, 2.0), **_level(10, 19, 0.0)}
    seed_initial_revision(levels)

    r = client.post("/api/trips/plan", json={
        "origin": "A", "destination": "B",
        "start_minute": 0, "window_minutes": 20,
        "draft": 1.0, "height": 3.0, "margin": 1.0,
    })
    assert r.status_code == 200
    assert r.json()["itinerary"][0]["start_minute"] == 10


# ---------------------------------------------------------------------------
# 4. 错误：未知节点
# ---------------------------------------------------------------------------

def test_unknown_node(client):
    add_node(client, "A")
    r = client.post("/api/trips/plan", json={
        "origin": "A", "destination": "ZZZ",
        "start_minute": 0, "window_minutes": 20,
        "draft": 1.0, "height": 0.0, "margin": 0.0,
    })
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "NODE_NOT_FOUND"
    assert r.json()["error"]["details"]["node_id"] == "ZZZ"


# ---------------------------------------------------------------------------
# 5. 错误：潮位缺分钟
# ---------------------------------------------------------------------------

def test_missing_tide_data(client):
    _linear_graph(client)
    add_segment(client, "A", "B", travel_time=2, depth=99.0)
    seed_initial_revision(_level(0, 9, 3.0))  # 只有 0..9，窗口要到 19

    r = client.post("/api/trips/plan", json={
        "origin": "A", "destination": "B",
        "start_minute": 0, "window_minutes": 20,
        "draft": 1.0, "height": 0.0, "margin": 0.0,
    })
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "MISSING_TIDE_DATA"
    assert err["details"]["total_missing"] == 10  # 10..19


# ---------------------------------------------------------------------------
# 6. 错误：窗口超过 1440；航时必须正整数
# ---------------------------------------------------------------------------

def test_window_limit_1440(client):
    _linear_graph(client)
    r = client.post("/api/trips/plan", json={
        "origin": "A", "destination": "B",
        "start_minute": 0, "window_minutes": 1441,
        "draft": 1.0, "height": 0.0, "margin": 0.0,
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_segment_travel_time_must_be_positive(client):
    _linear_graph(client)
    for bad in (0, -5):
        r = client.post("/api/segments", json={
            "origin": "A", "destination": "B",
            "travel_time": bad, "depth": 5.0,
        })
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# 7. 错误：窗口内无可行行程
# ---------------------------------------------------------------------------

def test_no_feasible_itinerary(client):
    _linear_graph(client)
    add_segment(client, "A", "B", travel_time=2, depth=4.0)
    seed_initial_revision(_level(0, 19, 0.0))  # 潮位始终为 0

    r = client.post("/api/trips/plan", json={
        "origin": "A", "destination": "B",
        "start_minute": 0, "window_minutes": 20,
        "draft": 6.0, "height": 0.0, "margin": 0.5,
    })
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "NO_FEASIBLE_ITINERARY"


# ---------------------------------------------------------------------------
# 8. 起终点相同：零航段立即抵达
# ---------------------------------------------------------------------------

def test_same_origin_destination(client):
    add_node(client, "A")
    seed_initial_revision(_level(0, 19, 0.0))
    r = client.post("/api/trips/plan", json={
        "origin": "A", "destination": "A",
        "start_minute": 0, "window_minutes": 20,
        "draft": 1.0, "height": 0.0, "margin": 0.0,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["arrival_minute"] == 0
    assert data["itinerary"] == []


# ---------------------------------------------------------------------------
# 9. 潮位更正：新修订继承旧值并覆盖指定分钟
# ---------------------------------------------------------------------------

def test_correction_creates_new_revision(client):
    _linear_graph(client)
    rev1 = seed_initial_revision(_level(0, 19, 3.0))

    r = client.post("/api/tide/corrections", json={
        "base_revision": rev1,
        "corrections": {10: 0.0, 11: 0.0},
        "note": "第 10/11 分钟改为低潮",
    })
    assert r.status_code == 201, r.text
    rev2 = r.json()["revision"]
    assert rev2 == rev1 + 1

    g = client.get(f"/api/tide/revisions/{rev2}", params={"from_minute": 0, "to_minute": 19})
    levels = {int(k): v for k, v in g.json()["levels"].items()}
    assert levels[9] == 3.0       # 继承
    assert levels[10] == 0.0      # 被覆盖
    assert levels[11] == 0.0
    assert levels[12] == 3.0


# ---------------------------------------------------------------------------
# 10. 确认行程：过期修订冲突不保存；复核通过保存快照；事后再更正不影响已确认行程
# ---------------------------------------------------------------------------

def _ab_edge_plan(client, rev):
    return client.post("/api/trips/plan", json={
        "origin": "A", "destination": "B",
        "start_minute": 0, "window_minutes": 20,
        "draft": 1.0, "height": 0.0, "margin": 0.0,
        "revision": rev,
    })


def test_confirm_stale_revision_conflicts_and_not_saved(client):
    _linear_graph(client)
    add_segment(client, "A", "B", travel_time=2, depth=99.0)
    rev1 = seed_initial_revision(_level(0, 19, 3.0))

    planned = _ab_edge_plan(client, rev1).json()

    # 潮位更正 -> 最新变为修订 #2
    r = client.post("/api/tide/corrections", json={"corrections": {5: 2.5}})
    assert r.status_code == 201
    rev2 = r.json()["revision"]

    # 用过期修订 #1 确认 -> 409，不保存
    r = client.post("/api/trips/confirm", json=planned)
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "REVISION_STALE"
    assert r.json()["error"]["details"]["current_revision"] == rev2
    assert client.get("/api/trips").json() == []


def test_confirm_success_snapshot_retained(client):
    _linear_graph(client)
    add_segment(client, "A", "B", travel_time=2, depth=99.0)
    rev1 = seed_initial_revision(_level(0, 19, 3.0))

    planned = _ab_edge_plan(client, rev1).json()
    r = client.post("/api/trips/confirm", json=planned)
    assert r.status_code == 201, r.text
    trip = r.json()
    assert trip["revision"] == rev1
    assert trip["arrival_minute"] == planned["arrival_minute"]

    # 再次更正
    r = client.post("/api/tide/corrections", json={"corrections": {0: 0.0}})
    assert r.status_code == 201

    # 已确认行程保留当时结果，仍可查询，内容不变
    g = client.get(f"/api/trips/{trip['trip_id']}")
    assert g.status_code == 200
    saved = g.json()
    assert saved["revision"] == rev1
    assert saved["itinerary"] == planned["itinerary"]
    assert saved["arrival_minute"] == planned["arrival_minute"]


def test_confirm_plan_mismatch_conflicts(client):
    _linear_graph(client)
    add_segment(client, "A", "B", travel_time=2, depth=99.0)
    rev1 = seed_initial_revision(_level(0, 19, 3.0))

    planned = _ab_edge_plan(client, rev1).json()
    # 篡改行程内容
    planned["itinerary"][0]["start_minute"] = 7
    r = client.post("/api/trips/confirm", json=planned)
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "PLAN_MISMATCH"
    assert client.get("/api/trips").json() == []


# ---------------------------------------------------------------------------
# 11. 演示数据初始化
# ---------------------------------------------------------------------------

def test_demo_seed(tmp_path, monkeypatch):
    db_file = tmp_path / "seed.db"
    monkeypatch.setattr(config, "DB_PATH", str(db_file))
    monkeypatch.setattr(config, "SEED_DEMO", True)

    conn = connect()
    init_db(conn)
    nodes = [r["node_id"] for r in conn.execute("SELECT node_id FROM nodes")]
    assert nodes == ["A", "B", "C", "D"]
    n_seg = conn.execute("SELECT COUNT(*) c FROM segments").fetchone()["c"]
    assert n_seg == 4
    n_tide = conn.execute(
        "SELECT COUNT(*) c FROM tide_values WHERE revision = 1"
    ).fetchone()["c"]
    assert n_tide == 1440
    conn.close()


# ---------------------------------------------------------------------------
# 12. 正航时回路：同一（节点, 分钟）不得剪掉互为前缀的较长序列
# ---------------------------------------------------------------------------

def _cycle_graph(client, levels=None):
    """航段 1:A→B、2:B→C、3:C→B、4:B→D 均 1 分钟；1~3 分钟 0..3 可通行，
    4 仅分钟 3 水深足够（静水深 4.0，需潮位 ≥0.5；1~3 静水深 4.6 全程够）。"""
    for n in ["A", "B", "C", "D"]:
        add_node(client, n)
    s1 = add_segment(client, "A", "B", 1, depth=4.6)
    s2 = add_segment(client, "B", "C", 1, depth=4.6)
    s3 = add_segment(client, "C", "B", 1, depth=4.6)
    s4 = add_segment(client, "B", "D", 1, depth=4.0)
    assert (s1, s2, s3, s4) == (1, 2, 3, 4)
    if levels is None:
        levels = {0: 0.0, 1: 0.0, 2: 0.0, 3: 10.0}
    rev = seed_initial_revision(levels)
    return rev


def _cycle_plan(client, rev):
    return client.post("/api/trips/plan", json={
        "origin": "A", "destination": "D",
        "start_minute": 0, "window_minutes": 4,
        "draft": 4.0, "height": 2.5, "margin": 0.5,
        "revision": rev,
    })


def test_cycle_prefix_sequence_not_pruned(client):
    # [1,4] 与 [1,2,3,4] 均于分钟 4 抵达；字典序 (1,2,3,4) < (1,4)。
    # 旧实现把 (B, 分钟3) 的较长前缀 (1,2,3) 剪掉而误返回 [1,4]。
    rev = _cycle_graph(client)
    r = _cycle_plan(client, rev)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["arrival_minute"] == 4
    assert [l["seg_id"] for l in data["itinerary"]] == [1, 2, 3, 4]
    assert [(l["start_minute"], l["end_minute"]) for l in data["itinerary"]] == [
        (0, 1), (1, 2), (2, 3), (3, 4)]
    assert data["total_wait_minutes"] == 0


def test_cycle_prefix_confirm_consistent(client):
    # 确认复算与预排必须得到同一行程（同事务重算逐段一致）
    rev = _cycle_graph(client)
    planned = _cycle_plan(client, rev).json()
    r = client.post("/api/trips/confirm", json=planned)
    assert r.status_code == 201, r.text
    saved = client.get(f"/api/trips/{r.json()['trip_id']}").json()
    assert [l["seg_id"] for l in saved["itinerary"]] == [1, 2, 3, 4]
    assert saved["arrival_minute"] == 4


def test_cycle_prefix_batch_plan_and_confirm_consistent(client):
    # 批次预排/确认复算与单船一致：回路航路 [1,2,3,4] 不丢失
    rev = _cycle_graph(client)
    ships = [
        {"origin": "A", "destination": "D", "start_minute": 0,
         "window_minutes": 4, "draft": 4.0, "height": 2.5, "margin": 0.5},
        {"origin": "A", "destination": "B", "start_minute": 0,
         "window_minutes": 4, "draft": 4.0, "height": 2.5, "margin": 0.5},
    ]
    r = client.post("/api/trips/batch/plan", json={"ships": ships, "revision": rev})
    assert r.status_code == 200, r.text
    plan = r.json()
    assert [l["seg_id"] for l in plan["ships"][0]["itinerary"]] == [1, 2, 3, 4]
    assert plan["ships"][0]["arrival_minute"] == 4

    body = {
        "ships": [
            {**{k: s[k] for k in ("origin", "destination", "start_minute",
                                  "window_minutes", "draft", "height", "margin")},
             "itinerary": s["itinerary"]}
            for s in plan["ships"]
        ],
        "revision": plan["revision"],
        "idempotency_key": "cycle-batch-1",
    }
    r = client.post("/api/trips/batch/confirm", json=body)
    assert r.status_code == 201, r.text
    assert [l["seg_id"] for l in r.json()["trips"][0]["itinerary"]] == [1, 2, 3, 4]
    # 同键重放仍返回首次结果
    r2 = client.post("/api/trips/batch/confirm", json=body)
    assert r2.status_code == 200
    assert r2.json()["replayed"] is True


def test_cycle_prefix_with_berth(client):
    # 泊位模式同样不得剪掉互为前缀的较长序列
    rev = _cycle_graph(client, {0: 0.0, 1: 0.0, 2: 0.0, 3: 10.0,
                                4: 0.0, 5: 0.0})
    r = client.post("/api/nodes/D/berths", json={"berth_id": "P1"})
    assert r.status_code == 201, r.text
    r = client.put("/api/berths/P1/calendar", json={
        "intervals": [{"start_minute": 0, "end_minute": 10}]})
    assert r.status_code == 200, r.text

    r = client.post("/api/trips/plan", json={
        "origin": "A", "destination": "D",
        "start_minute": 0, "window_minutes": 6,
        "draft": 4.0, "height": 2.5, "margin": 0.5,
        "revision": rev, "berth_id": "P1", "work_minutes": 2,
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["arrival_minute"] == 4
    assert [l["seg_id"] for l in data["itinerary"]] == [1, 2, 3, 4]
    assert data["work_start_minute"] == 4
    assert data["work_end_minute"] == 6


def test_dead_end_edge_not_taken(client):
    # 字典序更小的出航段若通往死路（无法抵达终点），必须跳过：
    # 2:B→C 是死路，3:B→D 仅分钟 2 可通行，应等待后走 [1,3]
    for n in ["A", "B", "C", "D"]:
        add_node(client, n)
    s1 = add_segment(client, "A", "B", 1, depth=4.6)
    s2 = add_segment(client, "B", "C", 1, depth=4.6)
    s3 = add_segment(client, "B", "D", 1, depth=4.0)
    assert (s1, s2, s3) == (1, 2, 3)
    rev = seed_initial_revision({0: 0.0, 1: 0.0, 2: 10.0, 3: 0.0, 4: 0.0})

    r = client.post("/api/trips/plan", json={
        "origin": "A", "destination": "D",
        "start_minute": 0, "window_minutes": 5,
        "draft": 4.0, "height": 2.5, "margin": 0.5,
        "revision": rev,
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["arrival_minute"] == 3
    assert [l["seg_id"] for l in data["itinerary"]] == [1, 3]
    assert [(l["start_minute"], l["end_minute"]) for l in data["itinerary"]] == [
        (0, 1), (2, 3)]
    assert data["total_wait_minutes"] == 1
