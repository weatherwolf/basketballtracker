"""
test_competition.py  —  verify ScoreTracker game logic
Run from repo root: python test_competition.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "raspberry"))
from competition import ScoreTracker, Player

W = 72
all_pass = True


def header(title):
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")


def shoot(tracker, label):
    p = tracker.players[tracker.shot_count]
    tag = "[REDEM]" if tracker.redemption_mode else "       "
    result = "GOAL" if label == "goal" else "miss"
    print(f"  {tag}  {p.name} shoots -> {result:<4}", end="   ")
    tracker._update_score(label)
    tracker.update_shot()

    scores  = "  ".join(f"{p.name}:{p.round_score}pts" for p in tracker.all_players)
    wins    = "  ".join(f"{p.name}:{p.rounds}W"        for p in tracker.all_players)
    nxt     = tracker.players[tracker.shot_count].name if not tracker.game_over else "--"
    redem   = " REDEM" if tracker.redemption_mode else ""
    print(f"[{scores}]  [{wins}]  next={nxt}{redem}")


def run_test(title, names, sequence, assertions):
    global all_pass
    header(title)
    players = [Player(n) for n in names]
    t = ScoreTracker("multiplayer", players)
    pm = {p.name: p for p in players}

    for label in sequence:
        shoot(t, label)
        if t.game_over:
            break

    print()
    ok = True
    for desc, check in assertions:
        passed = check(t, pm)
        mark = "PASS" if passed else "FAIL"
        print(f"    {mark}  {desc}")
        if not passed:
            ok = False
            all_pass = False
    return ok


# ---------------------------------------------------------------------------

run_test(
    "1. Clean round win — B has only 1 goal when A hits 3 (no redemption)",
    ["A", "B"],
    [
        "goal", "goal",   # A:1, B:1
        "goal", "miss",   # A:2, B:1
        "goal",           # A:3 — B not eligible (needs 2)
    ],
    [
        ("A wins round 1",         lambda t, p: p["A"].rounds == 1),
        ("B gets 0 rounds",        lambda t, p: p["B"].rounds == 0),
        ("round scores reset to 0",lambda t, p: p["A"].round_score == 0 and p["B"].round_score == 0),
        ("A leads next round",     lambda t, p: t.players[0].name == "A"),
    ]
)

run_test(
    "2. Redemption — B misses -> A wins round",
    ["A", "B"],
    [
        "goal", "goal",   # A:1, B:1
        "goal", "goal",   # A:2, B:2
        "goal",           # A:3 -> redemption (B eligible: 2 goals)
        "miss",           # B misses -> A wins
    ],
    [
        ("A wins round 1",             lambda t, p: p["A"].rounds == 1),
        ("B gets 0 rounds",            lambda t, p: p["B"].rounds == 0),
        ("round scores reset to 0",    lambda t, p: p["A"].round_score == 0 and p["B"].round_score == 0),
        ("A leads next round",         lambda t, p: t.players[0].name == "A"),
    ]
)

run_test(
    "3. Redemption — B scores -> tied, round NOT awarded to A",
    ["A", "B"],
    [
        "goal", "goal",   # A:1, B:1
        "goal", "goal",   # A:2, B:2
        "goal",           # A:3 -> redemption
        "goal",           # B scores -> tied
    ],
    [
        ("A gets 0 rounds (not awarded prematurely)", lambda t, p: p["A"].rounds == 0),
        ("B gets 0 rounds",                           lambda t, p: p["B"].rounds == 0),
        ("all scores set to scores_per_round-1 (2)",  lambda t, p: p["A"].round_score == 2 and p["B"].round_score == 2),
        ("A leads (rotation unchanged)",               lambda t, p: t.players[0].name == "A"),
        ("redemption mode cleared",                   lambda t, p: not t.redemption_mode),
    ]
)

run_test(
    "4. After tie (both at 2): rotation unchanged, A shoots again -> redemption -> B misses -> A wins",
    ["A", "B"],
    [
        "goal", "goal",   # A:1, B:1
        "goal", "goal",   # A:2, B:2
        "goal",           # A:3 -> redemption
        "goal",           # B ties -> both at 2, rotation restored: A leads
        "goal",           # A:3 -> redemption again (B eligible at 2)
        "miss",           # B misses -> A wins round
    ],
    [
        ("A wins round 1",          lambda t, p: p["A"].rounds == 1),
        ("B gets 0 rounds",         lambda t, p: p["B"].rounds == 0),
        ("round scores reset to 0", lambda t, p: p["A"].round_score == 0 and p["B"].round_score == 0),
        ("A leads next round",      lambda t, p: t.players[0].name == "A"),
    ]
)

run_test(
    "4b. Two ties in a row: rotation unchanged both times, A leads throughout",
    ["A", "B"],
    [
        "goal", "goal",   # A:1, B:1
        "goal", "goal",   # A:2, B:2
        "goal",           # A:3 -> redemption
        "goal",           # B ties -> both at 2, A leads (rotation unchanged)
        "goal",           # A:3 -> redemption again
        "goal",           # B ties again -> both at 2, A leads again
    ],
    [
        ("no round wins",              lambda t, p: p["A"].rounds == 0 and p["B"].rounds == 0),
        ("both at scores_per_round-1", lambda t, p: p["A"].round_score == 2 and p["B"].round_score == 2),
        ("A still leads (rotation never changed)", lambda t, p: t.players[0].name == "A"),
    ]
)

run_test(
    "5. B wins a round — A already shot this cycle, not eligible for redemption",
    ["A", "B"],
    [
        "goal", "goal",   # A:1, B:1
        "goal", "goal",   # A:2, B:2
        "miss",           # A misses: stays at 2
        "goal",           # B:3 -> B wins; A shot before B so no redemption
    ],
    [
        ("B wins round 1",  lambda t, p: p["B"].rounds == 1),
        ("A gets 0 rounds", lambda t, p: p["A"].rounds == 0),
        ("B leads next round", lambda t, p: t.players[0].name == "B"),
    ]
)

run_test(
    "6. Full game — A wins 3 rounds, B always misses redemption (including game-winning round)",
    ["A", "B"],
    # All 3 rounds: A gets 3, B gets 2, B misses redemption
    [
        "goal","goal","goal","goal","goal","miss",   # round 1
        "goal","goal","goal","goal","goal","miss",   # round 2
        "goal","goal","goal","goal","goal","miss",   # round 3: B still gets redemption
    ],
    [
        ("game is over",    lambda t, p: t.game_over),
        ("A wins 3 rounds", lambda t, p: p["A"].rounds == 3),
        ("B wins 0 rounds", lambda t, p: p["B"].rounds == 0),
    ]
)

run_test(
    "6b. Game-winning shot — B ties, rotation unchanged, A misses next, B wins round",
    ["A", "B"],
    [
        "goal","goal","goal","goal","goal","miss",   # round 1: A wins
        "goal","goal","goal","goal","goal","miss",   # round 2: A wins
        "goal","goal","goal","goal",                 # round 3: A:2, B:2
        "goal",                                      # A:3 (game-winning attempt) -> redemption
        "goal",                                      # B scores -> tied! both at 2, A leads (rotation unchanged)
        "miss",                                      # A misses -> A still at 2
        "goal",                                      # B:3 -> B wins round (A already shot, no redemption)
    ],
    [
        ("game is NOT over after B ties game-winner", lambda t, p: not t.game_over),
        ("A still has 2 rounds",                      lambda t, p: p["A"].rounds == 2),
        ("B wins round 3",                            lambda t, p: p["B"].rounds == 1),
    ]
)

run_test(
    "7. 3 players — only C is eligible; B (0 goals) does not get redemption",
    ["A", "B", "C"],
    [
        "goal", "miss", "goal",   # A:1, B:0, C:1
        "goal", "miss", "goal",   # A:2, B:0, C:2
        "goal",                   # A:3 -> only C eligible (needs 2; B has 0)
        "miss",                   # C misses -> A wins
    ],
    [
        ("A wins round 1",          lambda t, p: p["A"].rounds == 1),
        ("B gets 0 rounds",         lambda t, p: p["B"].rounds == 0),
        ("C gets 0 rounds",         lambda t, p: p["C"].rounds == 0),
        ("A leads next round",      lambda t, p: t.players[0].name == "A"),
    ]
)

run_test(
    "8. 3 players — C scores redemption, B missed earlier; C ties A",
    ["A", "B", "C"],
    [
        "goal", "miss", "goal",   # A:1, B:0, C:1
        "goal", "miss", "goal",   # A:2, B:0, C:2
        "goal",                   # A:3 -> only C eligible
        "goal",                   # C scores -> tied
    ],
    [
        ("no round win for A",   lambda t, p: p["A"].rounds == 0),
        ("no round win for C",   lambda t, p: p["C"].rounds == 0),
        ("all scores set to scores_per_round-1 (2)", lambda t, p: all(p[n].round_score == 2 for n in ["A","B","C"])),
        ("A leads (rotation unchanged)", lambda t, p: t.players[0].name == "A"),
    ]
)

# ---------------------------------------------------------------------------

print(f"\n{'=' * W}")
print(f"  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
print(f"{'=' * W}\n")
