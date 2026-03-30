import os
import pickle
import random
import time
from pathlib import Path

CELEBRATIONS_DIR = Path(__file__).parent / "celebrations"
MAX_SHOW_SECONDS = 7


class ScoreTracker:

    def __init__(self, match_type: str, players: list, show_celebrations: bool = False):
        self.all_players = players[:]
        self.players = players
        self.match = MatchType(match_type)
        self.shot_count = 0
        self.redemption_mode = False
        self.show_celebrations = show_celebrations
        self.game_over = False
        self._pending_winner = None           # set when entering redemption; cleared in _end_redemption
        self._pre_redemption_players = None  # player order saved on redemption entry; restored on tie

    def update_shot(self):
        self.shot_count += 1
        if self.redemption_mode:
            if self.shot_count >= len(self.players):
                self._end_redemption()
        elif self.shot_count >= len(self.players):
            self.shot_count = 0

    def _update_score(self, label):
        self.players[self.shot_count].record_shot(label)

        if self.redemption_mode:
            if label == "miss":
                self.players.pop(self.shot_count)
                self.shot_count -= 1
            else:
                self.players[self.shot_count].update_score()
            return

        if label == "miss":
            return

        self.players[self.shot_count].update_score()
        if self.show_celebrations:
            self._show_gif(self.players[self.shot_count].name)
        self._check_round_limit()

    def _check_round_limit(self):
        if self.players[self.shot_count].round_score < self.match.scores_per_round:
            return

        winner = self.players[self.shot_count]
        players_after = self.players[self.shot_count + 1:]
        eligible = [p for p in players_after if p.round_score == self.match.scores_per_round - 1]

        # No eligible players: end round immediately, then restore via empty redemption
        if not eligible:
            self._end_round(winner)
            if self.game_over:
                return
            self.players = [winner]
            self.shot_count = 0
            self.redemption_mode = True
            return

        # Defer round award until after redemption shots are taken
        self._pending_winner = winner
        self._pre_redemption_players = self.players[:]
        self.players = [winner] + eligible
        self.shot_count = 0
        self.redemption_mode = True

    def _end_round(self, winner):
        winner.rounds += 1
        print(f"\n  {winner.name} wins round {winner.rounds}/{self.match.number_of_rounds}!\n", flush=True)
        if winner.rounds >= self.match.number_of_rounds:
            self.game_over = True
            print(f"\n  {winner.name} wins the game!\n", flush=True)
            return
        for p in self.all_players:
            p.round_score = 0

    def _end_redemption(self):
        self.redemption_mode = False

        if self._pending_winner is None:
            # No-eligible path: _end_round already called, just restore full player list
            lead = self.players[0]
            self.players = [lead] + [p for p in self.all_players if p is not lead]
            self._pre_redemption_players = None
            self.shot_count = 0
            return

        winner = self._pending_winner
        self._pending_winner = None
        survivors = self.players[1:]  # eligible players still in (scored in redemption)

        if survivors:
            # Tied: restore the original rotation — nobody won, continue as before.
            # All players restart at scores_per_round - 1 (one goal away from triggering again).
            survivor_names = " & ".join(p.name for p in survivors)
            for p in self.all_players:
                p.round_score = self.match.scores_per_round - 1
            self.players = self._pre_redemption_players[:]
            print(f"\n  {survivor_names} {'ties' if len(survivors) == 1 else 'tie'} the round! No round win awarded — continuing.\n", flush=True)
        else:
            # No survivors — winner takes the round
            self._end_round(winner)
            if self.game_over:
                self._pre_redemption_players = None
                return
            self.players = [winner] + [p for p in self.all_players if p is not winner]

        self._pre_redemption_players = None
        self.shot_count = 0

    def print_status(self):
        p = self.players[self.shot_count]
        mode = " [REDEMPTION]" if self.redemption_mode else ""
        print(f"\n  Up next: {p.name}{mode}  |  Round score: {p.round_score}/{self.match.scores_per_round}  |  Rounds won: {p.rounds}")
        print(f"\n  Shooting percentages:")
        for player in self.players:
            marker = "*" if player is p else " "
            pct = (player.total_goals / player.total_shots * 100) if player.total_shots > 0 else 0.0
            print(f"    {marker} {player.name:<12} {player.total_goals:>2}/{player.total_shots:<2}  ({pct:.1f}%)")
        print(f"\n  Players remaining: {', '.join(p.name for p in self.players)}\n", flush=True)

    def _show_gif(self, player_name):
        player_dir = CELEBRATIONS_DIR / player_name
        assert player_dir.exists(), f"incorrect celebrations path, current path: {player_dir}"

        pkls = list(player_dir.glob("*.pkl"))
        assert pkls, f"no converted celebrations found for {player_name}, run preconvert_celebrations.py first"

        with open(random.choice(pkls), "rb") as f:
            data = pickle.load(f)

        frames = data["frames"]
        frame_duration = 1 / data["fps"]
        deadline = time.time() + MAX_SHOW_SECONDS

        while time.time() < deadline:
            for frame in frames:
                if time.time() >= deadline:
                    break
                os.system("cls" if os.name == "nt" else "clear")
                print(frame)
                time.sleep(frame_duration)


class MatchType:

    def __init__(self, match_type):
        self.number_of_rounds = 3
        self.scores_per_round = 3
        self._update_match_type(match_type)

    def _update_match_type(self, match_type):
        if match_type == "single_player":
            self.scores_per_round = 1000000


class Player:

    def __init__(self, name):
        self.name = name
        self.round_score = 0
        self.rounds = 0
        self.total_shots = 0
        self.total_goals = 0

    def update_score(self):
        self.round_score += 1

    def record_shot(self, label):
        self.total_shots += 1
        if label == "goal":
            self.total_goals += 1
