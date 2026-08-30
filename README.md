# 30-day-coding-challenge

A 30 day coding challenge to keep skills sharp and in use while developing elements I am interested in.

# Experiment 1: Micromouse

A 16×16 [Micromouse](https://en.wikipedia.org/wiki/Micromouse) simulator with a
pluggable bot interface, three competing agent styles, and a dashboard to watch
them race.

Inspiration: https://www.youtube.com/watch?v=ZMQbHMgK2rw

```bash
make install     # uv venv + editable install
make test        # 240 tests
make dash        # dashboard on http://localhost:8000
make race MAZES=12
```

---

## 1. The challenge

Micromouse began in the late 1970s: a small self-contained robot mouse must
find the centre of a 16×16 maze it has never seen, then race back through it.
The rules the simulator implements:

| Rule | Implementation |
|---|---|
| 16×16 grid of 180mm cells | `Maze(size=16)` |
| Start in a corner, walled on three sides | `(0,0)`, open north only |
| Goal is the 2×2 block at the centre | `{(7,7),(8,7),(7,8),(8,8)}` |
| Goal has a single gateway | enforced by the generator |
| Mouse is self-contained — no map given | the sim only reveals the current cell's walls |
| 10 minutes in the maze, as many runs as you like | `budget_s=600` |
| A run is start → goal; the return trip doesn't count | detected by the runner |
| Handicapped score = run time + search penalty | `run + maze_time/30` |

Contest mazes are deliberately **not** perfect mazes — they contain loops, so
several routes reach the centre and merely *finding* the goal is not the same as
finding the *fast* way there. The generator carves a spanning tree by randomised
DFS, knocks out extra walls to create loops, stamps on the start/goal fixtures,
then repairs any region the fixtures cut off.

**Time model.** Moving one cell costs 0.12s, each 90° turn 0.18s, hitting a wall
0.50s. Turns being priced means a straight corridor beats a staircase over the
same number of cells, so the speed run uses a turn-aware Dijkstra over
`(cell, heading)` states rather than a plain shortest-path.

Sources: [Micromouse Online rules](https://micromouseonline.com/micromouse-book/rules/) ·
[maze file formats](https://micromouseonline.com/micromouse-book/mazes-and-maze-solving/maze-files/) ·
[IEEE rules](https://www.marshall.edu/cecs/files/MicroMouse_Rules_2023.pdf) ·
[Wikipedia](https://en.wikipedia.org/wiki/Micromouse)

## 2. Plugging a bot in

A bot is any class that can look at an observation and name a direction:

```python
from micromouse.bots import Bot, register_bot
from micromouse.maze import Direction

@register_bot
class MyBot(Bot):
    name = "mybot"
    style = "classical"
    description = "Always drives north."

    def reset(self, obs): ...                  # new maze
    def act(self, obs) -> Direction | None:    # None retires from the attempt
        return Direction.N
    def on_step(self, obs, action, next_obs): ...   # optional learning hook
```

Drop the file in `src/micromouse/bots/` and it is auto-discovered by the CLI,
the dashboard and the tournament. The observation is **partial** by design:

```python
Observation(position, heading, walls, time_s, step, size, goal_cells)
```

`walls` covers the current cell only. There is no `maze` attribute — a test
asserts that, because a bot that could read the maze wouldn't be a micromouse.
Retiring matters: every extra second in the maze is charged back at 1/30 as
search penalty, so a bot with nothing left to learn should stop.

For learning agents there is also a Gymnasium environment (`MicromouseEnv`)
that passes the official `check_env` suite.

## 3. Reinforcement learning: what the literature says

The relevant findings for a maze POMDP with a mouse that senses one cell:

- **Recurrence is the single biggest lever.** Agents with memory consistently
  beat feedforward ones on maze navigation, because identical-looking corridors
  are indistinguishable without it.
  ([POPGym](https://arxiv.org/pdf/2303.01859), [memory-improvable domains](https://arxiv.org/pdf/2508.00046))
- **World-model / model-based methods dominate sample efficiency.** DreamerV3
  and its descendants outperform model-free PPO and TD3 on sensor-denied and
  visual navigation.
  ([DreamerNav](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1655171/full),
  [benchmark](https://arxiv.org/abs/2410.14616))
- **Model-free RL is weak at directed exploration.** Sparse-goal mazes need
  intrinsic motivation, optimism, or planning to avoid random-walk search.

That shaped the pool: one bot that plans over a learned model, one that learns a
reactive recurrent policy, and the classical algorithm as the control.

## 4. The three bots

| Bot | Style | Idea |
|---|---|---|
| `floodfill` | classical | Optimistic map, flood-fill gradient to the goal, then a turn-aware speed run over proven cells. No learning. |
| `dynaq` | tabular RL | Q-learning + **prioritised sweeping** over a learned model. Learns the maze *online, during the attempt*. |
| `ppo` | deep RL | Recurrent PPO (LSTM) with wall masking, pretrained on 64 other mazes. Carries no map. |

**`floodfill`** is the contest standard and the baseline to beat. Three phases —
search, return, speed run — then it retires.

**`dynaq`** starts knowing nothing and learns by trial and error inside the
10-minute budget. Plain Dyna-Q could not solve the maze at all (600s timeouts):
uniform replay propagates news of the goal far too slowly. **Prioritised
sweeping** — replaying the transitions whose value moved most, walking backwards
through predecessors — fixed it outright. A typical attempt:

```
run 1: 45.1s  (found the goal)
run 2: 156.1s (still exploring)
run 3: 18.1s
run 4:  7.3s  ← converged to the optimum
run 5:  7.3s
```

**`ppo`** is the only bot tested on *generalisation*: it never sees the race
mazes during training. Illegal directions are masked, so it cannot crash.

## 5. Results

12 mazes, identical for every bot, 10-minute budget each:

| bot | style | score | solved | wins | best run | search | explored |
|---|---|---|---|---|---|---|---|
| `floodfill` | classical | **6.61** | 12/12 | 12 | 5.62s | 15.6s | 28% |
| `dynaq` | tabular-rl | 18.19 | 12/12 | 0 | 9.38s | 123.7s | 80% |
| `ppo` | deep-rl | ∞ | 9/12 | 0 | 78.0s | 139.8s | 49% |

Lower score is better. Reproduce with `make race MAZES=12`.

**Flood-fill wins every maze, and it isn't close.** The honest read:

1. **Dyna-Q matches flood-fill's *routes* but not its *search*.** Its best runs
   are near-identical, because tabular Q-learning with enough planning sweeps
   converges to the same optimal path. It loses on the search penalty: it
   explores 80% of the maze against flood-fill's 28%. Knowing the shortest path
   and knowing when to *stop looking* are different problems, and only one of
   them is what RL optimises here.
2. **PPO failed to solve 3 of 12 mazes** and scores ∞ as a result. It reached
   only a 34% training success rate on 16×16. This is the expected result, not a
   bug: model-free RL on a sparse-goal POMDP is exactly the regime the
   literature says is hardest, and it is the only bot that has never seen the
   maze it is racing.
3. **The classical algorithm is very strong.** Forty years of micromouse
   refinement produced an algorithm that is optimal-ish, deterministic, and
   nearly free to compute. RL here is a study in what learning costs, not a
   free upgrade.

Where the RL bots would win: a maze that changes between runs, a mouse with
noisy sensors, or a scoring rule that rewards adaptation. Flood-fill's strength
is precisely that it assumes a static, perfectly-sensed maze.

## 6. Dashboard

`make dash` → http://localhost:8000

Pick a bot and a maze seed, watch the replay (scrub, speed control, explored
cells, optimal route overlay), then race the whole pool and read the leaderboard,
per-maze scores and score comparison.

## 7. Layout

```
src/micromouse/
  maze.py         Maze, wall bitmasks (N=1,E=2,S=4,W=8), .maz/text IO
  generator.py    procedural contest mazes
  sim.py          mouse physics, partial observation, timing
  env.py          Gymnasium environment
  mapping.py      the belief map a bot builds for itself
  scoring.py      handicapped time
  runner.py       one contest attempt
  tournament.py   fair head-to-head racing
  cli.py          list / maze / run / race / train / dash
  bots/           the plugin pool
  rl/ppo.py       recurrent PPO
  server/         FastAPI + dashboard
tests/            240 tests, written first
```

## 8. Commands

```bash
make help                          # all targets
make test                          # pytest
make cov                           # coverage
make run BOT=dynaq SEED=7          # one attempt
make race MAZES=20                 # tournament
make train BOT=ppo EPISODES=300000 # retrain the deep-RL policy
make dash PORT=8000                # dashboard
make docker-up                     # dashboard in docker
make docker-test                   # test suite in docker
```

## Tech reference

- gymnasium: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
- maze files: https://github.com/micromouseonline/mazefiles
