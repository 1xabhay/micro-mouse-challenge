"""The bot plugin contract and registry."""
import pytest

from micromouse.bots import Bot, available_bots, get_bot, register_bot
from micromouse.maze import Direction
from micromouse.sim import MouseSim
from micromouse.maze import Maze


class TestBotContract:
    def test_bot_requires_an_act_implementation(self):
        class Incomplete(Bot):
            name = "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_minimal_bot_only_needs_act(self):
        class Minimal(Bot):
            name = "minimal"

            def act(self, obs):
                return Direction.N

        bot = Minimal()
        obs = MouseSim(Maze(size=4)).reset()
        bot.reset(obs)                      # default reset is a no-op
        assert bot.act(obs) is Direction.N
        bot.on_step(obs, Direction.N, obs)  # default hook is a no-op

    def test_bot_exposes_a_name_and_style(self):
        class Named(Bot):
            name = "named"
            style = "classical"

            def act(self, obs):
                return Direction.N

        assert Named().name == "named"
        assert Named().style == "classical"


class TestRegistry:
    def test_register_and_retrieve(self):
        @register_bot
        class Temp(Bot):
            name = "temp-test-bot"

            def act(self, obs):
                return Direction.N

        assert "temp-test-bot" in available_bots()
        assert isinstance(get_bot("temp-test-bot"), Temp)

    def test_unknown_bot_raises_with_a_helpful_message(self):
        with pytest.raises(KeyError, match="nope"):
            get_bot("nope")

    def test_builtin_bots_are_discoverable(self):
        names = available_bots()
        assert {"floodfill", "dynaq", "ppo"} <= set(names)

    def test_get_bot_passes_through_kwargs(self):
        bot = get_bot("floodfill", name="custom")
        assert bot.name == "custom"
