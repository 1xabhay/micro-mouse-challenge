"""Pluggable mouse drivers."""

from .base import Bot
from .registry import available_bots, bot_info, get_bot, register_bot

__all__ = ["Bot", "available_bots", "bot_info", "get_bot", "register_bot"]
