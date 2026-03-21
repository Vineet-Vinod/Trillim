"""Tests for structured chat events."""

import unittest

from trillim.components.llm._events import ChatDoneEvent, ChatFinalTextEvent, ChatTokenEvent, ChatUsage


class ChatEventTests(unittest.TestCase):
    def test_event_types_are_stable(self):
        usage = ChatUsage(1, 2, 3, 0)
        self.assertEqual(ChatTokenEvent("a").type, "token")
        self.assertEqual(ChatFinalTextEvent("b").type, "final_text")
        self.assertEqual(ChatDoneEvent("c", usage).type, "done")
