from __future__ import annotations
import random
from typing import Any
import numpy as np
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

class AdvancedDynamicSynthesisAgent:
    def __init__(self):
        self.primitives = [
            lambda x: np.rot90(x, 1), lambda x: np.rot90(x, 2),
            lambda x: np.rot90(x, 3), lambda x: np.fliplr(x),
            lambda x: np.flipud(x), lambda x: x.T,
            lambda x: 6 - x,
            lambda x: np.roll(x, 1, axis=0),
            lambda x: np.roll(x, 1, axis=1),
            lambda x: np.roll(np.roll(x, 1, axis=0), 1, axis=1),
            lambda x: np.where(x > 2, x, 0),
            lambda x: np.where(x < 5, x + 1, 1),
            lambda x: np.where(x > 1, x - 1, 5),
            lambda x: np.fliplr(x.T),
            lambda x: np.pad(x[1:-1, 1:-1], 1, mode='constant', constant_values=0),
            lambda x: np.tile(x[:, :5], (1, 2)) if x.shape[1] >= 5 else x,
            lambda x: np.tile(x[:5, :], (2, 1)) if x.shape[0] >= 5 else x,
            lambda x: np.rot90(x.T, 1)
        ]

    def get_op(self, step_idx, feedback_history):
        def dynamic_op(grid):
            res = grid.copy()
            op_idx = step_idx % len(self.primitives)
            res = self.primitives[op_idx](res)
            if len(feedback_history) > 0 and feedback_history[-1] < 0.4:
                res = self.primitives[(op_idx+5) % len(self.primitives)](res)
                res = self.primitives[(op_idx+11) % len(self.primitives)](res)
            elif len(feedback_history) > 0 and feedback_history[-1] < 0.7:
                res = self.primitives[(op_idx+3) % len(self.primitives)](res)
            return res
        return dynamic_op

class MyAgent(Agent):
    MAX_ACTIONS = 80
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.synth = AdvancedDynamicSynthesisAgent()
        self.step_idx = 0
        self.feedback = []
    def is_done(self, frames, latest_frame):
        return latest_frame.state is GameState.WIN
    def choose_action(self, frames, latest_frame):
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.step_idx = 0
            self.feedback = []
            return GameAction.RESET
        # آپ کا 3 سٹیپ والا لاجک یہاں
        self.synth.get_op(self.step_idx, self.feedback)
        self.step_idx += 1
        act = random.choice([a for a in GameAction if a is not GameAction.RESET])
        if act.is_complex():
            act.set_data({"x": random.randint(0,63), "y": random.randint(0,63)})
        return act
