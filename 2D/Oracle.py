import time

import numpy as np

from State import State
from constants import move_action_to_deviation, Action
import random


class Oracle:

    def __init__(self, window_width, window_height, step_size, goal_config, init_state):
        self.window_width = window_width
        self.window_height = window_height
        self.step_size = step_size
        self.goal_config = goal_config
        self.init_state = init_state

    def get_available_actions(self, state, block_idx) -> []:
        position = state.get_position(block_idx)
        return [action for action in move_action_to_deviation if 0 <= position[0] + move_action_to_deviation[action][0] < self.window_width and 0 <= position[1] + move_action_to_deviation[action][1] < self.window_height]

    @staticmethod
    def get_oracle_best_action(state: State, block_idx, order=True):

        if state.goal_reached():
            return Action.DROP, order

        curr_position = state.get_position(block_idx)
        goal_position = state.get_goal_position(block_idx)

        if tuple(curr_position) == tuple(goal_position):
            return Action.DROP, False

        is_goal_blocked = any([tuple(goal_position) == tuple(block_position) for block_position in state.block_positions])
        am_blocking_goal = any([tuple(goal_position) == tuple(curr_position) for goal_position in state.goal_positions]) and not tuple(curr_position) == tuple(state.goal_positions[block_idx])

        is_left_good = state.is_action_good(Action.MOVE_LEFT, block_idx)
        is_right_good = state.is_action_good(Action.MOVE_RIGHT, block_idx)
        is_down_good = state.is_action_good(Action.MOVE_DOWN, block_idx)
        is_up_good = state.is_action_good(Action.MOVE_UP, block_idx)

        is_goal_to_the_left = goal_position[0] < curr_position[0]
        is_goal_to_the_right = goal_position[0] > curr_position[0]
        is_goal_to_the_top = goal_position[1] < curr_position[1]
        is_goal_to_the_bottom = goal_position[1] > curr_position[1]

        if order:
            if is_left_good and is_goal_to_the_left:
                return Action.MOVE_LEFT, order
            elif is_right_good and is_goal_to_the_right:
                return Action.MOVE_RIGHT, order
            elif is_down_good and is_goal_to_the_bottom:
                return Action.MOVE_DOWN, order
            elif is_up_good and is_goal_to_the_top:
                return Action.MOVE_UP, order
        else:
            if is_down_good and is_goal_to_the_bottom:
                return Action.MOVE_DOWN, order
            elif is_up_good and is_goal_to_the_top:
                return Action.MOVE_UP, order
            elif is_left_good and is_goal_to_the_left:
                return Action.MOVE_LEFT, order
            elif is_right_good and is_goal_to_the_right:
                return Action.MOVE_RIGHT, order

        allowed_actions = []

        is_left_allowed = state.is_action_allowed(Action.MOVE_LEFT, block_idx)
        is_right_allowed = state.is_action_allowed(Action.MOVE_RIGHT, block_idx)
        is_down_allowed = state.is_action_allowed(Action.MOVE_DOWN, block_idx)
        is_up_allowed = state.is_action_allowed(Action.MOVE_UP, block_idx)

        if is_left_allowed: allowed_actions.append(Action.MOVE_LEFT)
        if is_right_allowed: allowed_actions.append(Action.MOVE_RIGHT)
        if is_down_allowed: allowed_actions.append(Action.MOVE_DOWN)
        if is_up_allowed: allowed_actions.append(Action.MOVE_UP)

        if am_blocking_goal and allowed_actions:
            return random.choice(allowed_actions), not order

        if is_goal_blocked:
            return Action.DROP, not order

        return Action.DROP, not order

    @staticmethod
    def get_next_state(state, action, block_idx):
        if action == Action.PICK:
            new_state = state.copy()
            new_state.select(block_idx)
        elif action == Action.DROP:
            new_state = state.copy()
            new_state.deselect()
        else:
            new_state = state.get_next_state(block_idx, action)
        return new_state

    def run(self):
        curr_state = self.init_state
        n = curr_state.block_count

        # bring all blocks together
        actions = []
        block_count = curr_state.block_count
        curr_state.set_goal_positions(Oracle.get_goal_position(curr_state, self.goal_config, self.step_size))
        action = None

        block_world = BlockWorld(self.window_width, self.window_height, num_blocks=block_count, num_stacks=1, block_size=self.step_size)
        b_w_g_c = self.goal_config.copy()
        b_w_g_c.tolist().reverse()
        block_world.create_goal([b_w_g_c])

        block_world.pre_render()
        block_world.update_all_block_states(curr_state)
        block_world.render()

        flip_order = True

        while not curr_state.goal_reached():
            actions_taken = []

            if flip_order:
                this_range = range(block_count)
            else:
                this_range = range(block_count - 1, -1, -1)

            for block_idx in this_range:
                if curr_state.get_position(block_idx) != curr_state.goal_positions[block_idx]:
                    actions.append((Action.PICK, block_idx))
                    while action != Action.DROP:
                        time.sleep(0.5)
                        block_world.pre_render()
                        action, flip_order = self.get_best_action(curr_state, block_idx, flip_order)
                        if action:
                            actions_taken.append(action)
                            print(block_idx, curr_state, action)
                            actions.append(action)
                            curr_state = Oracle.get_next_state(curr_state, action, block_idx)
                            block_world.update_all_block_states(curr_state)
                            block_world.render()
                        else:
                            break
                action = None
            if len(actions_taken) == 3 and not curr_state.goal_reached():
                print("STUCK", actions_taken)

                # find all blocks which arent in their goal_pos
                conflicting_blocks = [idx for idx in range(curr_state.block_count) if curr_state.get_position(idx) == curr_state.goal_positions[idx]]

                # break
        time.sleep(2)
        print(actions)

if __name__ == '__main__':
    block_count = 2
    for _ in range(10):
        oracle = Oracle(300, 300, 50, np.random.permutation(block_count), State([(50, 150), (250, 50)], None, None))
        oracle.run()
