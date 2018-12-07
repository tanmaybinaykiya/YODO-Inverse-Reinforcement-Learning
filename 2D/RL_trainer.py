import json
import random
from collections import defaultdict

import numpy as np
import pygame

from BlockWorld import BlockWorld
from constants import Action, COLORS_STR
from utilities import load_obj, save_obj

# FILE_NAME = "Q/q_table_target_state_no_blank"

POSS_ACTIONS = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN, Action.PICK, Action.DROP]


class RLTrainer:

    def __init__(self, states_x=800, states_y=800, blocks_count=4, iteration_count=1000, debug=False):
        self.states_x = states_x
        self.states_y = states_y
        self.blocks_count = blocks_count
        self.non_pick_actions = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.DROP]
        self.actions = self.non_pick_actions.copy()
        self.actions.append(Action.PICK)
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        self.idx_to_action = {idx: action for idx, action in enumerate(self.actions)}
        self.iteration_count = iteration_count
        self.non_pick_actions_block_pairs = []
        self.pick_action_block_pairs = []
        for i in range(self.blocks_count):
            for j in range(len(self.non_pick_actions)):
                self.non_pick_actions_block_pairs.append((self.non_pick_actions[j], i))
        for i in range(self.blocks_count):
            self.pick_action_block_pairs.append((Action.PICK, i))

        self.order = True
        self.debug = debug

    def log(self, st, params=None, args=None):
        if self.debug:
            if params is not None:
                print(st % params)
            else:
                print(st, args)

    def get_allowed_actions_from_prev_action(self, prev_actn):
        if not prev_actn or prev_actn[0] == Action.DROP:
            return [(Action.PICK, np.random.randint(0, self.blocks_count))]
        else:
            return [(a, prev_actn[1]) for a in self.non_pick_actions]

    def get_random_action_from_prev_action(self, prev_actn):
        poss_actions = self.get_allowed_actions_from_prev_action(prev_actn)
        return poss_actions[np.random.randint(0, len(poss_actions))]

    def get_best_action(self, q, state):
        if state[-2] is None:
            max_q = [(a, q[state].get(a, 0)) for a in self.pick_action_block_pairs]
        else:
            max_q = [((a, state[-2]), q[state].get((a, state[-2]), 0)) for a in self.non_pick_actions]
        max_values = [max_q[i][1] for i in range(len(max_q))]
        max_actions = [max_q[i][0] for i in range(len(max_q))]
        if max_values.count(max(max_values)) > 1:
            max_locs = np.where(np.asarray(max_values) == max(max_values))[0]
            np.random.shuffle(max_locs)
            return max_actions[int(max_locs[0])]
        else:
            return max_actions[np.argmax(max_values)]

    def get_random_action(self, state):
        if state[-2] is None:
            return Action.PICK, np.random.randint(0, self.blocks_count)
        else:
            poss_actions = [(a, state[-2]) for a in self.non_pick_actions]
            return poss_actions[np.random.randint(0, len(poss_actions))]

    def get_next_action(self, state, q, nu):
        if state in q and len(q[state]) > 0 and np.random.rand() > nu:
            best_action = self.get_best_action(q, state)

            self.log("Best action:", args=best_action)
            return best_action
        else:
            next_action = self.get_random_action(state)
            self.log("Next action:", args=next_action)
            return next_action

    @staticmethod
    def get_oracle_best_action(state, block_idx, order=True):
        curr_position = state.get_position(block_idx)
        goal_position = state.get_goal_position(block_idx)

        if order:
            if goal_position[0] < curr_position[0]:
                if state.is_action_allowed(Action.MOVE_LEFT, block_idx):
                    return Action.MOVE_LEFT, order
                else:
                    if state.is_action_allowed(Action.MOVE_DOWN, block_idx):
                        return Action.MOVE_DOWN, not order
                    elif state.is_action_allowed(Action.MOVE_UP, block_idx):
                        return Action.MOVE_UP, not order

            elif goal_position[0] > curr_position[0]:
                if state.is_action_allowed(Action.MOVE_RIGHT, block_idx):
                    return Action.MOVE_RIGHT, order
                else:
                    if state.is_action_allowed(Action.MOVE_DOWN, block_idx):
                        return Action.MOVE_DOWN, not order
                    elif state.is_action_allowed(Action.MOVE_UP, block_idx):
                        return Action.MOVE_UP, not order
            elif goal_position[1] > curr_position[1]:
                if state.is_action_allowed(Action.MOVE_DOWN, block_idx):
                    return Action.MOVE_DOWN, order
                else:
                    if state.is_action_allowed(Action.MOVE_LEFT, block_idx):
                        return Action.MOVE_LEFT, not order
                    elif state.is_action_allowed(Action.MOVE_RIGHT, block_idx):
                        return Action.MOVE_RIGHT, not order

            elif goal_position[1] < curr_position[1]:
                if state.is_action_allowed(Action.MOVE_UP, block_idx):
                    return Action.MOVE_UP, order
                else:
                    if state.is_action_allowed(Action.MOVE_LEFT, block_idx):
                        return Action.MOVE_LEFT, not order
                    elif state.is_action_allowed(Action.MOVE_RIGHT, block_idx):
                        return Action.MOVE_RIGHT, not order
            else:
                return Action.DROP, order
        else:
            if goal_position[1] > curr_position[1]:
                if state.is_action_allowed(Action.MOVE_DOWN, block_idx):
                    return Action.MOVE_DOWN, order
                else:
                    if state.is_action_allowed(Action.MOVE_LEFT, block_idx):
                        return Action.MOVE_LEFT, not order
                    elif state.is_action_allowed(Action.MOVE_RIGHT, block_idx):
                        return Action.MOVE_RIGHT, not order
            elif goal_position[1] < curr_position[1]:
                if state.is_action_allowed(Action.MOVE_UP, block_idx):
                    return Action.MOVE_UP, order
                else:
                    if state.is_action_allowed(Action.MOVE_LEFT, block_idx):
                        return Action.MOVE_LEFT, not order
                    elif state.is_action_allowed(Action.MOVE_RIGHT, block_idx):
                        return Action.MOVE_RIGHT, not order
            elif goal_position[0] < curr_position[0]:
                if state.is_action_allowed(Action.MOVE_LEFT, block_idx):
                    return Action.MOVE_LEFT, order
                else:
                    if state.is_action_allowed(Action.MOVE_DOWN, block_idx):
                        return Action.MOVE_DOWN, not order
                    elif state.is_action_allowed(Action.MOVE_UP, block_idx):
                        return Action.MOVE_UP, not order

            elif goal_position[0] > curr_position[0]:
                if state.is_action_allowed(Action.MOVE_RIGHT, block_idx):
                    return Action.MOVE_RIGHT, order
                else:
                    if state.is_action_allowed(Action.MOVE_DOWN, block_idx):
                        return Action.MOVE_DOWN, not order
                    elif state.is_action_allowed(Action.MOVE_UP, block_idx):
                        return Action.MOVE_UP, not order

        return Action.DROP, order

    def get_next_action_supervised(self, state, state_s, q, nu):
        rand_val = np.random.rand()
        if rand_val < 0.33:
            if state_s.selected_index is None:
                return Action.PICK, random.randint(0, state_s.block_count - 1)
            else:
                best_action, self.order = RLTrainer.get_oracle_best_action(state_s, state_s.selected_index, self.order)
            self.log("Oracle action: %s", best_action)
            return best_action, state_s.selected_index
        elif rand_val < 0.67:
            if state in q and len(q[state]) > 0:
                best_action = self.get_best_action(q, state)

                if self.debug: print("Best action:", best_action)
                return best_action
            else:
                next_action = self.get_random_action(state)
                if self.debug: print("Random best action:", next_action)
                return next_action
        else:
            next_action = self.get_random_action(state)
            self.log("Random action:", args=next_action)
            return next_action

    @staticmethod
    def serialize_actions(actions_taken_all_demos):
        with open("state_action_map.json", 'w') as f:
            json.dump(actions_taken_all_demos, f, indent=2)

    @staticmethod
    def deserialize_actions(filename="state_action_map.json"):
        with open(filename, 'r') as f:
            actions_taken_all_demos = json.load(f)
        return actions_taken_all_demos

    def demo(self, demo_count=50):
        all_actions_taken = []
        for _ in range(demo_count):
            block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, 1, record=False)
            goal = block_world.get_state().goal_config
            all_actions_taken.append({"goal": goal, "actions": block_world.run_environment()})
        RLTrainer.serialize_actions(all_actions_taken)

    def test_q_learning_real(self, q_old, starting_nu=0.05):
        # goal_config = [np.random.permutation(self.blocks_count).tolist()]
        nu = starting_nu
        block_world = BlockWorld(screen_width=self.states_x, screen_height=self.states_y, goal_config=[[1, 0, 2]],
                                num_blocks=self.blocks_count, num_stacks=1, record=False)
        cnt = 0
        while cnt < self.iteration_count:
            cnt += 1
            block_world.pre_render(True)

            curr_state = block_world.get_state().get_state_as_tuple_pramodith()
            curr_state_s = block_world.get_state()
            self.log("State:%s, Q[%s]: %s", (curr_state_s, curr_state, q_old.get(curr_state, "EMPTY")))
            action, block_id = self.get_next_action(curr_state, q_old, nu)
            self.log("Action: ", args=(action, block_id))

            next_state = block_world.get_state().get_next_state((action, block_id), block_world.get_screen_dims())
            if curr_state_s.goal_reached():
                self.log("Converged in %d", cnt)
                return True, cnt
            block_world.update_state(next_state)

            block_world.render()  # time.sleep(0.1)
        return False, self.iteration_count

    def q_learning_supervised(self, q_old=None, starting_nu=0.5):
        alpha = 0.5
        gamma = 0.5
        if q_old is None:
            q_old = {}
        nu = starting_nu
        block_world = BlockWorld(screen_width=self.states_x, screen_height=self.states_y, goal_config=[[1,0,2]])

        self.log("Goal: %s", [COLORS_STR[i] for stack in block_world.get_state().goal_config for i in stack])

        ever_seen_goal = False
        cnt = 0
        q = q_old.copy()
        while cnt < self.iteration_count:
            block_world.pre_render(True)

            cnt += 1
            curr_state_s = block_world.get_state()
            curr_state_p = curr_state_s.get_state_as_tuple_pramodith()

            if curr_state_p not in q:
                q[curr_state_p] = {}
            action, block_id = self.get_next_action_supervised(curr_state_p, curr_state_s, q, nu)
            next_state_s = block_world.get_state().get_next_state((action, block_id), block_world.get_screen_dims())
            next_state_p = next_state_s.get_state_as_tuple_pramodith()
            self.log("Current State: %s, \nAction: %s, \nNext_state: %s", (curr_state_s, action, next_state_s))

            new_reward = block_world.get_reward_for_goal()
            # new_reward = block_world.get_reward_for_state_pramodith(curr_state_s)
            # new_reward += block_world.get_reward_for_state_action_pramodith(curr_state_p, next_state_p)

            if new_reward != 0:
                self.log("Next State: %s, New Reward: %s", (next_state_s, new_reward))

            ever_seen_goal = ever_seen_goal or new_reward == 1
            if (action, block_id) in q[curr_state_p]:
                q_sa = q[curr_state_p][(action, block_id)]
            else:
                q_sa = 0
                q[curr_state_p][(action, block_id)] = 0

            if next_state_p in q and len(q[next_state_p]) > 0:
                max_q_dash_s_dash_a_dash = max([q[next_state_p][a_dash] for a_dash in q[next_state_p]])
            else:
                max_q_dash_s_dash_a_dash = 0
            self.log("max_q: %s", max_q_dash_s_dash_a_dash)

            q[curr_state_p][(action, block_id)] += alpha * (new_reward + gamma * max_q_dash_s_dash_a_dash - q_sa)
            self.log("q: %s", q[curr_state_p][(action, block_id)])

            block_world.update_state(next_state_s)
            block_world.render()
        pygame.display.quit()
        return q

    def q_learning_real(self, q_old=None, starting_nu=0.9):
        alpha = 0.5
        gamma = 0.5

        if not q_old:
            q_old = {}

        nu = starting_nu
        block_world = BlockWorld(self.states_x, self.states_y, goal_config=[[1,0,2]], num_blocks=self.blocks_count, num_stacks=1, record=False)
        self.log("Goal: ", [COLORS_STR[i] for stack in block_world.get_state().goal_config for i in stack])

        ever_seen_goal = False
        cnt = 0
        q = q_old.copy()
        while cnt < self.iteration_count:
            cnt += 1
            # while not converged:
            block_world.pre_render(True)
            curr_state_s = block_world.get_state()
            curr_state_p = curr_state_s.get_state_as_tuple_pramodith()

            if curr_state_p not in q:
                q[curr_state_p] = {}

            action, block_id = self.get_next_action(curr_state_p, q, nu)
            next_state_s = block_world.get_state().get_next_state((action, block_id), block_world.get_screen_dims())
            next_state_p = next_state_s.get_state_as_tuple_pramodith()
            self.log("Current State: %s, \nAction: %s, \nNext_state: %s", (curr_state_s, action, next_state_s))

            new_reward = block_world.get_reward_for_goal()
            new_reward += block_world.penalize_if_not_moved_in_goal_position(curr_state_s, next_state_s)
            # new_reward = block_world.get_reward_for_state_pramodith(curr_state_s)
            # new_reward += block_world.get_reward_for_drop(action)
            # new_reward += block_world.get_reward_for_state_action_pramodith(curr_state_p, next_state_p)

            # new_reward = block_world.old_get_reward_for_state(next_state)
            # new_reward += block_world.old_get_reward_for_state_action_pramodith(curr_state, next_state)

            # if new_reward != 0:
            #     self.log("Next State: %s, New Reward: %s", (next_state_s, new_reward))

            ever_seen_goal = ever_seen_goal or new_reward == 1
            if (action, block_id) in q[curr_state_p]:
                q_sa = q[curr_state_p][(action, block_id)]
            else:
                q_sa = 0
                q[curr_state_p][(action, block_id)] = 0

            if next_state_p in q and len(q[next_state_p]) > 0:
                max_q_dash_s_dash_a_dash = max([q[next_state_p][a_dash] for a_dash in q[next_state_p]])
            else:
                max_q_dash_s_dash_a_dash = 0
            self.log("Max Q: %s", max_q_dash_s_dash_a_dash)

            q[curr_state_p][(action, block_id)] += alpha * (new_reward + gamma * max_q_dash_s_dash_a_dash - q_sa)
            self.log("Q[%s]: %s" %(curr_state_p, q[curr_state_p][(action, block_id)]))

            block_world.update_state(next_state_s)
            block_world.render()
        pygame.display.quit()
        return q

    def random_exploration(self):
        gamma = 0.1
        q = defaultdict(lambda: 0)
        episode_count = 2
        prev_action = None
        for ep in range(episode_count):
            block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, 1, record=True)
            print("Goal: ", [COLORS_STR[i] for stack in block_world.get_state().goal_config for i in stack])
            while not block_world.get_state().goal_reached():
                block_world.pre_render()
                state = block_world.get_state()
                action, block_id = self.get_random_action_from_prev_action(prev_action)
                next_state = block_world.get_state().get_next_state((action, block_id), block_world.get_screen_dims())
                q_val = gamma * max([q[next_state, b] for b in self.get_allowed_actions_from_prev_action((action, block_id))])
                q[(block_world.get_state().get_state_as_tuple(), action)] = block_world.get_reward_for_state(state) + q_val
                block_world.update_state(next_state)
                prev_action = action, block_id
                block_world.render()

    def random_exploration2(self):
        episode_count = 2
        prev_action = None
        for ep in range(episode_count):
            block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, 1, record=True)
            print("Goal: ", [COLORS_STR[i] for stack in block_world.get_state().goal_config for i in stack])
            while not block_world.get_state().goal_reached():
                block_world.pre_render()
                action, block_id = self.get_random_action_from_prev_action(prev_action)
                print("Action chosen :", action, block_id)
                if action != Action.DROP and action != Action.PICK:
                    next_state = block_world.get_state().get_next_state((action, block_id), block_world.get_screen_dims())
                    block_world.update_state(next_state)

                prev_action = action, block_id
                block_world.render()


def train_supervised(filename):
    print("Training supervised...")
    q = None
    for _ in range(20):
        q = RLTrainer(states_x=350, states_y=350, blocks_count=3, iteration_count=3000, debug=False).q_learning_supervised(q_old = q)
    save_obj(q, filename)


def test_rl(q_file_name):
    print("Testing...")
    q_loaded = load_obj(q_file_name)
    success = []
    failures = 0
    for _ in range(20):
        result, iter_count = RLTrainer(states_x=350, states_y=350, blocks_count=3, iteration_count=1000, debug=False).test_q_learning_real(q_loaded)
        if result:
            print("\tSuccess", iter_count)
            success.append(iter_count)
        else:
            print("\tFailed")
            failures += 1
    print("\tSuccess: ", success, "Failed: ", failures)


def train_unsupervised(filename):
    print("Training Unsupervised...")
    q = None
    for _ in range(1):
        q = RLTrainer(states_x=350, states_y=350, blocks_count=3, iteration_count=1000, debug=False).q_learning_real(q_old=q)
    save_obj(q, filename)


if __name__ == '__main__':
    supervised_filename = "Q/q_table_50X300_supervised"
    unsupervised_filename = "Q/q_table_50X300_unsupervised"

    print("-"*100)
    train_supervised(supervised_filename)
    test_rl(supervised_filename)
    print("=" * 100)
    # train_unsupervised(unsupervised_filename)
    # test_rl(unsupervised_filename)
    # print("-" * 100)
