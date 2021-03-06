import json
import random
from collections import defaultdict

import numpy as np
import pygame

from BlockWorld import BlockWorld
from constants import Action, COLORS_STR
from utilities import load_obj, save_obj
from State import State
from Oracle import Oracle
# FILE_NAME = "Q/q_table_target_state_no_blank"

POSS_ACTIONS = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN, Action.PICK, Action.DROP]


class RLTrainer:

    def __init__(self, states_x=800, states_y=800, blocks_count=4, iteration_count=1000, debug=False, goal_config=None):
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
        self.goal_config = goal_config

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

    def get_best_action_state_p(self, q, state):
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

    def get_best_action(self, q, state_t, state: State):
        if state.selected_index is None:
            max_q = [(a, q[state_t].get(a, 0)) for a in self.pick_action_block_pairs]
        else:
            max_q = [((a, state.selected_index), q[state_t].get(a, 0)) for a in self.non_pick_actions]
        max_values = [max_q[i][1] for i in range(len(max_q))]
        max_actions = [max_q[i][0] for i in range(len(max_q))]
        if max_values.count(max(max_values)) > 1:
            max_locs = np.where(np.asarray(max_values) == max(max_values))[0]
            np.random.shuffle(max_locs)
            return max_actions[int(max_locs[0])]
        else:
            return max_actions[np.argmax(max_values)]

    def get_random_action_s(self, state: State):
        if state.selected_index is None:
            return Action.PICK, np.random.randint(0, self.blocks_count)
        else:
            poss_actions = [(a, state.selected_index) for a in self.non_pick_actions]
            return poss_actions[np.random.randint(0, len(poss_actions))]

    def get_random_action_p(self, state):
        if state[-2] is None:
            return Action.PICK, np.random.randint(0, self.blocks_count)
        else:
            poss_actions = [(a, state[-2]) for a in self.non_pick_actions]
            return poss_actions[np.random.randint(0, len(poss_actions))]

    def get_next_action(self, state_t, state, q, nu):
        if state_t in q and len(q[state_t]) > 0 and np.random.rand() < nu:
            best_action = self.get_best_action(q, state_t, state)

            self.log("Best action:", args=best_action)
            return best_action
        else:
            next_action = self.get_random_action_s(state)
            self.log("Next action:", args=next_action)
            return next_action

    def get_next_action_supervised_p(self, state_p, state_s, q, nu):
        rand_val = np.random.rand()
        if rand_val < 0.33:
            if state_s.selected_index is None:
                return Action.PICK, random.randint(0, state_s.block_count - 1)
            else:
                best_action, self.order = Oracle.get_oracle_best_action(state_s, state_s.selected_index, self.order)
            self.log("Oracle action: %s", best_action)
            return best_action, state_s.selected_index
        elif rand_val < 0.67:
            if state_p in q and len(q[state_p]) > 0:
                best_action = self.get_best_action(q, state_p)

                if self.debug: print("Best action:", best_action)
                return best_action
            else:
                next_action = self.get_random_action_p(state_p)
                if self.debug: print("Random best action:", next_action)
                return next_action
        else:
            next_action = self.get_random_action_p(state_p)
            self.log("Random action:", args=next_action)
            return next_action

    def get_next_action_supervised_t(self, state_t, state_s, q, nu):
        rand_val = np.random.rand()
        if rand_val < 0.8:
            if state_s.selected_index is None:
                return Action.PICK, random.randint(0, state_s.block_count - 1)
            else:
                best_action, self.order = Oracle.get_oracle_best_action(state_s, state_s.selected_index, self.order)
            self.log("Oracle action: %s", best_action)
            return best_action, state_s.selected_index
        # elif rand_val < 0.67:
        #     if state_t in q and len(q[state_t]) > 0:
        #         best_action = self.get_best_action(q, state_t, state_s)
        #
        #         if self.debug: print("Best action:", best_action)
        #         return best_action
        #     else:
        #         next_action = self.get_random_action_s(state_s)
        #         if self.debug: print("Random best action:", next_action)
        #         return next_action
        else:
            next_action = self.get_random_action_s(state_s)
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
        block_world = BlockWorld(screen_width=self.states_x, screen_height=self.states_y, goal_config=self.goal_config,
                                num_blocks=self.blocks_count, num_stacks=1, record=False)
        cnt = 0
        while cnt < self.iteration_count:
            cnt += 1
            block_world.pre_render(True)

            curr_state_t = block_world.get_state().get_medial_state_repr()
            curr_state_s = block_world.get_state()
            self.log("State:%s, Q[%s]: %s", (curr_state_s, curr_state_t, q_old.get(curr_state_t, "EMPTY")))
            action, block_id = self.get_next_action(curr_state_t, curr_state_s, q_old, nu)
            self.log("Action: ", args=(action, block_id))

            next_state = block_world.get_state().get_next_state((action, block_id), block_world.get_screen_dims())
            if curr_state_s.goal_reached():
                self.log("Converged in %d", cnt)
                return True, cnt
            block_world.update_state(next_state)

            block_world.render()
            # time.sleep(0.1)
        return False, self.iteration_count

    def q_learning_supervised(self, q_old=None, starting_nu=0.5):
        alpha = 0.5
        gamma = 0.5
        if q_old is None:
            q_old = {}
        q = q_old
        nu = starting_nu
        block_world = BlockWorld(screen_width=self.states_x, screen_height=self.states_y, num_blocks=self.blocks_count,
            goal_config=self.goal_config)
        cnt = 0
        while cnt < self.iteration_count:
            block_world.pre_render(True)

            cnt += 1
            curr_state_s = block_world.get_state()
            curr_state_t = curr_state_s.get_medial_state_repr()

            if curr_state_t not in q:
                q[curr_state_t] = {}
            action, block_id = self.get_next_action_supervised_t(curr_state_t, curr_state_s, q, nu)
            next_state_s = block_world.get_state().get_next_state((action, block_id), block_world.get_screen_dims())
            next_state_t = next_state_s.get_medial_state_repr()
            # next_state_p = next_state_s.get_state_as_tuple_pramodith()
            self.log("Current State: %s, \nAction: %s, \nNext_state: %s", (curr_state_t, action, next_state_t))

            new_reward = BlockWorld.get_reward_for_goal(curr_state_s)
            # new_reward = block_world.get_reward_for_state_pramodith(curr_state_s)
            # new_reward += block_world.get_reward_for_state_action_pramodith(curr_state_p, next_state_p)

            if new_reward != 0:
                self.log("Next State: %s, New Reward: %s", (next_state_s, new_reward))

            if action == Action.PICK:
                action_ser = action, block_id
            else:
                action_ser = action

            if action_ser not in q[curr_state_t]:
                q[curr_state_t][action_ser] = 0

            if next_state_t in q and len(q[next_state_t]) > 0:
                max_q_dash_s_dash_a_dash = max([q[next_state_t][a_dash] for a_dash in q[next_state_t]])
            else:
                max_q_dash_s_dash_a_dash = 0
            self.log("max_q: %s", max_q_dash_s_dash_a_dash)

            q[curr_state_t][action ] += alpha * (new_reward + gamma * max_q_dash_s_dash_a_dash - q[curr_state_t][action_ser])
            self.log("q: %s", q[curr_state_t][action_ser])

            if curr_state_s.goal_reached():
                break

            block_world.update_state(next_state_s)
            block_world.render()

        pygame.display.quit()
        return q

    def q_learning_real(self, q_old=None, starting_nu=0.5):
        alpha = 0.5
        gamma = 0.5

        if not q_old:
            q_old = {}

        nu = starting_nu
        block_world = BlockWorld(self.states_x, self.states_y, goal_config=self.goal_config, num_blocks=self.blocks_count,
            num_stacks=1, record=False)
        self.log("Goal: ", [COLORS_STR[i] for stack in block_world.get_state().goal_config for i in stack])

        # ever_seen_goal = False
        cnt = 0
        q = q_old.copy()
        while cnt < self.iteration_count:
            cnt += 1
            # while not converged:
            block_world.pre_render(True)
            curr_state_s = block_world.get_state()
            curr_state_p = curr_state_s.get_state_pramodith_repr()
            curr_state_t = curr_state_s.get_medial_state_repr()

            if curr_state_t not in q:
                q[curr_state_t] = {}

            action, block_id = self.get_next_action(curr_state_t, curr_state_s, q, nu)
            next_state_s = block_world.get_state().get_next_state((action, block_id), block_world.get_screen_dims())
            next_state_p = next_state_s.get_state_pramodith_repr()
            next_state_t = next_state_s.get_medial_state_repr()
            self.log("Current State: %s, \nAction: %s, \nNext_state: %s", (curr_state_t, action, next_state_t))

            # new_reward = block_world.get_reward_for_goal(curr_state_s)
            # # new_reward += block_world.get_manhattan_distance_reward()
            # # new_reward += block_world.penalize_if_not_moved_in_goal_position(curr_state_s, next_state_s)
            # new_reward = block_world.get_reward_for_state_pramodith(curr_state_s)
            # # new_reward += block_world.get_reward_for_drop(action)
            #
            # # new_reward += block_world.get_sparse_reward_for_state_pramodith(curr_state=curr_state_s , next_state=next_state_s, next_state_p=next_state_p, curr_state_p=curr_state_p)
            #
            # # new_reward = block_world.get_reward_for_state(next_state)
            # new_reward += block_world.get_reward_for_state_action_pramodith(curr_state, next_state)

            new_reward = block_world.get_reward_for_state_pramodith(state= curr_state_s, next_state=next_state_s, next_state_p=next_state_p, curr_state_p=curr_state_p)
            new_reward += block_world.get_reward_for_state_action_pramodith(curr_state_p, next_state_p)


            # if new_reward != 0:
            #     self.log("Next State: %s, New Reward: %s", (next_state_s, new_reward))

            if action == Action.PICK:
                action_ser = action, block_id
            else:
                action_ser = action

            if action_ser not in q[curr_state_t]:
                q[curr_state_t][action_ser] = 0

            if next_state_t in q and len(q[next_state_t]) > 0:
                max_q_dash_s_dash_a_dash = max([q[next_state_t][a_dash] for a_dash in q[next_state_t]])
            else:
                max_q_dash_s_dash_a_dash = 0
            self.log("Max Q: %s", max_q_dash_s_dash_a_dash)

            q[curr_state_t][action_ser] += alpha * (new_reward + gamma * max_q_dash_s_dash_a_dash - q[curr_state_t][action_ser])
            self.log("Q[%s][%s]: %s, Action:, ", (curr_state_t, q[curr_state_t][action_ser]), args=action_ser)

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
    for itc in range(1):
        q = RLTrainer(states_x=350, states_y=350, blocks_count=2, iteration_count=10, debug=True, goal_config=[[1, 0, 2]]).q_learning_supervised(q_old=q)
        if itc % 10 == 0:
            print("Iterations:[%d], Q size:[%d]" % (itc, len(q)))
        save_obj(q, filename)


def test_rl(q_file_name):
    print("Testing...")
    q_loaded = load_obj(q_file_name)
    success = []
    failures = 0
    for _ in range(20):
        result, iter_count = RLTrainer(states_x=350, states_y=350, blocks_count=3, iteration_count=1000, debug=False, goal_config=[[1, 0, 2]]).test_q_learning_real(q_loaded)
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
    # q = load_obj(filename)

    for _ in range(10):
        q = RLTrainer(states_x=350, states_y=350, blocks_count=3, iteration_count=5000, debug=False, goal_config=[[1,0,2]]).q_learning_real(q_old=q)
        save_obj(q, filename)


if __name__ == '__main__':
    supervised_filename = "Q/q_table_50X300_supervised"
    unsupervised_filename = "Q/q_table_50X300_unsupervised"

    print("-"*100)
    # train_supervised(supervised_filename)
    # test_rl(supervised_filename)
    print("=" * 100)
    train_unsupervised(unsupervised_filename)
    test_rl(unsupervised_filename)
    print("-" * 100)
