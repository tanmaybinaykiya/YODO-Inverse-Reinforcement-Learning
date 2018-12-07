import json
import time
from collections import defaultdict

import numpy as np

from BlockWorld_t import BlockWorld
from constants import Action, COLORS_STR


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
        self.debug = debug

    def get_allowed_actions_from_prev_action(self, prev_actn):
        if not prev_actn or prev_actn[0] == Action.DROP:
            return [(Action.PICK, np.random.randint(0, self.blocks_count))]
        else:
            return [(a, prev_actn[1]) for a in self.non_pick_actions]

    def get_random_action_from_prev_action(self, prev_actn):
        poss_actions = self.get_allowed_actions_from_prev_action(prev_actn)
        return poss_actions[np.random.randint(0, len(poss_actions))]

    @staticmethod
    def get_best_action(q, state):
        return max([(a, q[state][a]) for a in q[state]], key=lambda x: x[1])[0]

    def get_random_action(self, state):
        if state[-2] is None:
            return Action.PICK, np.random.randint(0, self.blocks_count)
        else:
            poss_actions = [(a, state[-2]) for a in self.non_pick_actions]
            return poss_actions[np.random.randint(0, len(poss_actions))]

    def get_next_action(self, state, q, nu):
        if len(q[state]) > 0 and np.random.rand() > nu:
            best_action = RLTrainer.get_best_action(q, state)
            if self.debug: print("Best action:", best_action)
            return best_action
        else:
            next_action = self.get_random_action(state)
            if self.debug: print("Next action:", next_action)
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
            goal = block_world.goal_config
            all_actions_taken.append({"goal": goal, "actions": block_world.run_environment()})
        RLTrainer.serialize_actions(all_actions_taken)

    def q_learning_supervised(self):
        gamma = 0.5
        alpha = 0.5
        q = defaultdict(lambda: defaultdict(lambda: 0))
        demos = RLTrainer.deserialize_actions()

        for demo in demos:
            block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, 1, record=False)
            block_world.create_goal(demo["goal"])
            for action in demo["actions"]:
                curr_state = BlockWorld.convert_state_dict_to_tuple(action["state"])
                action, sel_id = BlockWorld.parse_action(action["action"])
                if action != Action.FINISHED:
                    block_id = sel_id if action == Action.PICK else curr_state[-1]
                    next_state = block_world.get_next_state_based_on_state_tuple(curr_state, (action, block_id))
                    new_reward = block_world.get_reward_for_state(next_state, block_world.goal_config)
                    q_i = q[curr_state][(action, block_id)]
                    if len(q[next_state]) > 0:
                        max_q = max([q[next_state][a_dash] for a_dash in q[next_state]])
                    else:
                        max_q = 0
                    q[curr_state][(action, block_id)] = ((1 - alpha) * q_i) + (alpha * (new_reward + gamma * max_q))
        return q

    def q_learning_fixed_goal(self, starting_nu=1.0):
        alpha = 0.5
        gamma = 0.5


    def q_learning_real(self, starting_nu=1.0):
        alpha = 0.5
        gamma = 0.5
        converged = False
        q_old = defaultdict(lambda: defaultdict(lambda: 0))
        goal_config = [np.random.permutation(self.blocks_count).tolist()]
        nu = starting_nu

        for _ in range(30):
            block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, 1, record=False)
            block_world.create_goal(goal_config)
            if self.debug: print("Goal: ", [COLORS_STR[i] for stack in block_world.goal_config for i in stack])
            iter_v = 0
            ever_seen_goal = False
            while not converged or iter_v < 5000:
                q = q_old.copy()
                block_world.pre_render()

                curr_state = block_world.get_state_as_tuple()
                if self.debug: print("Current State: ", curr_state)
                action, block_id = self.get_next_action(curr_state, q, nu)
                if self.debug: print("Action: ", action, block_id)

                next_state = block_world.get_next_state_based_on_state_tuple(curr_state, (action, block_id))
                new_reward = block_world.get_reward_for_state(next_state)
                if self.debug: print("next_state: ", next_state)
                if self.debug: print("new_reward: ", new_reward)

                ever_seen_goal = ever_seen_goal or new_reward == 1
                q_sa = q[curr_state][(action, block_id)]

                if len(q[next_state]) > 0:
                    max_q_dash_s_dash_a_dash = max([q[next_state][a_dash] for a_dash in q[next_state]])
                else:
                    max_q_dash_s_dash_a_dash = 0
                if self.debug: print("max_q:", max_q_dash_s_dash_a_dash)

                q[curr_state][(action, block_id)] += alpha * (new_reward + gamma * max_q_dash_s_dash_a_dash - q_sa)
                if self.debug: print("q:", q[curr_state][(action, block_id)])

                block_world.update_state_from_tuple(next_state)

                nu *= 0.9995

                block_world.render()
                print("iter:", iter_v)
                converged = ever_seen_goal and q == q_old
                q_old = q
                time.sleep(0.1)
                iter_v += 1
        print(q)

    def q_learning(self, q=None, starting_nu=1.0, decay_nu=True, decay_rate=0.9995):
        gamma = 0.5
        alpha = 0.5
        episode_count = 100
        if not q:
            q = defaultdict(lambda: defaultdict(lambda: 0))
        success_count = 0

        for ep in range(episode_count):
            block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, 1, record=False)
            nu = starting_nu
            print("Goal: ", [COLORS_STR[i] for stack in block_world.goal_config for i in stack])
            for iteration in range(self.iteration_count):
                block_world.pre_render()

                curr_state = block_world.get_state_as_tuple()
                if self.debug: print("Current State: ", curr_state)
                action, block_id = self.get_next_action(curr_state, q, nu)
                if self.debug: print("Action: ", action, block_id)

                next_state = block_world.get_next_state_based_on_state_tuple(curr_state, (action, block_id))
                new_reward = block_world.get_reward_for_state(next_state, block_world.goal_config)
                if self.debug: print("next_state: ", next_state)
                if self.debug: print("new_reward: ", new_reward)

                q_i = q[curr_state][(action, block_id)]

                if len(q[next_state]) > 0:
                    max_q = max([q[next_state][a_dash] for a_dash in q[next_state]])
                else:
                    max_q = 0

                if self.debug: print("max_q:", max_q)

                q[curr_state][(action, block_id)] = ((1 - alpha) * q_i) + (alpha * (new_reward + gamma * max_q))
                if self.debug: print("q:", q[curr_state][(action, block_id)])

                block_world.update_state_from_tuple(next_state)

                block_world.render()
                if new_reward == 1:
                    print("Goal State Reached!!! in %d iterations" % iteration)
                    success_count += 1
                    break

                if decay_nu and iteration > 50:
                    nu = decay_rate * nu

                if iteration % 100 == 1:
                    print("EP[%d]It[%d]: Q[%d], nu:[%f]" % (ep, iteration, len(q), nu))

        print("success_count: ", success_count)

    def q_learning_random(self):
        episode_count = 100
        success_count = 0

        for ep in range(episode_count):
            block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, 1, record=False)
            print("Goal: %s" % [COLORS_STR[i] for stack in block_world.goal_config for i in stack])
            for iteration in range(5000):
                block_world.pre_render()

                curr_state = block_world.get_state_as_tuple()
                action, block_id = self.get_next_action(curr_state, q, nu)

                next_state = block_world.get_next_state_based_on_state_tuple(curr_state, (action, block_id))
                is_goal_next = block_world.get_reward_for_state(next_state, block_world.goal_config) == 0
                if self.debug: print("Current State: ", curr_state, is_goal_next)

                block_world.update_state_from_tuple(next_state)

                block_world.render()
                if is_goal_next:
                    print("Goal State Reached!!! in %d iterations" % iteration)
                    success_count += 1
                    break

                if iteration % 100 == 1:
                    print(ep, iteration, )
                if self.debug: print(iteration)
        print("success_count: ", success_count)

    def random_exploration(self):
        gamma = 0.1
        q = defaultdict(lambda: 0)
        episode_count = 2
        prev_action = None
        for ep in range(episode_count):
            block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, 1, record=True)
            print("Goal: ", [COLORS_STR[i] for stack in block_world.goal_config for i in stack])
            while block_world.get_reward() != 0:
                block_world.pre_render()
                state = block_world.get_state_as_tuple()
                action, block_id = self.get_random_action_from_prev_action(prev_action)
                next_state = block_world.get_next_state_based_on_state_tuple(state, (action, block_id))
                q_val = gamma * max(
                    [q[next_state, b] for b in self.get_allowed_actions_from_prev_action((action, block_id))])
                q[(block_world.get_state_as_tuple(), action)] = block_world.get_reward_for_state(state,
                                                                                                 block_world.goal_config.tolist()) + q_val
                block_world.update_state_from_tuple(next_state)
                prev_action = action, block_id
                block_world.render()

    def random_exploration2(self):
        episode_count = 2
        prev_action = None
        for ep in range(episode_count):
            block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, 1, record=True)
            print("Goal: ", [COLORS_STR[i] for stack in block_world.goal_config for i in stack])
            while block_world.get_reward() != 0:
                block_world.pre_render()
                action, block_id = self.get_random_action_from_prev_action(prev_action)
                print("Action chosen :", action, block_id)
                if action != Action.DROP and action != Action.PICK:
                    block_world.move_block_by_action(action, block_id)

                prev_action = action, block_id
                block_world.render()


def train():
    rl_trainer = RLTrainer(states_x=350, states_y=300, blocks_count=3, iteration_count=5000, debug=False)
    rl_trainer.demo()


def test():
    rl_trainer = RLTrainer(states_x=350, states_y=300, blocks_count=3, iteration_count=5000, debug=False)
    q_supervised = rl_trainer.q_learning_supervised()
    rl_trainer.q_learning(q_supervised, starting_nu=0.5, decay_nu=True, decay_rate=0.9997)


if __name__ == '__main__':
    RLTrainer(states_x=200, states_y=200, blocks_count=2, iteration_count=5000, debug=True).q_learning_real()
