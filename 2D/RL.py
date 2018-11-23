from collections import defaultdict

import numpy as np

from BlockWorld_t import BlockWorld
from constants import Action, COLORS_STR


class RLTrainer():

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
        if state[-1] is None:
            return Action.PICK, np.random.randint(0, self.blocks_count)
        else:
            poss_actions = [(a, state[-1]) for a in self.non_pick_actions]
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

    def q_learning(self):
        gamma = 0.5
        alpha = 0.5
        episode_count = 100
        q = defaultdict(lambda: defaultdict(lambda: 0))
        success_count = 0

        for ep in range(episode_count):
            block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, 1, record=False)
            nu = 1.0
            print("Goal: ", [COLORS_STR[i] for stack in block_world.goal_config for i in stack])
            for iteration in range(self.iteration_count):
                block_world.prerender()

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
                if new_reward == 0:
                    print("Goal State Reached!!! in %d iterations" % iteration)
                    success_count += 1
                    break

                if iteration > 50:
                    nu = 0.9995 * nu

                if iteration % 100 == 1:
                    print("EP[%d]It[%d]: Q[%d], nu:[%f]" % (ep, iteration, len(q), nu))

        print("success_count: ", success_count)

    def q_learning_random(self):
        episode_count = 100
        q = defaultdict(lambda: defaultdict(lambda: 0))
        success_count = 0

        for ep in range(episode_count):
            block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, 1, record=False)
            nu = 1
            print("Goal: %s" % [COLORS_STR[i] for stack in block_world.goal_config for i in stack])
            for iteration in range(5000):
                block_world.prerender()

                curr_state = block_world.get_state_as_tuple()
                action, block_id = self.get_next_action(curr_state, q, nu)

                next_state = block_world.get_next_state_based_on_state_tuple(curr_state, (action, block_id))
                new_reward = block_world.get_reward_for_state(next_state, block_world.goal_config)
                if self.debug: print("Current State: ", curr_state, new_reward)

                block_world.update_state_from_tuple(next_state)

                block_world.render()
                if new_reward == 0:
                    print("Goal State Reached!!! in %d iterations" % iteration)
                    success_count += 1
                    break

                # if iteration > 50:
                #     nu = 0.9995 * nu

                if iteration % 100 == 1:
                    print(ep, iteration, len(q), nu)
                if self.debug: print(iteration, nu)
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
                block_world.prerender()
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
                # time.sleep(5)

    def random_exploration2(self):
        episode_count = 2
        prev_action = None
        for ep in range(episode_count):
            block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, 1, record=True)
            print("Goal: ", [COLORS_STR[i] for stack in block_world.goal_config for i in stack])
            while block_world.get_reward() != 0:
                block_world.prerender()
                action, block_id = self.get_random_action_from_prev_action(prev_action)
                print("Action chosen :", action, block_id)
                if action != Action.DROP and action != Action.PICK:
                    block_world.move_block_by_action(action, block_id)

                prev_action = action, block_id
                block_world.render()


def main():
    RLTrainer(states_x=350, states_y=300, blocks_count=3, iteration_count=5000, debug=False).q_learning()


if __name__ == '__main__':
    main()
