import json
from collections import defaultdict

import numpy as np
import pygame
from BlockWorld_t import BlockWorld
from constants import Action, COLORS_STR
import time
import numpy as np
from Oracle import Oracle
import pickle
from State import State

POSS_ACTIONS=[Action.MOVE_LEFT,Action.MOVE_RIGHT,Action.MOVE_UP,Action.MOVE_DOWN,Action.PICK,Action.DROP]
class RLTrainer:

    def __init__(self, states_x=800, states_y=800, blocks_count=4, stack_count=1, iteration_count=1000, debug=False):
        self.states_x = states_x
        self.order = True
        self.states_y = states_y
        self.blocks_count = blocks_count
        self.stack_count= stack_count
        self.non_pick_actions = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT,Action.DROP]
        self.actions = self.non_pick_actions.copy()
        self.actions.append(Action.PICK)
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        self.idx_to_action = {idx: action for idx, action in enumerate(self.actions)}
        self.iteration_count = iteration_count
        self.non_pick_actions_block_pairs=[]
        self.pick_action_block_pairs=[]
        for i in range(self.blocks_count):
            for j in range(len(self.non_pick_actions)):
                self.non_pick_actions_block_pairs.append((self.non_pick_actions[j],i))
        for i in range(self.blocks_count):
            self.pick_action_block_pairs.append((Action.PICK,i))

        self.debug = debug

    def get_allowed_actions_from_prev_action(self, prev_actn):
        if not prev_actn or prev_actn[0] == Action.DROP:
            return [(Action.PICK, np.random.randint(0, self.blocks_count))]
        else:
            return [(a, prev_actn[1]) for a in self.non_pick_actions]

    def get_random_action_from_prev_action(self, prev_actn):
        poss_actions = self.get_allowed_actions_from_prev_action(prev_actn)
        return poss_actions[np.random.randint(0, len(poss_actions))]


    def get_best_action(self,q, state):
        if state[-2] == None:
            max_q=[(a,q[state].get(a,0)) for a in self.pick_action_block_pairs]
        else:

            max_q=[((a,state[-2]),q[state].get((a,state[-2]),0)) for a in self.non_pick_actions ]
        max_values=[max_q[i][1] for i in range(len(max_q))]
        max_actions=[max_q[i][0] for i in range(len(max_q))]
        if max_values.count(max(max_values))>1:
            max_locs=np.where(np.asarray(max_values)==max(max_values))[0]
            np.random.shuffle(max_locs)
            return max_actions[int(max_locs[0])]
        else:
            return max_actions[np.argmax(max_values)]
        #return max([(a, q[state][a]) for a in q[state]], key=lambda x: x[1])[0]

    def get_random_action(self, state):
        if state[-2] == None:
            return Action.PICK, np.random.randint(0, self.blocks_count)
        else:
            poss_actions = [(a, state[-2]) for a in self.non_pick_actions]
            return poss_actions[np.random.randint(0, len(poss_actions))]

    def get_next_action(self, state, q, nu):
        if state in q and len(q[state]) > 0 and np.random.rand() > nu:
            best_action = self.get_best_action(q, state)

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
    @staticmethod
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

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

    def test_q_learning_real(self,q_old,starting_nu=0.1):
        goal_config = [np.random.permutation(self.blocks_count).tolist()]
        nu = starting_nu
        block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, self.stack_count, record=False)
        ever_seen_goal=False
        cnt=0
        while cnt<self.iteration_count:
            cnt+=1
            #q = q_old.copy()
            block_world.pre_render()

            curr_state = block_world.get_state_as_tuple_pramodith2()
            if self.debug and curr_state in q: print("Current State: %s" +str(curr_state), q[curr_state])
            action, block_id = self.get_next_action(curr_state, q_old, nu)
            if self.debug: print("Action: ", action, block_id)

            next_state = block_world.get_next_state_based_on_state_tuple(curr_state, (action, block_id))
            new_reward = block_world.get_reward_for_state(next_state, curr_state)
            new_reward += block_world.get_reward_for_state_action_pramodith(curr_state, next_state)
            print("Reward")
            print(new_reward)

            if new_reward>=100:
                print("Converged in %d", cnt)
                return cnt
            #if self.debug:

            print("q:", q_old.get(str(curr_state),None))
            block_world.update_state_from_tuple_pramodith(next_state)

            block_world.render()
            #time.sleep(0.1)
        return cnt


    def q_learning_real(self, starting_nu=0.0,use_old=True):
        alpha = 1
        gamma = 0.1
        converged = False
        if use_old:
            q_old=RLTrainer.load_obj("Q\q_oracle")
        else:
            q_old={}
        #q_old = defaultdict(lambda: defaultdict(lambda: 0))
        #goal_config = [np.random.permutation(self.blocks_count).tolist()]
        nu = starting_nu
        block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, self.stack_count, record=False)
        if self.debug: print("Goal: ", [[COLORS_STR[i] for i in stack if i>=0] for stack in block_world.goal_config])

        ever_seen_goal = False
        cnt=0
        q = q_old.copy()
        while cnt<self.iteration_count:
            cnt+=1
        #while not converged:
            block_world.pre_render(True)

            curr_state = block_world.get_state_as_tuple_pramodith2()
            if curr_state not in q:
                q[curr_state]={}
            print("Current State: ", curr_state)
            state_s = State([[block_world.block_dict[i].rect.centerx,block_world.block_dict[i].rect.centery] for i in range(self.blocks_count)],
                          block_world.selected_block_id,block_world.goal_config)
            # if state_s.selected_index == None:
            #     action,block_id=(Action.PICK, np.random.randint(0, state_s.block_count - 1))
            #
            #
            # else:
            #     action, self.order = Oracle.get_oracle_best_action(state_s, state_s.selected_index, self.order)
            #     block_id = state_s.selected_index

            action, block_id = self.get_next_action_supervised_t(state_t=None, state_s=state_s, q=None, nu=0)

            #action, block_id = self.get_next_action(curr_state, q, nu)
            #if action==Action.DROP:
            #    s=0
            if self.debug: print("Action: ", action, block_id)
            next_state = block_world.get_next_state_based_on_state_tuple(curr_state, (action, block_id))
            new_reward = block_world.get_reward_for_state(next_state,curr_state)
            new_reward+= block_world.get_reward_for_state_action_pramodith(curr_state,next_state)
            if new_reward>1 or new_reward<-1:
                if self.debug: print("next_state: ", next_state)
                if self.debug: print("new_reward: ", new_reward)


            ever_seen_goal = ever_seen_goal or new_reward == 1
            if (action,block_id) in q[curr_state]:
                q_sa = q[curr_state][(action, block_id)]
            else:
                q_sa=0
                q[curr_state][(action,block_id)]=0

            if next_state in q and len(q[next_state]) > 0:
                max_q_dash_s_dash_a_dash = max([q[next_state][a_dash] for a_dash in q[next_state]])
            else:
                max_q_dash_s_dash_a_dash = 0
            if self.debug: print("max_q:", max_q_dash_s_dash_a_dash)
            if new_reward > 70:
                q[curr_state][(action, block_id)] = ((1 - alpha) * q_sa) + (alpha * (new_reward))
                break
            else:
                q[curr_state][(action, block_id)] += alpha * (new_reward + gamma * (max_q_dash_s_dash_a_dash) - q_sa)

            if self.debug: print("q:", q[curr_state][(action, block_id)])

            block_world.update_state_from_tuple_pramodith(next_state)

            if cnt>4000 and cnt%250==0 and nu>0.05:
                alpha-=0.1
            # print(cnt)
            #    nu-=0.1
            #nu *= 0.9995

            block_world.render()

            converged = ever_seen_goal and q == q_old
            #q_old = q
            time.sleep(0.1)
        pygame.display.quit()
        #self.test_q_learning_real(q)
        RLTrainer.save_obj(q,"Q\q_oracle")
        #with open ("Q\q_table1.json",'w') as f:
        #    json.dump(q,f,indent=5)

    def q_learning(self, q=None, starting_nu=1.0, decay_nu=True, decay_rate=0.9995):
        gamma = 0.1
        alpha = 1
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

    def get_next_action_supervised_t(self, state_t, state_s, q, nu):
        # rand_val = np.random.rand()
        # if rand_val < 0.8:
        if state_s.selected_index is None:
            return Action.PICK, np.random.randint(0, state_s.block_count - 1)
        else:
            best_action, self.order = Oracle.get_oracle_best_action(state_s, state_s.selected_index, self.order)
        print("Oracle action: %s"% best_action)
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
        # else:
        #     next_action = self.get_random_action_s(state_s)
        #     self.log("Random action:", args=next_action)
        #     return next_action


def train():
    rl_trainer = RLTrainer(states_x=350, states_y=350, blocks_count=3, iteration_count=5000, debug=False)
    rl_trainer.demo()


def test():
    rl_trainer = RLTrainer(states_x=350, states_y=300, blocks_count=3, iteration_count=5000, debug=False)
    q_supervised = rl_trainer.q_learning_supervised()
    rl_trainer.q_learning(q_supervised, starting_nu=0.9, decay_nu=True, decay_rate=0.9997)


if __name__ == '__main__':
    use_old=False
    nu = 0.1

    for i in range(1):
         RLTrainer(states_x=350, states_y=350, blocks_count=3,stack_count=1, iteration_count=5000, debug=True).q_learning_real(use_old=use_old,starting_nu=nu)
         use_old=True


    #q = RLTrainer.load_obj("Q\q_oracle")
    '''
    iterations = []
    for i in range(100):
        iter = RLTrainer(states_x=350, states_y=350, blocks_count=3, stack_count=1, iteration_count=1000,debug=True).test_q_learning_real(q,starting_nu=0.05)
        iterations.append(iter)
    print(iterations)
    print(iterations.count(1000))
    print(sum(iterations) / len(iterations))
    '''