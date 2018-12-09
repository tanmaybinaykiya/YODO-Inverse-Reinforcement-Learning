import pickle
import time

import numpy as np
import pygame
from pygame.locals import *

from BlockWorld_t import BlockWorld
from constants import Action, CLICKED_COLOR, COLORS, COLORS_STR


class Demonstrations:

    def __init__(self, states_x=800, states_y=800, blocks_count=4, stack_count=1, iteration_count=1000, debug=False):
        self.states_x = states_x
        self.order = True
        self.states_y = states_y
        self.blocks_count = blocks_count
        self.stack_count = stack_count
        self.non_pick_actions = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.DROP]
        self.actions = self.non_pick_actions.copy()
        self.actions.append(Action.PICK)
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        self.idx_to_action = {idx: action for idx, action in enumerate(self.actions)}
        self.iteration_count = iteration_count
        self.non_pick_actions_block_pairs = []
        self.pick_action_block_pairs = []
        for i in range(self.blocks_count):
            self.pick_action_block_pairs.append((Action.PICK, i))
        self.debug = debug

    def get_best_action(self, q, state):
        if state[-2] == None:
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
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def test_q_learning_real(self, q_old, starting_nu=0.1):
        nu = starting_nu
        block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, self.stack_count, record=False)
        cnt = 0
        while cnt < self.iteration_count:
            cnt += 1
            block_world.pre_render()
            curr_state = block_world.get_state_as_tuple_pramodith2()
            if self.debug and curr_state in q_old: print("Current State: %s" + str(curr_state), q_old[curr_state])
            action, block_id = self.get_next_action(curr_state, q_old, nu)
            if self.debug: print("Action: ", action, block_id)

            next_state = block_world.get_next_state_based_on_state_tuple(curr_state, (action, block_id))
            new_reward = block_world.get_reward_for_state(next_state, curr_state)
            new_reward += block_world.get_reward_for_state_action_pramodith(curr_state, next_state)
            print("Reward")
            print(new_reward)

            if new_reward >= 100:
                print("Converged in %d", cnt)
                return cnt

            print("q:", q_old.get(str(curr_state), None))
            block_world.update_state_from_tuple_pramodith(next_state)

            block_world.render()  # time.sleep(0.1)
        return cnt

    def q_learning_real(self, starting_nu=0.0, use_old=True, record=False, demo_id=1, goal_config=None):
        alpha = 1
        gamma = 0.1
        action = None
        picked = False
        paused = False
        user_choice = True
        user_motion_pick = False
        rendered_pick = True
        record_actions = {}
        if use_old:
            if demo_id == 1:
                q_old = Demonstrations.load_obj("q_table/q_3_blocks_all_goals")
            else:
                q_old = Demonstrations.load_obj("q_table/q_demo_" + str(demo_id - 1))
        else:
            q_old = {}
        nu = starting_nu
        block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, self.stack_count, record=False, goal_config=goal_config)
        if record:
            record_actions["starting_state"] = [(block_world.block_dict[i].rect.centerx, block_world.block_dict[i].rect.centery) for i in range(self.blocks_count)]
            record_actions["goal_config"] = [block_world.goal_config]
            record_actions["actions"] = []
        if self.debug: print("Goal: ", [[COLORS_STR[i] for i in stack if i >= 0] for stack in block_world.goal_config])

        cnt = 0
        q = q_old.copy()
        while cnt < self.iteration_count:
            cnt += 1
            block_world.pre_render()
            curr_state = block_world.get_state_as_tuple_pramodith2()
            if curr_state not in q:
                q[curr_state] = {}
            if self.debug: print("Current State: ", curr_state)
            user_choice = True
            action = None
            user_motion_pick = False
            while user_choice or paused:
                time.sleep(0.5)
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        print("")
                        if event.key == K_SPACE:
                            if paused == False:
                                paused = True
                            else:
                                block_world.block_dict[block_id].surf.fill(COLORS[block_id])
                                user_motion_pick = False
                                paused = False
                                user_choice = False
                        if rendered_pick and paused:
                            print('Waiting for user correction.')
                            print(block_id)
                            if event.key == K_UP:
                                print("d")
                                user_choice = True
                                picked = False
                                user_motion_pick = True
                                action = Action.MOVE_UP
                                user_choice = True
                            elif event.key == K_DOWN:
                                print("d")
                                user_choice = True
                                picked = False
                                user_motion_pick = True
                                action = Action.MOVE_DOWN
                            elif event.key == K_LEFT:
                                print("d")
                                user_choice = True
                                user_motion_pick = True
                                picked = False
                                action = Action.MOVE_LEFT
                            elif event.key == K_RIGHT:
                                print("d")
                                user_choice = True
                                user_motion_pick = True
                                picked = False
                                action = Action.MOVE_RIGHT
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if paused:
                            pos = pygame.mouse.get_pos()
                            for block in block_world.block_dict.values():
                                if block.rect.collidepoint(pos):
                                    if block_id:
                                        block_world.block_dict[block_id].surf.fill(COLORS[block_id])
                                    action = Action.PICK
                                    block_id = block.id
                                    user_choice = True
                                    picked = True
                                    block_world.block_dict[block_id].surf.fill(CLICKED_COLOR[block_id])
                                    rendered_pick = False
                                    break

                if not user_motion_pick and paused == False:
                    user_choice = False
                if paused == False or (not rendered_pick or user_motion_pick):
                    break

            if not user_choice:
                action, block_id = self.get_next_action(curr_state, q, nu)

                if record:
                    record_actions["actions"].append(('algorithm', action, block_id))
            else:
                if action == Action.PICK:
                    rendered_pick = True
                print('Skipping models choice to listen to the expert')
                if record and action:
                    record_actions["actions"].append(('user', action, block_id))

            if self.debug: print("Action: ", action, block_id)
            next_state = block_world.get_next_state_based_on_state_tuple(curr_state, (action, block_id))
            new_reward = block_world.get_reward_for_state(next_state, curr_state)
            new_reward += block_world.get_reward_for_state_action_pramodith(curr_state, next_state)
            if new_reward > 1 or new_reward < -1:
                if self.debug: print("next_state: ", next_state)
                if self.debug: print("new_reward: ", new_reward)

            if (action, block_id) in q[curr_state]:
                q_sa = q[curr_state][(action, block_id)]
            else:
                q_sa = 0
                q[curr_state][(action, block_id)] = 0

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

            block_world.render()

            time.sleep(0.1)
        pygame.display.quit()
        Demonstrations.save_obj(q, "q_table/q_demo_" + str(demo_id))
        Demonstrations.save_obj(record_actions, "state_action_recording/demo_" + str(demo_id))


if __name__ == '__main__':
    use_old = True
    nu = 0.1
    demo_id = 1
    demo_goal_config = [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]
    np.random.shuffle(demo_goal_config)
    print("*"*150)
    print("Welcome to the Block World demonstration.")
    print("*" * 150)
    print("Controls: \n\n"
          "\tUP_ARROW:MOVE UP \n\n"
          "\tDOWN_ARROW: MOVE DOWN \n\n"
          "\tRIGHT_ARROW: MOVE RIGHT \n\n"
          "\tLEFT_ARROW: MOVE LEFT \n\n"
          "\tMOUSE_CLICK: PICKS A BLOCK \n\n"
          "\tSPACE_BAR: PAUSES/UNPAUSES THE GAME")
    print("*" * 150)
    print("The simulation shows a partially working algorithm that tries to stack blocks in the given goal configuration. \n"
          "The agent will attempt to move the blocks. Your task is to correct the agent whenever you think it's making a wrong move.")
    print("Whenever you think that the agent is stuck or performing the wrong action, press the space bar to pause the game\n"
          "You can then use the mouse to click on the block that you would like to move and then use the arrow keys to move the block in the desired direction. \n"
          "When you think you have corrected the agent enough, press the space bar to unpause the game and let the algorithm take over. \n\n"
          "You can Pause/Unpause the game as many times as you like")
    print("The block that you pick will have a ligher shade compared to its original color.\n")
    print("The goal configuration is given in the tiny screen at the top right corner\n")
    print("The first two demonstrations will be mock rounds for you to get used to the environment.")
    print("="*150)
    input = input("Enter yes when you are ready for the task ")
    while input != "Yes" and input != "yes":
        input = input("Invalid Input. Enter yes when you are ready for the task ")

    if input.lower() == 'yes':
        for i in range(10):
            if i < 2:
                chosen_goal = 0
            else:
                chosen_goal = i // 2
            goal_config = demo_goal_config[chosen_goal]
            Demonstrations(states_x=350, states_y=350, blocks_count=3, stack_count=1, iteration_count=5000, debug=False).q_learning_real(use_old=use_old, starting_nu=nu, demo_id=demo_id, record=True, goal_config=goal_config)
            use_old = True
            demo_id += 1
    print("Thank you for your time.\n")