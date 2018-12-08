import json
import argparse
import numpy as np
import pygame
from pygame.locals import *
from BlockWorld_t import BlockWorld
from constants import Action, COLORS_STR,CLICKED_COLOR,COLORS
import time
import numpy as np
import pickle

class Demonstrations:

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
            self.pick_action_block_pairs.append((Action.PICK,i))
        self.debug = debug


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

    def test_q_learning_real(self,q_old,starting_nu=0.1):
        nu = starting_nu
        block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, self.stack_count, record=False)
        cnt=0
        while cnt<self.iteration_count:
            cnt+=1
            block_world.pre_render()
            curr_state = block_world.get_state_as_tuple_pramodith2()
            if self.debug and curr_state in q_old: print("Current State: %s" +str(curr_state), q_old[curr_state])
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

            print("q:", q_old.get(str(curr_state),None))
            block_world.update_state_from_tuple_pramodith(next_state)

            block_world.render()
            #time.sleep(0.1)
        return cnt

    def q_learning_real(self, starting_nu=0.0,use_old=True,record=False,demo_id=1,goal_config=None):
        alpha = 1
        gamma = 0.1
        record_actions={}
        if use_old:
            if demo_id==1:
                q_old=Demonstrations.load_obj("q_table\q_3_blocks_all_goals")
            else:
                q_old=Demonstrations.load_obj("q_table\q_demo_"+str(demo_id-1))
        else:
            q_old={}
        nu = starting_nu
        block_world = BlockWorld(self.states_x, self.states_y, self.blocks_count, self.stack_count, record=False,goal_config=goal_config)
        if record:
            record_actions["starting_state"]=[(block_world.block_dict[i].rect.centerx,block_world.block_dict[i].rect.centery) for i in range(self.blocks_count)]
            record_actions["goal_config"]=[block_world.goal_config]
            record_actions["actions"]=[]
        if self.debug: print("Goal: ", [[COLORS_STR[i] for i in stack if i>=0] for stack in block_world.goal_config])

        cnt=0
        q = q_old.copy()
        while cnt<self.iteration_count:
            cnt+=1
            block_world.pre_render()
            curr_state = block_world.get_state_as_tuple_pramodith2()
            if curr_state not in q:
                q[curr_state]={}
            print("Current State: ", curr_state)

            user_choice=False
            for i in range(10):

                time.sleep(0.1)
                for event in pygame.event.get():
                    if event.type == KEYUP:
                        user_choice=True

                    if event.type == KEYDOWN:

                        if event.key ==K_SPACE:
                            action=Action.DROP
                        elif event.key == K_UP:
                            action = Action.MOVE_UP
                        elif event.key == K_DOWN:
                            action = Action.MOVE_DOWN
                        elif event.key == K_LEFT:
                            action = Action.MOVE_LEFT
                        elif event.key == K_RIGHT:
                            action = Action.MOVE_RIGHT
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        pos = pygame.mouse.get_pos()
                        for block in block_world.block_dict.values():
                            if block.rect.collidepoint(pos):
                                user_choice=True
                                if block_id:
                                    block_world.block_dict[block_id].surf.fill(COLORS[block_id])
                                block_id = block.id
                                action=Action.PICK
                                #block_world.block_dict[block_id].surf.fill(CLICKED_COLOR[block_id])
                                break

            if not user_choice:
                action, block_id = self.get_next_action(curr_state, q, nu)

                if record:
                    record_actions["actions"].append(('algorithm',action,block_id))
            else:
                print('Skipping models choice to listen to the expert')
                if record:
                    record_actions["actions"].append(('user',action,block_id))
            if action==Action.DROP:
                block_world.block_dict[block_id].surf.fill(COLORS[block_id])
            else:
                block_world.block_dict[block_id].surf.fill(CLICKED_COLOR[block_id])


            if self.debug: print("Action: ", action, block_id)
            next_state = block_world.get_next_state_based_on_state_tuple(curr_state, (action, block_id))
            new_reward = block_world.get_reward_for_state(next_state,curr_state)
            new_reward+= block_world.get_reward_for_state_action_pramodith(curr_state,next_state)
            if new_reward>1 or new_reward<-1:
                if self.debug: print("next_state: ", next_state)
                if self.debug: print("new_reward: ", new_reward)


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

            #if cnt>4000 and cnt%250==0 and nu>0.05:
            #    alpha-=0.1

            block_world.render()

            time.sleep(0.1)
        pygame.display.quit()
        Demonstrations.save_obj(q,"q_table\q_demo_"+str(demo_id))
        Demonstrations.save_obj(record_actions,"state_action_recording/demo_"+str(demo_id))

if __name__ == '__main__':
    use_old=True
    nu = 0.1
    demo_id=1
    goal_config=[[[0,1],[1,0]],[[0,1,2],[2,1,0],[0,1,2]]]
    print("Welcome to the blockworld demonstration.")
    print("Controls are: \n\n UP_ARROW:MOVE_UP \n\n DOWN_ARROW: MOVE_DOWN \n\n RIGHT_ARROW: MOVE_RIGHT \n\n LEFT_ARROW: MOVE_LEFT \n\n MOUSE_CLICK: PICKS A BLOCK \n\n SPACE_BAR: DROPS THE SELECTED BLOCK")
    print("\nThe currently active block has a ligher shade of it's original color and on dropping the block, it becomes darker.")
    print("\nThe goal configuration is given in the tiny screen at the top right corner")

    goal_choice=input("Press 1 if you want to demonstrate the same goal, 2 if you want to demonstrate a random goal.")
    while(goal_choice==1 or goal_choice==2):
        print("Wrong choice!")
        goal_choice = input("Press 1 if you want to demonstrate the same goal, 2 if you want to demonstrate a random goal.")
    num_blocks=input("Enter the number of blocks you want in the world options are 2 or 3 \n")
    while (num_blocks == 2 or num_blocks == 3):
        print("Wrong choice!")
        num_blocks = input("Enter the number of blocks you want in the world options are 2 or 3 \n")
    goal_choice=int(goal_choice)
    num_blocks = int(num_blocks)
    if goal_choice==1:
        chosen_goal=np.random.randint(0,num_blocks-1)
        goal_config=goal_config[num_blocks-2][chosen_goal]
    else:
        goal_config=None
    input=input("Press SPACE when you are ready for the task")


    if input==" ":
        for i in range(2):
             Demonstrations(states_x=350, states_y=350, blocks_count=num_blocks,stack_count=1, iteration_count=5000, debug=True)\
                 .q_learning_real(use_old=use_old,starting_nu=nu,demo_id=demo_id,record=True,goal_config=goal_config)
             use_old=True
             demo_id+=1


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