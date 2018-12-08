import os
import random

import numpy as np
import pygame
from pygame.locals import *

from constants import *

import pygame

from constants import COLORS


class Block(pygame.sprite.Sprite):

    def __init__(self, color_index, grid_center, block_size=50):
        super(Block, self).__init__()
        self.block_size = block_size
        self.surf = pygame.Surface((block_size, block_size))
        self.surf.fill(COLORS[color_index])
        self.id = color_index
        self.rect = self.surf.get_rect(center=grid_center)

class BlockWorld:

    def __init__(self, screen_width, screen_height, num_blocks=3, num_stacks=1, block_size=50, record=False):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.goal_screen_dim = (self.screen_width // 5, self.screen_width // 5)
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.target_blocks={}

        self.blocks, self.block_dict = self.create_blocks()

        self.record = record
        if self.record:
            self.demo_id = len(list(os.walk(FRAME_LOCATION)))
            self.demo_dir = os.path.join(FRAME_LOCATION, str(self.demo_id))
            os.mkdir(self.demo_dir)

        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.master_surface = pygame.Surface(self.screen.get_size())
        self.master_surface.fill((0, 0, 0))

        self.goal_surface = pygame.Surface(self.goal_screen_dim)
        self.goal_surface.fill(GRAY)
        self.goal_config = self.create_goal()
        pygame.init()

        self.selected_block_id = None
        self.actions_taken = []

    @staticmethod
    def distance(pts1, pts2):
        return (pts1[0] - pts2[0]) ** 2 + (pts1[1] - pts2[1]) ** 2

    def create_goal(self, goal_config=None):
        if goal_config is None:
            goal_config = (-np.ones((self.num_stacks, self.num_blocks), dtype=np.int8)).tolist()

        # choosing the order for blocks to be placed in the goal screen.
        block_order = [i for i in range(self.num_blocks)]

        seed = np.random.randint(0, self.num_stacks)
        #block_order=[0,1]
        random.shuffle(block_order)
        last_used_block = 0
        blocks_per_stack = self.num_blocks // self.num_stacks
        block_size = self.goal_screen_dim[0] // 10
        if self.num_stacks > 1:
            left_padding = (self.goal_screen_dim[0] - block_size + (3 * (self.num_stacks - 1) * block_size // 2)) // 2
        else:
            left_padding = (self.goal_screen_dim[0] - block_size) // 2

        bottom = self.goal_screen_dim[1] - 2 * block_size
        for stack_num in range(self.num_stacks):
            self.target_blocks[stack_num]={}
            for i in range(blocks_per_stack):
                pygame.draw.rect(self.goal_surface, COLORS[block_order[last_used_block]],
                                 (stack_num * (block_size + 5) + left_padding,  bottom-block_size * i, block_size,
                                  block_size))
                goal_config[stack_num][i] = block_order[last_used_block]
                if last_used_block>0 and i>0:
                    self.target_blocks[stack_num][block_order[last_used_block-1]]=block_order[last_used_block]
                last_used_block += 1

        if self.num_blocks % 2 == 1 and last_used_block != self.num_blocks:
            while last_used_block != self.num_blocks:
                pygame.draw.rect(self.goal_surface, COLORS[block_order[last_used_block]],
                                 seed * (block_size+5) + 40, bottom-block_size * blocks_per_stack)
                goal_config[seed][np.where(goal_config[seed] == -1)[0][0]] = block_order[last_used_block]
                self.target_blocks[seed][block_order[last_used_block-1]]=block_order[last_used_block]
                last_used_block += 1
                blocks_per_stack += 1
        return goal_config

    def euclidean_dist(self,point1,point2):
        return np.square(point1.rect.centerx-point2.rect.midbottom[0])+np.square(point1.rect.centery-point2.rect.midbottom[1])

    def get_reward_for_state_action_pramodith(self, state,next_state):
        if state[-2]==next_state[-2]:
            curr_dist=state[0]
            next_dist=next_state[0]
            if next_dist<curr_dist:
                return 1
            elif next_dist==curr_dist:
                return 0
            else:
                return -5
        return 0

    def get_reward_for_state(self, block_states,curr_state):
        reward=0
        goal_reward=self.get_sparse_reward_for_state_pramodith(block_states,curr_state)
        if goal_reward!=0:
            return goal_reward

        else:
            for block_id in range(len(self.block_dict)):
                for block_id2 in range(len(self.block_dict)):
                    if block_id!=block_id2:
                        if self.euclidean_dist(self.block_dict[block_id],self.block_dict[block_id2])<55:
                            reward+=5
            return reward
        '''
        if block_states[0][0]<30 and block_states[0][1]<30:
            return 1
        else:
            return 0
        '''


    def get_sparse_reward_for_state_pramodith(self, block_states,prev_state):
        sel_block=block_states[-2]
        score = 0
        num_stacks_aligned=1
        flag2=0
        reward=0
        if sel_block==None and self.euclidean_dist(self.block_dict[prev_state[-1][0]],self.block_dict[prev_state[-1][1]])>5000:
            reward -=0.1
        if any([block_states[i]!=prev_state[i] for i in range(1,len(block_states))]) or type (block_states[0])!=type(prev_state[0]) or block_states[0]!=prev_state[0]:
            for stack_num in self.target_blocks:
                for key in self.target_blocks[stack_num]:
                    if self.block_dict[key].rect.centerx==self.block_dict[self.target_blocks[stack_num][key]].rect.centerx and \
                            self.block_dict[key].rect.centery-self.block_dict[self.target_blocks[stack_num][key]].rect.centery==self.block_size:
                        score+=1
                    if sel_block==key or sel_block==self.target_blocks[stack_num][key]:
                        flag2+=1
                if score>0:
                    return 10*(5*score)*num_stacks_aligned+reward

        else:
            return -0.1 + reward

        return reward

    def get_sparse_reward_for_state(self, block_states):
        goal_config = block_states[-1]
        score = 0
        block_size = self.block_size
        for stack in goal_config:
            for i in range(1, len(stack)):
                curr_block, prev_block = block_states[stack[i]], block_states[stack[i - 1]]
                this_score = - np.abs(curr_block[0] - prev_block[0]) - np.abs(prev_block[1] - curr_block[1] - block_size)
                score += this_score
        if score == 0:
            return 1000
        else:
            return 0

    def get_dense_reward_for_state_pramodith(self, block_states):
        goal_config = block_states[-1]
        score = 0
        count = 0.0
        max_x, max_y, block_size = self.screen_width, self.screen_height, self.block_size
        for stack in goal_config:
            for i in range(1, len(stack)):
                curr_block, prev_block = self.block_dict[stack[i]].rect, self.block_dict[stack[i-1]].rect
                this_score = max_x + max_y - np.abs(curr_block.centerx - prev_block.centerx) - np.abs(
                    prev_block.centerx - curr_block.centerx - block_size)
                this_score /= (max_x + max_y)
                score += this_score
                count += 1
        return score / count

    def get_dense_reward_for_state(self, block_states):
        goal_config = block_states[-1]
        score = 0
        count = 0.0
        max_x, max_y, block_size = self.screen_width, self.screen_height, self.block_size
        for stack in goal_config:
            for i in range(1, len(stack)):
                curr_block, prev_block = block_states[stack[i]], block_states[stack[i - 1]]
                this_score = max_x + max_y - np.abs(curr_block[0] - prev_block[0]) - np.abs(
                    prev_block[1] - curr_block[1] - block_size)
                this_score /= (max_x + max_y)
                score += this_score
                count += 1
        return score / count

    def get_reward(self):
        block_states = {idx: (self.block_dict[idx].rect.center[0], self.block_dict[idx].rect.center[1])
                        for idx in self.block_dict}
        return self.get_reward_for_state(block_states, self.goal_config) +self.get_dense_reward_for_state(block_states)

    def get_state_as_tuple(self):
        # curr_state is a n-tuple( (x1, y1), (x2, y2), (x3, y3), (x4, y4), selectedBlockId, (goal_config))
        some_list = [0 for _ in range(self.num_blocks + 1)]
        for block_id in self.block_dict:
            some_list[block_id] = (self.block_dict[block_id].rect.centerx, self.block_dict[block_id].rect.centery)
        some_list[-1] = self.selected_block_id
        some_list.append(tuple([tuple(x) for x in self.goal_config]))
        return tuple(some_list)

    def get_state_as_tuple_pramodith(self):
        # curr_state is a n-tuple( (x1, y1), (x2, y2), (x3, y3), (x4, y4), selectedBlockId, (goal_config))
        some_list = [-1 for _ in range(4)]
        ind=0
        # distances.append(np.square(self.block_dict[block_id].rect.centerx-self.block_dict[block_id2].rect.centerx)+\
        #                       np.square(self.block_dict[block_id].rect.centery-self.block_dict[block_id2].rect.centery))

        #for block_id in sorted(self.block_dict.keys()):
        directions = ["-", "-","-"]
        if self.selected_block_id!=None:
            for stack_num in self.target_blocks:
                if self.selected_block_id in self.target_blocks[stack_num]:
                    target_id=self.target_blocks[stack_num][self.selected_block_id]
                    some_list[0] = self.euclidean_dist(self.block_dict[self.selected_block_id],self.block_dict[target_id])
                    if self.block_dict[self.selected_block_id].rect.centerx-self.block_dict[target_id].rect.centerx>0:
                        directions[0]='l'
                    elif self.block_dict[self.selected_block_id].rect.centerx-self.block_dict[target_id].rect.centerx<0:
                        directions[0]='r'
                    if self.block_dict[self.selected_block_id].rect.centery-self.block_dict[target_id].rect.centery>0:
                        directions[1]='u'
                    elif self.block_dict[self.selected_block_id].rect.centery-self.block_dict[target_id].rect.centery<0:
                        directions[1]='d'
                    other_block= list(set([i for i in range(self.num_blocks)]).difference(set([self.selected_block_id,target_id])))[0]
                    if self.block_dict[other_block].rect.centerx==self.block_dict[target_id].rect.centerx and self.block_dict[other_block].rect.centerx-50==self.block_dict[other_block].rect.centerx:
                        directions[2]="b"


                else:
                    for stack_num in self.target_blocks:
                        for key,value in self.target_blocks[stack_num].items():
                            if value==self.selected_block_id:
                                target_id=key
                                some_list[0] = self.euclidean_dist(self.block_dict[self.selected_block_id],self.block_dict[target_id]) - self.block_size/2
                                if self.block_dict[self.selected_block_id].rect.centerx - self.block_dict[
                                    target_id].rect.centerx > 0:
                                    directions[0] = 'l'
                                elif self.block_dict[self.selected_block_id].rect.centerx - self.block_dict[target_id].rect.centerx < 0:
                                    directions[0] = 'r'
                                if self.block_dict[self.selected_block_id].rect.centery - self.block_dict[target_id].rect.centery > 0:
                                    directions[1] = 'u'
                                elif self.block_dict[self.selected_block_id].rect.centery - self.block_dict[target_id].rect.centery < 0:
                                    directions[1] = 'd'
                                other_block = list(set([i for i in range(self.num_blocks)]).difference(set([self.selected_block_id, target_id])))[0]
                                if self.block_dict[other_block].rect.centerx == self.block_dict[target_id].rect.centerx and self.block_dict[other_block].rect.centerx +50 == self.block_dict[other_block].rect.centerx:
                                    directions[2] = "b"
                    for key in self.target_blocks[stack_num]:
                        some_list[1]+=self.euclidean_dist(self.block_dict[key],self.block_dict[self.target_blocks[stack_num][key]])
        else:
            distances=[]
            for stack_num in self.target_blocks:
                for key in self.target_blocks[stack_num]:
                    distances.append(self.euclidean_dist(self.block_dict[key],self.block_dict[self.target_blocks[stack_num][key]]))

            some_list[0]=tuple(distances)

        some_list[2]=tuple(directions)
        some_list[-1] = self.selected_block_id
        some_list.append(tuple([tuple(x) for x in self.goal_config]))
        return tuple(some_list)

    def get_state_as_tuple_pramodith2(self):
        # curr_state is a n-tuple( (x1, y1), (x2, y2), (x3, y3), (x4, y4), selectedBlockId, (goal_config))
        some_list = [0 for _ in range(4)]
        ind = 0
        # distances.append(np.square(self.block_dict[block_id].rect.centerx-self.block_dict[block_id2].rect.centerx)+\
        #                       np.square(self.block_dict[block_id].rect.centery-self.block_dict[block_id2].rect.centery))

        # for block_id in sorted(self.block_dict.keys()):
        directions = ["-", "-","-"]
        target_id=None
        source_id=None
        if self.selected_block_id != None:
            source_id=self.selected_block_id
            for stack_num in self.target_blocks:
                if self.selected_block_id in self.target_blocks[stack_num]:
                    target_id = self.target_blocks[stack_num][self.selected_block_id]
                    some_list[0] = self.euclidean_dist(self.block_dict[self.selected_block_id],self.block_dict[target_id])
                    if self.block_dict[self.selected_block_id].rect.centerx - self.block_dict[
                        target_id].rect.centerx > 0:
                        directions[0] = 'l'
                    elif self.block_dict[self.selected_block_id].rect.centerx - self.block_dict[
                        target_id].rect.centerx < 0:
                        directions[0] = 'r'
                    if self.block_dict[self.selected_block_id].rect.centery - self.block_dict[
                        target_id].rect.centery > 0:
                        directions[1] = 'u'
                    elif self.block_dict[self.selected_block_id].rect.centery - self.block_dict[
                        target_id].rect.centery < 0:
                        directions[1] = 'd'
                    #other_block = \
                    #list(set([i for i in range(self.num_blocks)]).difference(set([self.selected_block_id, target_id])))[
                    #    0]
                    #if self.block_dict[other_block].rect.centerx == self.block_dict[target_id].rect.centerx and \
                    #        self.block_dict[other_block].rect.centery - 50 == self.block_dict[target_id].rect.centery:
                    #    directions[2] = "b"
                else:
                    target_id=self.selected_block_id
                    for stack_num in self.target_blocks:
                        for key, value in self.target_blocks[stack_num].items():
                            if value == self.selected_block_id:
                                source_id = key
                                some_list[0] = self.euclidean_dist(self.block_dict[source_id],self.block_dict[target_id])
                                if self.block_dict[self.selected_block_id].rect.centerx - self.block_dict[
                                    source_id].rect.centerx > 0:
                                    directions[0] = 'l'
                                elif self.block_dict[self.selected_block_id].rect.centerx - self.block_dict[
                                    source_id].rect.centerx < 0:
                                    directions[0] = 'r'
                                if self.block_dict[self.selected_block_id].rect.centery - self.block_dict[
                                    source_id].rect.centery > 0:
                                    directions[1] = 'u'
                                elif self.block_dict[self.selected_block_id].rect.centery - self.block_dict[
                                    source_id].rect.centery < 0:
                                    directions[1] = 'd'
                                #other_block = list(set([i for i in range(self.num_blocks)]).difference(
                                #    set([self.selected_block_id, target_id])))[0]
                                #if self.block_dict[other_block].rect.centerx == self.block_dict[
                                #    target_id].rect.centerx and self.block_dict[other_block].rect.centery + 50 == \
                                #        self.block_dict[target_id].rect.centery:
                                #    directions[2] = "b"
                for key in self.target_blocks[stack_num]:
                    #print(self.block_dict[key].rect.centerx, self.block_dict[key].rect.centery)
                    #print(self.block_dict[self.target_blocks[stack_num][key]].rect.centerx,
                    #      self.block_dict[self.target_blocks[stack_num][key]].rect.centery)
                    some_list[1] += self.euclidean_dist(self.block_dict[key],self.block_dict[self.target_blocks[stack_num][key]])
        else:
            distances = []
            for stack_num in self.target_blocks:
                for key in self.target_blocks[stack_num]:
                    distances.append(self.euclidean_dist(self.block_dict[key],
                                                         self.block_dict[self.target_blocks[stack_num][key]]))

            some_list[0] = tuple(distances)

        some_list[2] = tuple(directions)
        some_list[-1] = self.selected_block_id
        some_list.append((source_id,target_id))

        return tuple(some_list)

    def get_state_as_dict(self):
        # curr_state is a n-tuple( (x1, y1), (x2, y2), (x3, y3), (x4, y4), selectedBlockId)
        block_dict = self.block_dict
        state = {
            "positions": {block_id: (block_dict[block_id].rect.centerx, block_dict[block_id].rect.centery) for block_id
                          in block_dict},
            "selected": self.selected_block_id if self.selected_block_id is not None else -1
        }
        return state

    @staticmethod
    def are_intersecting(rect1, dx, dy, other_rect):
        return (other_rect.top <= rect1.top + dy < other_rect.bottom
                and (other_rect.left <= rect1.left + dx < other_rect.right
                     or other_rect.left < rect1.right + dx <= other_rect.right)) \
               or (other_rect.top < rect1.bottom + dy <= other_rect.bottom
                   and (other_rect.left <= rect1.left + dx < other_rect.right
                        or other_rect.left < rect1.right + dx <= other_rect.right)) or (rect1.top+dy<0 or rect1.bottom+dy>350 or rect1.left+dx<0 or rect1.right+dx>350 )

    def create_blocks(self):
        blocks = pygame.sprite.Group()
        #grid_centers=[(325,325),(25,25)]
        grid_centers=[(25+50*np.random.randint(6),25+50*np.random.randint(6)) for _ in range(self.num_blocks)]
        #grid_centers = [(i, self.screen_height // 2) for i in range(25, self.screen_width, 50)]
        #for i,blockCenterIdx in enumerate(range(len(grid_centers))):
        for i, blockCenterIdx in enumerate(np.random.choice(len(grid_centers), self.num_blocks, replace=False)):
            blocks.add(Block(i, grid_centers[blockCenterIdx], self.block_size))
            #blocks.add(Block(i, (325,325), self.block_size))
        block_dict = {block.id: block for block in blocks.sprites()}
        return blocks, block_dict

    def get_next_state_based_on_state_tuple(self, state, action):
        # action is (Action, blockId)
        # print("get_next_state_based_on_state_tuple: ", state, action)
        sel_block_id = state[-2]
        if sel_block_id: assert sel_block_id < self.block_size
        state_l = list(state)
        if action[0] == Action.DROP:
            state_l[-2] = None
            state_l[2]=('-','-')
            distances=[]
            for stack_num in self.target_blocks:
                for key in self.target_blocks[stack_num]:
                    distances.append(self.euclidean_dist(self.block_dict[key], self.block_dict[self.target_blocks[stack_num][key]]))
            state_l[0] = tuple(distances)
            state_l[1]=0
            state_l[-1]=(None,None)
        else:
            state_l[-2] = action[1]
            state_l=self.get_next_state_pramodith(action[0],state_l[-2],state_l)
        return tuple(state_l)

    def get_next_state(self, action, sel_block_id):
        if action in move_action_to_deviation:
            dx, dy = move_action_to_deviation[action]
        else:
            raise IOError("Invalid Action", action)
        rectangle = self.block_dict[sel_block_id].rect
        not_intersections = [not BlockWorld.are_intersecting(rectangle, dx, dy, other_block.rect) for other_block in
                             self.blocks if other_block.rect != rectangle]
        orig_pos = rectangle.center
        if all(not_intersections):
            next_pos = (orig_pos[0] + dx, orig_pos[1] + dy)
            if self.is_in_bounding_box(next_pos):
                return next_pos
        return orig_pos

    def get_next_state_pramodith(self, action, sel_block_id,state_l):
        dx=None
        dy=None
        if action in move_action_to_deviation:
            dx, dy = move_action_to_deviation[action]
        else:
            raise IOError("Invalid Action", action)
        rectangle = self.block_dict[sel_block_id].rect
        not_intersections = [not BlockWorld.are_intersecting(rectangle, dx, dy, other_block.rect) for other_block in
                             self.blocks if other_block.rect != rectangle]
        orig_pos = rectangle.center
        if all(not_intersections):
            state_l[1] = 0

            distances=[]
            if dx!=0 or dy!=0:
                self.block_dict[sel_block_id].rect.centerx+=dx
                self.block_dict[sel_block_id].rect.centery+=dy
            if self.selected_block_id==None:
                self.selected_block_id=sel_block_id
            directions = ["-", "-","-"]
            source_id=None
            target_id=None
            state_l[-1]=(source_id,target_id)

            for stack_num in self.target_blocks:
                if sel_block_id in self.target_blocks[stack_num]:
                    source_id=sel_block_id
                    target_id = self.target_blocks[stack_num][sel_block_id]
                    state_l[0] = self.euclidean_dist(self.block_dict[sel_block_id],self.block_dict[target_id])
                    if self.block_dict[self.selected_block_id].rect.centerx - self.block_dict[target_id].rect.centerx > 0:
                        directions[0] = 'l'
                    elif self.block_dict[self.selected_block_id].rect.centerx - self.block_dict[target_id].rect.centerx < 0:
                        directions[0] = 'r'
                    if self.block_dict[self.selected_block_id].rect.centery - self.block_dict[target_id].rect.centery > 0:
                        directions[1] = 'u'
                    elif self.block_dict[self.selected_block_id].rect.centery - self.block_dict[target_id].rect.centery < 0:
                        directions[1] = 'd'
                    #other_block = \
                    #list(set([i for i in range(self.num_blocks)]).difference(set([self.selected_block_id, target_id])))[0]
                    #if self.block_dict[other_block].rect.centerx == self.block_dict[target_id].rect.centerx and \
                    #        self.block_dict[other_block].rect.centery - 50 == self.block_dict[target_id].rect.centery:
                    #    directions[2] = "b"
                else:
                    target_id = self.selected_block_id
                    for stack_num in self.target_blocks:
                        for key, value in self.target_blocks[stack_num].items():
                            if value == self.selected_block_id:
                                source_id = key
                                state_l[0] = self.euclidean_dist(self.block_dict[source_id],
                                                                   self.block_dict[target_id])
                                if self.block_dict[self.selected_block_id].rect.centerx - self.block_dict[source_id].rect.centerx > 0:
                                    directions[0] = 'l'
                                elif self.block_dict[self.selected_block_id].rect.centerx - self.block_dict[source_id].rect.centerx < 0:
                                    directions[0] = 'r'
                                if self.block_dict[self.selected_block_id].rect.centery - self.block_dict[source_id].rect.centery > 0:
                                    directions[1] = 'u'
                                elif self.block_dict[self.selected_block_id].rect.centery - self.block_dict[source_id].rect.centery < 0:
                                    directions[1] = 'd'
                                #other_block = list(set([i for i in range(self.num_blocks)]).difference(
                                #    set([self.selected_block_id, target_id])))[0]
                                #if self.block_dict[other_block].rect.centerx == self.block_dict[
                                #    target_id].rect.centerx and self.block_dict[other_block].rect.centery + 50 == \
                                #        self.block_dict[target_id].rect.centery:
                                #    directions[2] = "b"


                for key in self.target_blocks[stack_num]:
                    #print(self.block_dict[key].rect.centerx, self.block_dict[key].rect.centery)
                    #print(self.block_dict[self.target_blocks[stack_num][key]].rect.centerx, self.block_dict[self.target_blocks[stack_num][key]].rect.centery)
                    state_l[1] += self.euclidean_dist(self.block_dict[key],self.block_dict[self.target_blocks[stack_num][key]])

            state_l[2] = tuple(directions)
            state_l[-1]=(source_id, target_id)

        return state_l


    def update_state_from_tuple_pramodith(self, state_tuple):
        sel_block_id = state_tuple[-2]
        if sel_block_id is not None:
            self.selected_block_id = sel_block_id
        else:
            self.selected_block_id = None

    def update_state_from_tuple(self, state_tuple):
        sel_block_id = state_tuple[-2]
        if sel_block_id is not None:
            self.block_dict[sel_block_id].rect.centerx = state_tuple[sel_block_id][0]
            self.block_dict[sel_block_id].rect.centery = state_tuple[sel_block_id][1]
            self.selected_block_id = sel_block_id
        else:
            self.selected_block_id = None

    def update_state(self, sel_block_id, next_state):
        self.block_dict[sel_block_id].rect.centerx = next_state[0]
        self.block_dict[sel_block_id].rect.centery = next_state[1]

    def move_block_by_key(self, key, sel_block_id):
        return self.move_block_by_action(key_to_action[key], sel_block_id)

    def move_block_by_action(self, action, sel_block_id):
        next_state = self.get_next_state(action, sel_block_id)
        self.update_state(sel_block_id, next_state)
        return action, sel_block_id

    def pre_render(self, drop_events=True):
        self.screen.blit(self.master_surface, (0, 0))

        # rendering the goal screen
        self.screen.blit(self.goal_surface, (self.screen_width - self.goal_screen_dim[0], 0))

        if drop_events:
            pygame.event.get()

    def render(self, filename=None):
        for block in self.blocks:
            self.screen.blit(block.surf, block.rect)
        pygame.display.flip()
        if filename:
            pygame.image.save(self.screen, filename)

    def is_in_bounding_box(self, next_pos):
        return (self.block_size / 2) <= next_pos[0] <= (self.screen_width - self.block_size / 2) \
               and (self.block_size / 2) <= next_pos[1] <= (self.screen_height - self.block_size / 2)

    def record_action(self, state=None, action=None):
        action_value = action.value
        if action == Action.PICK:
            action_value = "%s-%d" % (action_value, self.selected_block_id)
        if state is None:
            self.actions_taken.append({"state": self.get_state_as_dict(), "action": action_value})
        else:
            self.actions_taken.append({"state": state, "action": action_value})

    def run_environment(self):
        running = True

        # Required for DQN to map frames to actions.

        # Create the surface and pass in a tuple with its length and width
        # to indicate the end of the demonstration
        prev_action_key = None

        while running:
            self.pre_render(False)
            for event in pygame.event.get():
                state = self.get_state_as_dict()
                if event.type == KEYUP:
                    prev_action_key = None
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        prev_action_key = None
                        running = False

                    elif event.key == K_RSHIFT:
                        prev_action_key = None
                        self.record_action(state=state, action=Action.FINISHED)
                        print("Finished Demonstration:", self.actions_taken)
                        running = False

                    elif event.key == K_SPACE:
                        self.block_dict[self.selected_block_id].surf.fill(COLORS[self.selected_block_id])
                        self.record_action(state=state, action=Action.DROP)
                        self.selected_block_id = None

                    elif event.key in {K_UP, K_DOWN, K_LEFT, K_RIGHT}:
                        prev_action_key = event.key

                # Check for QUIT event; if QUIT, set running to false
                # elif event.type == QUIT or (hasattr(event, "key") and event.key == K_ESCAPE):
                #     print("writing to file")
                #     prev_action_key = None
                #     self.record_action(action=Action.FINISHED)
                #     # self.serialize_actions()
                #     print("Finished Demonstration:", self.actions_taken)
                #     running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    prev_action_key = None
                    pos = pygame.mouse.get_pos()
                    for block in self.blocks:
                        if block.rect.collidepoint(pos):
                            if self.selected_block_id is not None:
                                self.record_action(state=state, action=Action.DROP)
                                self.block_dict[self.selected_block_id].surf.fill(COLORS[self.selected_block_id])
                            self.selected_block_id = block.id
                            self.record_action(state=state, action=Action.PICK)
                            self.block_dict[block.id].surf.fill(WHITE)
                            break

                if prev_action_key:
                    action_taken = self.move_block_by_key(prev_action_key, self.selected_block_id)
                    if action_taken:
                        self.record_action(state=state, action=action_taken[0])

                self.render()
        return self.actions_taken

    @staticmethod
    def convert_state_dict_to_tuple(state_dict):
        state = [tuple(state_dict["positions"][key]) for key in sorted([key for key in state_dict["positions"]], key=lambda x: int(x))]
        selected_id = state_dict["selected"] if state_dict["selected"] != -1 else None
        state.append(selected_id)
        return tuple(state)

    @staticmethod
    def parse_action(action_value):
        action_vals = action_value.split("-")
        action = Action(action_vals[0])
        if len(action_vals) > 1:
            return action, int(action_vals[1])
        else:
            return action, None
