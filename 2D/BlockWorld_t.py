import os
import random

import numpy as np
import pygame
from pygame.locals import *

from Block import Block
from Oracle import State
from constants import *


class BlockWorld:

    def __init__(self, screen_width, screen_height, num_blocks=3, num_stacks=1, block_size=50, record=False):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.goal_screen_dim = (self.screen_width // 5, self.screen_width // 5)
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.block_size = block_size

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
        block_order = goal_config[0] # [i for i in range(self.num_blocks)]
        seed = np.random.randint(0, self.num_stacks)
        # random.shuffle(block_order)
        last_used_block = 0
        blocks_per_stack = self.num_blocks // self.num_stacks
        block_size = self.goal_screen_dim[0] // 10
        if self.num_stacks > 1:
            left_padding = (self.goal_screen_dim[0] - block_size + (3 * (self.num_stacks - 1) * block_size // 2)) // 2
        else:
            left_padding = (self.goal_screen_dim[0] - block_size) // 2

        bottom = self.goal_screen_dim[1] - 2 * block_size
        for stack_num in range(self.num_stacks):
            for i in range(blocks_per_stack):
                pygame.draw.rect(self.goal_surface, COLORS[block_order[last_used_block]], (stack_num * (block_size + 5) + left_padding, bottom - block_size * i, block_size, block_size))
                goal_config[stack_num][i] = block_order[last_used_block]
                last_used_block += 1

        if self.num_blocks % 2 == 1 and last_used_block != self.num_blocks:
            while last_used_block != self.num_blocks:
                pygame.draw.rect(self.goal_surface, COLORS[block_order[last_used_block]], seed * 35 + 40, 150 - block_size * blocks_per_stack)
                goal_config[seed][np.where(goal_config[seed] == -1)[0][0]] = block_order[last_used_block]
                last_used_block += 1
                blocks_per_stack += 1
        return goal_config

    def get_reward_for_state(self, block_states):
        return self.get_sparse_reward_for_state(block_states)

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
            return 1
        else:
            return 0

    def get_dense_reward_for_state(self, block_states):
        goal_config = block_states[-1]
        score = 0
        count = 0.0
        max_x, max_y, block_size = self.screen_width, self.screen_height, self.block_size
        for stack in goal_config:
            for i in range(1, len(stack)):
                curr_block, prev_block = block_states[stack[i]], block_states[stack[i - 1]]
                this_score = max_x + max_y - np.abs(curr_block[0] - prev_block[0]) - np.abs(prev_block[1] - curr_block[1] - block_size)
                this_score /= (max_x + max_y)
                score += this_score
                count += 1
        return score / count

    def get_reward(self):
        block_states = {idx: (self.block_dict[idx].rect.center[0], self.block_dict[idx].rect.center[1]) for idx in self.block_dict}
        return self.get_reward_for_state(block_states, self.goal_config)

    def get_state_as_tuple(self):
        # curr_state is a n-tuple( (x1, y1), (x2, y2), (x3, y3), (x4, y4), selectedBlockId, (goal_config))
        some_list = [0 for _ in range(self.num_blocks + 1)]
        for block_id in self.block_dict:
            some_list[block_id] = (self.block_dict[block_id].rect.centerx, self.block_dict[block_id].rect.centery)
        some_list[-1] = self.selected_block_id
        some_list.append(tuple([tuple(x) for x in self.goal_config]))
        return tuple(some_list)

    def get_state_as_dict(self):
        # curr_state is a n-tuple( (x1, y1), (x2, y2), (x3, y3), (x4, y4), selectedBlockId)
        block_dict = self.block_dict
        state = {"positions": {block_id: (block_dict[block_id].rect.centerx, block_dict[block_id].rect.centery) for block_id in block_dict}, "selected": self.selected_block_id if self.selected_block_id is not None else -1}
        return state

    @staticmethod
    def are_intersecting(rect1, dx, dy, other_rect):
        return (other_rect.top <= rect1.top + dy < other_rect.bottom and (other_rect.left <= rect1.left + dx < other_rect.right or other_rect.left < rect1.right + dx <= other_rect.right)) or (other_rect.top < rect1.bottom + dy <= other_rect.bottom and (other_rect.left <= rect1.left + dx < other_rect.right or other_rect.left < rect1.right + dx <= other_rect.right))

    def create_blocks(self):
        blocks = pygame.sprite.Group()
        grid_centers = [(i, self.screen_height // 2) for i in range(25, self.screen_width, 50)]
        for i, blockCenterIdx in enumerate(np.random.choice(len(grid_centers), self.num_blocks, replace=False)):
            blocks.add(Block(i, grid_centers[blockCenterIdx], self.block_size))
        block_dict = {block.id: block for block in blocks.sprites()}
        return blocks, block_dict

    def get_next_state_based_on_state_tuple(self, state, action):
        # action is (Action, blockId)
        # print("get_next_state_based_on_state_tuple: ", state, action)
        sel_block_id = state[-2]
        if sel_block_id: assert sel_block_id < self.block_size
        state_l = list(state)
        if action[0] == Action.PICK:
            state_l[-2] = action[1]
        elif action[0] == Action.DROP:
            state_l[-2] = None
        else:
            state_l[sel_block_id] = self.get_next_state(action[0], sel_block_id)
        return tuple(state_l)

    def get_state_as_tuple_pramodith(self):
        # curr_state is a n-tuple( (x1, y1), (x2, y2), (x3, y3), (x4, y4), selectedBlockId, (goal_config))
        some_list = [-1 for _ in range(3)]
        ind=0
        # distances.append(np.square(self.block_dict[block_id].rect.centerx-self.block_dict[block_id2].rect.centerx)+\
        #                       np.square(self.block_dict[block_id].rect.centery-self.block_dict[block_id2].rect.centery))

        #for block_id in sorted(self.block_dict.keys()):
        directions = ["-", "-"]
        if self.selected_block_id!=None:
            if self.selected_block_id in self.target_blocks:
                target_id=self.target_blocks[self.selected_block_id]
                some_list[0] = np.square(self.block_dict[self.selected_block_id].rect.centerx - self.block_dict[target_id].rect.centerx) + \
                                np.square(self.block_dict[self.selected_block_id].rect.centery - self.block_dict[target_id].rect.centery)
                if self.block_dict[self.selected_block_id].rect.centerx-self.block_dict[target_id].rect.centerx>0:
                    directions[0]='l'
                elif self.block_dict[self.selected_block_id].rect.centerx-self.block_dict[target_id].rect.centerx<0:
                    directions[0]='r'
                if self.block_dict[self.selected_block_id].rect.centery-self.block_dict[target_id].rect.centery>0:
                    directions[1]='u'
                elif self.block_dict[self.selected_block_id].rect.centery-self.block_dict[target_id].rect.centery<0:
                    directions[1]='d'
            else:
                for key,value in self.target_blocks.items():
                    if value==self.selected_block_id:
                        target_id=key
                        some_list[0] = np.square(self.block_dict[self.selected_block_id].rect.centerx - self.block_dict[target_id].rect.centerx) + \
                                       np.square(self.block_dict[self.selected_block_id].rect.centery - self.block_dict[target_id].rect.centery)
                        if self.block_dict[self.selected_block_id].rect.centerx - self.block_dict[
                            target_id].rect.centerx > 0:
                            directions[0] = 'l'
                        elif self.block_dict[self.selected_block_id].rect.centerx - self.block_dict[target_id].rect.centerx < 0:
                            directions[0] = 'r'
                        if self.block_dict[self.selected_block_id].rect.centery - self.block_dict[target_id].rect.centery > 0:
                            directions[1] = 'u'
                        elif self.block_dict[self.selected_block_id].rect.centery - self.block_dict[target_id].rect.centery < 0:
                            directions[1] = 'd'
        else:
            distances=[]
            for key in self.target_blocks:
                distances.append(self.euclidean_dist(self.block_dict[key],self.block_dict[self.target_blocks[key]]))

            some_list[0]=tuple(distances)

        some_list[1]=tuple(directions)
        some_list[-1] = self.selected_block_id
        some_list.append(tuple([tuple(x) for x in self.goal_config]))


        return tuple(some_list)

    def get_next_state(self, action, sel_block_id):
        if action in move_action_to_deviation:
            dx, dy = move_action_to_deviation[action]
        else:
            raise IOError("Invalid Action", action)
        rectangle = self.block_dict[sel_block_id].rect
        not_intersections = [not BlockWorld.are_intersecting(rectangle, dx, dy, other_block.rect) for other_block in self.blocks if other_block.rect != rectangle]
        orig_pos = rectangle.center
        if all(not_intersections):
            next_pos = (orig_pos[0] + dx, orig_pos[1] + dy)
            if self.is_in_bounding_box(next_pos):
                return next_pos
        return orig_pos

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

    def update_all_states(self, state: State):
        for idx, block in enumerate(state.block_positions):
            self.block_dict[idx].rect.center = tuple(block)
        self.selected_block_id = state.selected_index

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
        return (self.block_size / 2) <= next_pos[0] <= (self.screen_width - self.block_size / 2) and (self.block_size / 2) <= next_pos[1] <= (self.screen_height - self.block_size / 2)

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
