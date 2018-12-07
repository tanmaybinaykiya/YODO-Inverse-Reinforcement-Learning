import os
import random
import time

import numpy as np
import pygame

from Block import Block
from State import State
from constants import *


class BlockWorld:

    def __init__(self, screen_width=500, screen_height=500, init_config=None, goal_config=None, selected_id=None, num_blocks=3, num_stacks=1, block_size=50, record=False):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.goal_screen_dim = (self.screen_width // 5, self.screen_width // 5)
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.block_size = block_size

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

        self.blocks = pygame.sprite.Group()
        self.block_dict = {}

        # if goal_config is not None:
        #     self.goal_config = goal_config
        # else:
        #     self.goal_config = BlockWorld.create_goal(num_blocks)
        self.state = self.initialize_state(init_config, goal_config, selected_id)
        self.create_block_dicts()

        pygame.init()
        self.pre_render(True)
        self.render()
        self.pre_render(True)
        self.selected_block_id = None
        self.actions_taken = []

    def get_screen_dims(self):
        return self.screen_width, self.screen_height

    def get_state(self) -> State:
        return self.state

    def initialize_state(self, init_config, goal_config, selected_id):
        if goal_config is None:
            goal_config = BlockWorld.create_goal(self.num_blocks)

        if init_config is None:
            bs = self.block_size
            init_config = [[bs // 2 + bs * np.random.randint(6), bs // 2 + bs * np.random.randint(6)] for _ in range(self.num_blocks)]

        return State(block_positions=init_config, selected_index=selected_id, goal_config=goal_config)

    def create_block_dicts(self):
        for i, block_position in enumerate(self.state.block_positions):
            this_block = Block(i, (block_position[0], block_position[1]), self.block_size)
            self.blocks.add(this_block)
            self.block_dict[i] = this_block

    def update_block_dicts(self):
        for i in self.block_dict:
            self.block_dict[i].rect.centerx = self.state.get_position(i)[0]
            self.block_dict[i].rect.centery = self.state.get_position(i)[1]

    @staticmethod
    def distance(pts1, pts2):
        return (pts1[0] - pts2[0]) ** 2 + (pts1[1] - pts2[1]) ** 2

    @staticmethod
    def create_goal(num_blocks):
        block_order = list(range(num_blocks))
        random.shuffle(block_order)
        return [block_order]

    # unused
    def pre_render(self, drop_events=False):
        self.screen.blit(self.master_surface, (0, 0))
        self.screen.blit(self.goal_surface, (self.screen_width - self.goal_screen_dim[0], 0))
        if drop_events:
            pygame.event.get()

    def render(self, filename=None):
        self.update_block_dicts()
        self.render_blocks()
        self.render_goal()
        pygame.display.flip()
        if filename and self.record:
            pygame.image.save(self.screen, filename)

    def render_blocks(self):
        for block in self.blocks:
            self.screen.blit(block.surf, block.rect)

    def render_goal(self):
        self.goal_surface.fill(GRAY)
        block_size = self.goal_screen_dim[0] // 10
        if self.num_stacks > 1:
            left_padding = (self.goal_screen_dim[0] - block_size + (3 * (self.num_stacks - 1) * block_size // 2)) // 2
        else:
            left_padding = (self.goal_screen_dim[0] - block_size) // 2

        bottom = self.goal_screen_dim[1] - 2 * block_size
        for stack_num, stack in enumerate(self.state.goal_config):
            for i, block in enumerate(stack):
                pygame.draw.rect(self.goal_surface, COLORS[block], (stack_num * (block_size + 5) + left_padding, bottom - block_size * i, block_size, block_size))
        self.screen.blit(self.goal_surface, (self.screen_width - self.goal_screen_dim[0], 0))

    def get_reward_for_state(self, state:State):
        return self.get_sparse_reward_for_state(state)

    def get_sparse_reward_for_state(self, state:State):
        goal_config = state.goal_config
        score = 0
        block_size = self.block_size
        for stack in goal_config:
            for i in range(1, len(stack)):
                curr_block, prev_block = self.state.get_position(stack[i]), self.state.get_position(stack[i - 1])
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

    def update_block_state_from_state_obj(self):
        sel_block_id = self.state.selected_index
        if sel_block_id is not None:
            self.block_dict[sel_block_id].rect.centerx = self.state.block_positions[sel_block_id][0]
            self.block_dict[sel_block_id].rect.centery = self.state.block_positions[sel_block_id][1]
            self.selected_block_id = sel_block_id
        else:
            self.selected_block_id = None

    def update_state(self, next_state: State):
        self.state = next_state
        self.update_block_state_from_state_obj()

    def get_reward_for_state_action_pramodith(self, state, next_state):
        if state[-2] == next_state[-2]:
            curr_dist = state[0]
            next_dist = next_state[0]
            if next_dist < curr_dist:
                return 1
            elif next_dist == curr_dist:
                return 0
            else:
                return -1
        return 0


if __name__ == '__main__':
    BlockWorld(screen_width=500, screen_height=500)
