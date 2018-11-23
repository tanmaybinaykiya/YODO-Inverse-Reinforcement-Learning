import json
import os
import random
import time

import numpy as np
import pygame
from pygame.locals import *

from Block import Block
from constants import *


class BlockWorld:

    def __init__(self, screen_width, screen_height, num_blocks=3, num_stacks=1, block_size=50, record=False):

        self.timer_start = time.time()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.reward = 0
        self.block_size = block_size
        self.goal_config = -np.ones((self.num_stacks, self.num_blocks), dtype=np.int8)
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.master_surface = pygame.Surface(self.screen.get_size())
        self.master_surface.fill((0, 0, 0))
        self.grid_centers = [(i, 500) for i in range(25, screen_width, 50)]
        self.blocks = pygame.sprite.Group()
        self.create_blocks(num_blocks, block_size)
        self.block_dict = {block.id: block for block in self.blocks.sprites()}
        self.demo_id = len(list(os.walk(FRAME_LOCATION)))
        self.demo_dir = os.path.join(FRAME_LOCATION, str(self.demo_id))
        self.record = record
        self.selected_block_id = None
        os.mkdir(self.demo_dir)
        self.state_action_map = self.load_state_dict()
        pygame.init()
        self.create_goal()

    def load_state_dict(self):
        if os.path.isfile("state_action_map.json"):
            with open("state_action_map.json", 'r') as f:
                state_action_map = json.loads(f.read())
                state_action_map[self.demo_id] = {}
        else:
            state_action_map = {self.demo_id: {}}
        return state_action_map

    @staticmethod
    def distance(pts1, pts2):
        return (pts1[0] - pts2[0]) ** 2 + (pts1[1] - pts2[1]) ** 2

    def create_goal(self, block_size=30, goal_screen=(200, 200)):

        # choosing the order for blocks to be placed in the goal screen.
        block_order = [i for i in range(self.num_blocks)]
        seed = np.random.randint(0, self.num_stacks)
        random.shuffle(block_order)

        self.goal_surface = pygame.Surface(goal_screen)
        self.goal_surface.fill(GRAY)
        last_used_block = 0
        blocks_per_stack = self.num_blocks // self.num_stacks
        for stack_num in range(self.num_stacks):
            for i in range(blocks_per_stack):
                pygame.draw.rect(self.goal_surface, COLORS[block_order[last_used_block]],
                                 (stack_num * 35 + 40, 150 - block_size * i, block_size, block_size))
                self.goal_config[stack_num][i] = block_order[last_used_block]
                last_used_block += 1

        if self.num_blocks % 2 == 1 and last_used_block != self.num_blocks:
            while last_used_block != self.num_blocks:
                pygame.draw.rect(self.goal_surface, COLORS[block_order[last_used_block]],
                                 seed * 35 + 40, 150 - block_size * blocks_per_stack)
                self.goal_config[seed][np.where(self.goal_config[seed] == -1)[0][0]] = block_order[last_used_block]
                last_used_block += 1
                blocks_per_stack += 1

    def get_reward_for_state(self, block_states, goal_config):
        score = 0
        max_x, max_y, block_size = self.screen_width, self.screen_height, self.block_size
        for stack in goal_config:
            for i in range(1, len(stack)):
                curr_block, prev_block = block_states[stack[i]], block_states[stack[i - 1]]
                this_score = max_x + max_y - np.abs(curr_block[0] - prev_block[0]) - np.abs(
                    prev_block[1] - curr_block[1] - block_size)
                score += this_score
        return score

    def get_reward_for_state_tuple(self, block_states, goal_config):
        score = 0
        max_x, max_y, block_size = self.screen_width, self.screen_height, self.block_size
        for stack in goal_config:
            for i in range(1, len(stack)):
                curr_block, prev_block = block_states[stack[i]], block_states[stack[i - 1]]
                this_score = max_x + max_y - np.abs(curr_block[0] - prev_block[0]) - np.abs(
                    prev_block[1] - curr_block[1] - block_size)
                score += this_score
        return score

    def get_reward(self):
        block_states = {idx: (self.block_dict[idx].rect.center[0], self.block_dict[idx].rect.center[1])
                        for idx in self.block_dict}
        return self.get_reward_for_state(block_states, self.goal_config.tolist())

    def get_state_as_tuple(self):

        # curr_state is a n-tuple( (x1, y1), (x2, y2), (x3, y3), (x4, y4), selectedBlockId)
        some_list = [0 for _ in range(self.num_blocks + 1)]
        for block_id in self.block_dict:
            some_list[block_id] = (self.block_dict[block_id].rect.centerx, self.block_dict[block_id].rect.centery)
        some_list[-1] = self.selected_block_id

        return tuple(some_list)

    @staticmethod
    def are_intersecting(rect1, dx, dy, other_rect):
        return (other_rect.top <= rect1.top + dy < other_rect.bottom
                and (other_rect.left <= rect1.left + dx < other_rect.right
                     or other_rect.left < rect1.right + dx <= other_rect.right)) \
               or (other_rect.top < rect1.bottom + dy <= other_rect.bottom
                   and (other_rect.left <= rect1.left + dx < other_rect.right
                        or other_rect.left < rect1.right + dx <= other_rect.right))

    def create_blocks(self, num_blocks, block_size):
        for i, blockCenterIdx in enumerate(np.random.choice(len(self.grid_centers), num_blocks, replace=False)):
            self.blocks.add(Block(i, self.grid_centers[blockCenterIdx], block_size))
        pygame.display.flip()

    def move_block(self, key, sel_block_id):
        if key in key_to_action:
            action = key_to_action[key]
            return self.move_block_by_action[action, sel_block_id]
        else:
            raise IOError("Invalid Key", key)

    def get_next_state_based_on_state_tuple(self, state, action):
        # action is (Action, blockId)
        print("get_next_state_based_on_state_tuple: ", state, action)
        sel_block_id = state[-1]
        state_l = list(state)
        if action[0] == Action.PICK:
            state_l[-1] = action[1]
        elif action[0] == Action.DROP:
            state_l[-1] = None
        else:
            state_l[sel_block_id] = self.get_next_state(action[0], sel_block_id)

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

    def update_state_from_tuple(self, state_tuple):
        sel_block_id = state_tuple[-1]
        if sel_block_id is not None:
            self.block_dict[sel_block_id].rect.centerx = state_tuple[sel_block_id][0]
            self.block_dict[sel_block_id].rect.centery = state_tuple[sel_block_id][1]
            self.selected_block_id = sel_block_id
        else:
            self.selected_block_id = None

    def update_state(self, sel_block_id, next_state):
        self.block_dict[sel_block_id].rect.centerx = next_state[0]
        self.block_dict[sel_block_id].rect.centery = next_state[1]

    def move_block_by_action(self, action, sel_block_id):
        next_state = self.get_next_state(action, sel_block_id)
        self.update_state(sel_block_id, next_state)
        return action, sel_block_id

    def run_environment(self):
        actions_taken = []
        running = True
        frame_num = 0
        done = False
        # Required for DQN to map frames to actions.

        # Create the surface and pass in a tuple with its length and width
        sel_block_id = None
        prev_action = None

        while running:
            most_recent_action = None
            self.prerender()

            # for loop through the event queue
            for event in pygame.event.get():
                # Our main loop!
                if event.type == KEYUP:
                    prev_action = None
                # Check for KEYDOWN event; KEYDOWN is a constant defined in pygame.locals, which we imported earlier
                if event.type == KEYDOWN:
                    # If the Esc key has been pressed set running to false to exit the main loop
                    if event.key == K_ESCAPE:
                        prev_action = None
                        running = False

                    elif event.key == K_RSHIFT:
                        prev_action = None
                        actions_taken.append((Action.FINISHED, None))
                        most_recent_action = action_to_ind[Action.FINISHED.value]
                        done = 1
                        print("Finished Demonstration:", actions_taken)

                    elif event.key == K_SPACE:
                        print("Dropped")
                        self.selected_block_id = None
                        self.block_dict[sel_block_id].surf.fill(COLORS[sel_block_id])
                        actions_taken.append((Action.DROP, sel_block_id))
                        most_recent_action = action_to_ind[Action.DROP.value]

                    elif event.key in {K_UP, K_DOWN, K_LEFT, K_RIGHT}:
                        prev_action = event.key

                # Check for QUIT event; if QUIT, set running to false
                elif event.type == QUIT or (hasattr(event, "key") and event.key == K_ESCAPE):
                    print("writing to file")
                    prev_action = None
                    actions_taken.append((Action.FINISHED, None))
                    most_recent_action = Action.FINISHED
                    with open("state_action_map.json", 'w') as f:
                        json.dump(self.state_action_map, f, indent=4)
                    print("Finished Demonstration:", actions_taken)
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    prev_action = None
                    pos = pygame.mouse.get_pos()
                    for block in self.blocks:
                        if block.rect.collidepoint(pos):
                            self.selected_block_id = block.id
                            sel_block_id = block.id
                            print("Frame: %d" % frame_num)
                            actions_taken.append((Action.PICK, block.id))
                            most_recent_action = action_to_ind[Action.PICK.value + str(block.id)]
                            self.block_dict[block.id].surf.fill(WHITE)
                            break

            if prev_action:
                action_taken = self.move_block(prev_action, sel_block_id)
                if action_taken:
                    actions_taken.append(action_taken)
                    most_recent_action = action_to_ind[action_taken[0].value.split("_")[1]]

            if most_recent_action is not None:
                self.state_action_map[self.demo_id][frame_num] = [most_recent_action, self.reward, done]
            filename = self.demo_dir + "/%d.png" % frame_num if self.record else None
            self.render(filename)

    def prerender(self):
        self.screen.blit(self.master_surface, (0, 0))

        # rendering the goal screen
        self.screen.blit(self.goal_surface, (800, 0))

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
