import json
import os
import random
import time
from collections import defaultdict

import cv2
import numpy as np
import pygame
import torch
from PIL import Image
from pygame.locals import *

from Block import Block
from DQN import DQN
from constants import *


class BlockWorld:

    def __init__(self, screen_width, screen_height, num_blocks=2, num_stacks=2, block_size=50):

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
        os.mkdir(self.demo_dir)
        self.state_action_map = self.load_state_dict()
        pygame.init()

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

    def create_goal(self, num_stacks, block_order, seed, block_size=30, goal_screen=(200, 200)):
        self.goal_surface = pygame.Surface(goal_screen)
        self.goal_surface.fill(GRAY)
        last_used_block = 0
        blocks_per_stack = self.num_blocks // num_stacks
        for stack_num in range(num_stacks):
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
        print(self.goal_config)

    def get_reward_per_state(self):
        print([COLORS_STR[i] for stack in self.goal_config for i in stack])
        score = 0
        max_x, max_y, block_size = self.screen_width, self.screen_height, self.block_size
        for stack in self.goal_config.tolist():
            for i in range(1, len(stack)):
                curr_block, prev_block = self.block_dict[stack[i]].rect.center, self.block_dict[stack[i - 1]].rect.center
                this_score = - np.abs(curr_block[0] - prev_block[0]) - np.abs(prev_block[1] - curr_block[1] - block_size)
                print("Reward[%s, %s]:%d" % (COLORS_STR[i], COLORS_STR[i-1], this_score))
                score += this_score
        return score

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

    def move_block(self, action, drag, rectangle, sel_block_id):
        action_taken = None
        if action == K_UP:
            action_name = Action.MOVE_UP
            dx, dy = 0, -10
        elif action == K_DOWN:
            action_name = Action.MOVE_DOWN
            dx, dy = 0, 10
        elif action == K_LEFT:
            action_name = Action.MOVE_LEFT
            dx, dy = -10, 0
        elif action == K_RIGHT:
            action_name = Action.MOVE_RIGHT
            dx, dy = 10, 0
        else:
            raise IOError("Invalid Action", action)

        if drag:
            all_dists = [not BlockWorld.are_intersecting(rectangle, dx, dy, other_block.rect) for other_block in
                         self.blocks if other_block.rect != rectangle]
            if all(all_dists):
                action_taken = (action_name, sel_block_id)
                rectangle.centerx += dx
                rectangle.centery += dy
        print(self.get_reward_per_state())
        return action_taken

    def evaluate_reward(self, margin=5):
        self.reward = 0
        for i in range(self.num_blocks):
            block_pos = np.where(self.goal_config == i)
            desired_neighbors = defaultdict(int, V=-1)
            if block_pos[0][0] - 1 >= 0:
                desired_neighbors['left'] = self.goal_config[block_pos[0][0] - 1][block_pos[1][0]]
                # if pygame.sprite.collide_rect(self.block_dict[i], self.block_dict[desired_neighbors['left']]:
                if desired_neighbors['left'] >= 0 and self.block_dict[i].rect.centerx > self.block_dict[
                    desired_neighbors['left']].rect.centerx and \
                    self.block_dict[
                        desired_neighbors['left']].rect.centery - margin < self.block_dict[i].rect.centery < \
                    self.block_dict[desired_neighbors['left']].rect.centery + margin:
                    self.reward += 1

            if block_pos[0][0] + 1 < self.num_stacks:
                desired_neighbors['right'] = self.goal_config[block_pos[0][0] + 1][block_pos[1][0]]
                if desired_neighbors['right'] >= 0 and self.block_dict[i].rect.centerx < self.block_dict[
                    desired_neighbors['right']].rect.centerx and \
                    self.block_dict[
                        desired_neighbors['right']].rect.centery - margin < self.block_dict[i].rect.centery < \
                    self.block_dict[desired_neighbors['right']].rect.centery + margin:
                    self.reward += 1

            if block_pos[1][0] - 1 >= 0:
                desired_neighbors['bottom'] = self.goal_config[block_pos[0][0]][block_pos[1][0] - 1]
                if desired_neighbors['bottom'] >= 0 and \
                    self.block_dict[i].rect.centery - self.block_dict[
                    desired_neighbors['bottom']].rect.centery > -55 and \
                    self.block_dict[
                        desired_neighbors['bottom']].rect.centerx - margin < self.block_dict[i].rect.centerx < \
                    self.block_dict[
                        desired_neighbors['bottom']].rect.centerx + margin:
                    self.reward += 1

            if block_pos[1][0] + 1 < self.num_blocks:
                desired_neighbors['top'] = self.goal_config[block_pos[0][0]][block_pos[1][0] + 1]
                if desired_neighbors['top'] >= 0 and \
                    self.block_dict[i].rect.centery - self.block_dict[desired_neighbors['top']].rect.centery < 55 and \
                    self.block_dict[
                        desired_neighbors['top']].rect.centerx - margin < self.block_dict[i].rect.centerx < \
                        self.block_dict[desired_neighbors['top']].rect.centerx + margin:
                    self.reward += 1

    def run_environment(self, record=True):
        actions_taken = []
        running = True
        frame_num = 0

        # Required for DQN to map frames to actions.

        most_recent_action = None
        # Create the surface and pass in a tuple with its length and width
        drag = False
        # to indicate the end of the demonstration
        done = 0

        rectangle = None
        sel_block_id = None
        prev_action = None

        # choosing the order for blocks to be placed in the goal screen.
        block_order = [i for i in range(self.num_blocks)]
        seed = np.random.randint(0, self.num_stacks)
        random.shuffle(block_order)
        self.create_goal(self.num_stacks, block_order, seed)

        while running:
            most_recent_action = None
            if self.reward <= 0:
                self.reward = -0.1
            elif self.reward > 0:
                self.reward += -0.01
            # time.sleep(1. /30)
            self.screen.blit(self.master_surface, (0, 0))

            # rendering the goal screen
            self.screen.blit(self.goal_surface, (800, 0))

            if record:

                # for loop through the event queue
                for event in pygame.event.get():
                    # Our main loop!
                    # Check for KEYDOWN event; KEYDOWN is a constant defined in pygame.locals, which we imported earlier
                    if event.type == KEYUP:
                        prev_action = None
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
                            drag = False
                            rectangle = None
                            self.block_dict[sel_block_id].surf.fill(COLORS[sel_block_id])
                            actions_taken.append((Action.DROP, sel_block_id))
                            self.evaluate_reward()
                            most_recent_action = action_to_ind[Action.DROP.value]
                            print(self.reward)

                        elif event.key in {K_UP, K_DOWN, K_LEFT, K_RIGHT}:
                            prev_action = event.key

                    # Check for QUIT event; if QUIT, set running to false
                    elif event.type == QUIT or (hasattr(event, "key") and event.key == K_ESCAPE):
                        print("writing to file")
                        prev_action = None
                        actions_taken.append((Action.FINISHED, None))
                        most_recent_action = Action.FINISHED
                        with open("state_action_map.json", 'w') as f:
                            json.dump(self.state_action_map, f, indent=5)
                        print("Finished Demonstration:", actions_taken)
                        running = False

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        prev_action = None
                        pos = pygame.mouse.get_pos()
                        for block in self.blocks:
                            if block.rect.collidepoint(pos):
                                drag = True
                                rectangle = block.rect
                                sel_block_id = block.id
                                print("Frame: %d" % frame_num)
                                actions_taken.append((Action.PICK, block.id))
                                most_recent_action = action_to_ind[Action.PICK.value + str(block.id)]
                                self.block_dict[block.id].surf.fill(WHITE)
                                break

                if prev_action:
                    action_taken = self.move_block(prev_action, drag, rectangle, sel_block_id)
                    if action_taken:
                        actions_taken.append(action_taken)
                        most_recent_action = action_to_ind[action_taken[0].value.split("_")[1]]

                # for block in self.blocks:
                #    self.screen.blit(block.surf, block.rect)

                # pygame.display.flip()
                filename = self.demo_dir + "/%d.png" % frame_num
                frame_num += 1
                if most_recent_action is not None:
                    self.state_action_map[self.demo_id][frame_num] = [most_recent_action, self.reward, done]
                for block in self.blocks:
                    self.screen.blit(block.surf, block.rect)
                pygame.display.flip()
                pygame.image.save(self.screen, filename)

            else:
                dqn = DQN(12)

                dqn.load_state_dict(torch.load("dqn.pth"))
                dqn = dqn.cuda()
                img_string = pygame.image.tostring(self.screen, "RGB", False)
                # Image.frombytes("RGB", (1000, 950), img_string).show()
                curr_frame = np.asarray(Image.frombytes("RGB", (1000, 950), img_string))
                curr_frame = cv2.resize(curr_frame, (100, 95))
                # print(curr_frame)

                action = torch.argmax(dqn(torch.Tensor(curr_frame).view(1, 3, 100, 95))).item()
                print(ind_to_action[action])
                # while ind_to_action[action]!=Action.FINISHED.value:
                #    time.sleep(1/10)
                action = ind_to_action[action]
                print(action)
                for i in range(0, self.num_blocks):
                    if action == "PICK" + str(i):
                        sel_block_id = i
                        self.block_dict[sel_block_id].surf.fill(WHITE)
                if action == Action.DROP.value:
                    self.block_dict[sel_block_id].surf.fill(COLORS[sel_block_id])
                    sel_block_id = None
                elif action == Action.MOVE_DOWN.value:
                    self.block_dict[sel_block_id].centerY += 10
                elif action == Action.MOVE_UP.value:
                    self.block_dict[sel_block_id].centerY -= 10
                elif action == Action.MOVE_LEFT.value:
                    self.block_dict[sel_block_id].centerX -= 10
                elif action == Action.MOVE_RIGHT.value:
                    self.block_dict[sel_block_id].centerX += 10
                elif action == Action.FINISHED.value:
                    print("GAME OVER")
                    return
                img_string = pygame.image.tostring(self.screen, "RGB", False)
                curr_frame = np.asarray(Image.frombytes("RGB", (1000, 950), img_string))
                curr_frame = cv2.resize(curr_frame, (100, 95))
                action = torch.argmax(dqn(torch.Tensor(curr_frame).view(1, 3, 100, 95))).item()
                for block in self.blocks:
                    self.screen.blit(block.surf, block.rect)
                pygame.display.flip()
                if ind_to_action[action] == Action.FINISHED.value:
                    return
