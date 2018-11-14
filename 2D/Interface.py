import time

import numpy as np
import pygame
from pygame.locals import *

RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
COLORS = [RED, BLUE, GREEN, YELLOW, CYAN, WHITE]

actions_taken = []


class Blocks(pygame.sprite.Sprite):

    def __init__(self, color_index, grid_center, block_size=50):
        super(Blocks, self).__init__()
        self.surf = pygame.Surface((block_size, block_size))
        self.surf.fill(COLORS[color_index])
        self.id = color_index
        self.rect = self.surf.get_rect(center=grid_center)


class BlockWorld:

    def __init__(self, screen_width, screen_height, num_blocks=2, record=True):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.master_surface = pygame.Surface(self.screen.get_size())
        self.master_surface.fill((0, 0, 0))
        self.grid_centers = [(i, 500) for i in range(25, screen_width, 50)]
        self.blocks = pygame.sprite.Group()
        self.create_blocks(num_blocks)
        self.record = record
        pygame.init()

        self.run_environment()
        pass

    @staticmethod
    def distance(pts1, pts2):
        return (pts1[0] - pts2[0]) ** 2 + (pts1[1] - pts2[1]) ** 2

    @staticmethod
    def are_intersecting(rect1, dx, dy, other_rect):
        return (other_rect.top <= rect1.top+dy <= other_rect.bottom
                and (other_rect.left <= rect1.left+dx <= other_rect.right
                     or other_rect.left <= rect1.right+dx <= other_rect.right)) \
               or (other_rect.top <= rect1.bottom+dy <= other_rect.bottom
                   and (other_rect.left <= rect1.left+dx <= other_rect.right
                        or other_rect.left <= rect1.right+dx <= other_rect.right))

    def create_blocks(self, num_blocks=2, block_size=50):
        for i, blockCenterIdx in enumerate(np.random.choice(len(self.grid_centers), num_blocks, replace=False)):
            self.blocks.add(Blocks(i, self.grid_centers[blockCenterIdx]))
        pygame.display.flip()

    def move_block(self, action, drag, rectangle, sel_block_id):
        if action == K_UP:
            this_action = 'MOVE_UP'
            dx, dy = 0, -5
        elif action == K_DOWN:
            this_action = 'MOVE_DOWN'
            dx, dy = 0, 5
        elif action == K_LEFT:
            this_action = 'MOVE_LEFT'
            dx, dy = -5, 0
        elif action == K_RIGHT:
            this_action = 'MOVE_RIGHT'
            dx, dy = 5, 0
        else:
            raise IOError("Invalid Action", action)

        if drag:
            all_dists = [not BlockWorld.are_intersecting(rectangle, dx, dy, other_block.rect) for other_block in self.blocks if other_block.rect != rectangle]
            if all(all_dists):
                actions_taken.append((this_action, sel_block_id))
                rectangle.centerx += dx
                rectangle.centery += dy

    def run_environment(self):
        running = True
        # Create the surface and pass in a tuple with its length and width
        drag = False
        rectangle = None
        sel_block_id = None
        prev_action = None

        while running:
            time.sleep(1. / 25)
            self.screen.blit(self.master_surface, (0, 0))
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
                        actions_taken.append(('FINISHED', None))
                        print("Finished Demonstration")
                        print(actions_taken)
                    elif event.key == K_SPACE:
                        print("Dropped")
                        drag = False
                        rectangle = None
                        actions_taken.append(('DROP', sel_block_id))
                    elif event.key in set([K_UP, K_DOWN, K_LEFT, K_RIGHT]):
                        prev_action = event.key
                # Check for QUIT event; if QUIT, set running to false
                elif event.type == QUIT:
                    prev_action = None
                    actions_taken.append(('FINISHED', None))
                    print("Finished Demonstration")
                    print(actions_taken)
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    prev_action = None
                    pos = pygame.mouse.get_pos()
                    for block in self.blocks:
                        if block.rect.collidepoint(pos):
                            print('Picked up')
                            drag = True
                            rectangle = block.rect
                            sel_block_id = block.id
                            actions_taken.append(('PICK', block.id))
                            break

            if prev_action:
                self.move_block(prev_action, drag, rectangle, sel_block_id)
            for block in self.blocks:
                self.screen.blit(block.surf, block.rect)

            pygame.display.flip()

            if self.record:
                filename = "./screen_capture/%d.png" % time.time()
                pygame.image.save(self.screen, filename)


b = BlockWorld(1000, 950, 5, record=False)
