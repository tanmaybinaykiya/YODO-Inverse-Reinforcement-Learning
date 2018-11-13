import pygame.camera
import random
import pygame
from pygame.locals import *
import numpy as np
import time

RED=(255,0,0)
BLUE=(0,0,255)
GREEN=(0,255,0)
YELLOW=(0,255,255)
CYAN=(255,255,0)
WHITE=(255,255,255)
COLORS=[RED,BLUE,GREEN,YELLOW,CYAN,WHITE]

actions_taken={"MOVE":[],"MOVE_RIGHT":[],"MOVE_LEFT":[],"MOVE_UP":[],"MOVE_DOWN":[],"PICK":[], "DROP":[], "DRAG":[], "FINISHED":[]}
class Blocks(pygame.sprite.Sprite):

    def __init__(self,color_index,grid_centers,block_size=50):
        super(Blocks, self).__init__()
        self.surf = pygame.Surface((block_size,block_size))
        self.surf.fill(COLORS[color_index])
        self.id=color_index
        self.rect = self.surf.get_rect(center=grid_centers[random.randint(0,len(grid_centers))])


class BlockWorld():

    def __init__(self,screen_width,screen_height,num_blocks=2):
        self.timer_start=time.time()
        self.screen_width=screen_width
        self.screen_height=screen_height
        self.screen=pygame.display.set_mode((screen_width, screen_height))
        self.master_surface = pygame.Surface(self.screen.get_size())
        self.master_surface.fill((0,0,0))
        self.grid_centers=[]
        for y in range(0,screen_height-51,51):
            pygame.draw.line(self.master_surface,(255,0,255),(0,y),(screen_width,y),1)
        for x in range(0,screen_width-51,51):
            pygame.draw.line(self.master_surface,(255,0,255),(x,0),(x,screen_height),1)
        for i in range(25,screen_height,51):
            for j in range(25,screen_width,51):
                self.grid_centers.append((i+0.5,j+0.5))
        self.blocks = pygame.sprite.Group()
        self.create_blocks(num_blocks)
        pygame.init()

        #TODO: Camera Code
        #pygame.camera.init()

        self.run_environment()
        pass

    def distance(self,pts1,pts2):
        return np.sqrt(np.square(pts1[0]-pts2[0])+np.square(pts1[1]-pts2[1]))

    def create_blocks(self, num_blocks=2, block_size=50):
        for i in range(num_blocks):
            block=Blocks(i,self.grid_centers)
            self.blocks.add(block)

        # Give the surface a color to differentiate it from the background

        pygame.display.flip()

    def move_block(self,action,drag,rectangle,sel_block_id):
        if action == K_UP:
            if drag:
                other_blocks = [b for b in self.blocks if b.rect != rectangle]
                collision_indices = rectangle.collidelistall(other_blocks)
                if len(collision_indices) != 0:
                    for ind in collision_indices:
                        if self.distance((other_blocks[ind].rect.centerx, other_blocks[ind].rect.centery),
                                         (rectangle.centerx, rectangle.centery)) < \
                                self.distance(
                                    (other_blocks[ind].rect.centerx, other_blocks[ind].rect.centery - 5),
                                    (rectangle.centerx, rectangle.centery)):
                            break
                    else:
                        actions_taken['MOVE_UP'].append(
                            (time.time() - self.timer_start, sel_block_id))
                        rectangle.centery -= 5
                elif rectangle:
                    actions_taken['MOVE_UP'].append((time.time() - self.timer_start, sel_block_id))
                    rectangle.centery -= 5

        elif action == K_DOWN:
            if drag:
                other_blocks = [b for b in self.blocks if b.rect != rectangle]
                collision_indices = rectangle.collidelistall(other_blocks)
                if len(collision_indices) != 0:
                    for ind in collision_indices:
                        if self.distance((other_blocks[ind].rect.centerx, other_blocks[ind].rect.centery),
                                         (rectangle.centerx, rectangle.centery)) < \
                                self.distance(
                                    (other_blocks[ind].rect.centerx, other_blocks[ind].rect.centery + 5),
                                    (rectangle.centerx, rectangle.centery)):
                            break
                    else:
                        actions_taken['MOVE_DOWN'].append(
                            (time.time() - self.timer_start, sel_block_id))
                        rectangle.centery += 5
                elif rectangle:
                    actions_taken['MOVE_DOWN'].append((time.time() - self.timer_start, sel_block_id))
                    rectangle.centery += 5

        elif action == K_LEFT:
            if drag:
                other_blocks = [b for b in self.blocks if b.rect != rectangle]
                collision_indices = rectangle.collidelistall(other_blocks)
                if len(collision_indices) != 0:
                    for ind in collision_indices:
                        if self.distance((other_blocks[ind].rect.centerx, other_blocks[ind].rect.centery),
                                         (rectangle.centerx, rectangle.centery)) < \
                                self.distance(
                                    (other_blocks[ind].rect.centerx - 5, other_blocks[ind].rect.centery),
                                    (rectangle.centerx, rectangle.centery)):
                            break
                    else:
                        actions_taken['MOVE_LEFT'].append(
                            (time.time() - self.timer_start, sel_block_id))
                        rectangle.centerx -= 5
                elif rectangle:
                    actions_taken['MOVE_LEFT'].append((time.time() - self.timer_start, sel_block_id))
                    rectangle.centerx -= 5

        elif action == K_RIGHT:
            if drag:
                other_blocks = [b for b in self.blocks if b.rect != rectangle]
                collision_indices = rectangle.collidelistall(other_blocks)
                if len(collision_indices) != 0:
                    for ind in collision_indices:
                        if self.distance((other_blocks[ind].rect.centerx, other_blocks[ind].rect.centery),
                                         (rectangle.centerx, rectangle.centery)) < \
                                self.distance(
                                    (other_blocks[ind].rect.centerx + 5, other_blocks[ind].rect.centery),
                                    (rectangle.centerx, rectangle.centery)):
                            break
                    else:
                        actions_taken['MOVE_RIGHT'].append(
                            (time.time() - self.timer_start, sel_block_id))
                        rectangle.centerx += 5
                elif rectangle:
                    actions_taken['MOVE_RIGHT'].append((time.time() - self.timer_start, sel_block_id))
                    rectangle.centerx += 5

    def run_environment(self):
        running = True
        # Create the surface and pass in a tuple with its length and width
        drag = False
        rectangle = None
        sel_block_id=None
        prev_action=None

        '''
        #TODO: Camera Code
        cam = pygame.camera.Camera("/dev/video0", (640, 480))
        cam.start()
        '''

        while running:
            time.sleep(1. / 25)
            self.screen.blit(self.master_surface,(0,0))
            # for loop through the event queue
            for event in pygame.event.get():
                # Our main loop!
                # Check for KEYDOWN event; KEYDOWN is a constant defined in pygame.locals, which we imported earlier
                if event.type ==KEYUP:
                    prev_action=None
                if event.type == KEYDOWN:
                    # If the Esc key has been pressed set running to false to exit the main loop
                    if event.key == K_ESCAPE:
                        prev_action=None
                        running = False

                    elif event.key == K_RSHIFT:
                        prev_action=None
                        actions_taken['FINISHED'].append(time.time()-self.timer_start)
                        print("Finished Demonstration")
                        print(actions_taken)

                    elif event.key == K_SPACE:
                        print("Dropped")
                        drag = False
                        rectangle = None
                        actions_taken['DROP'].append((time.time() - self.timer_start, sel_block_id))
                        pass

                    elif event.key==K_UP:
                        prev_action=K_UP

                    elif event.key==K_DOWN:
                        prev_action=K_DOWN

                    elif event.key==K_LEFT:
                        prev_action=K_LEFT

                    elif event.key==K_RIGHT:
                        prev_action=K_RIGHT

                # Check for QUIT event; if QUIT, set running to false
                elif event.type == QUIT:
                    prev_action=None
                    actions_taken['FINISHED'].append(time.time() - self.timer_start)
                    print("Finished Demonstration")
                    print(actions_taken)
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    prev_action=None
                    pos=pygame.mouse.get_pos()
                    for block in self.blocks:
                        if block.rect.collidepoint(pos):
                            print('Picked up')
                            drag=True
                            rectangle=block.rect
                            sel_block_id=block.id
                            actions_taken['PICK'].append((time.time() - self.timer_start, event.pos,block.id))
                            break

                '''
                elif event.type == pygame.MOUSEMOTION:
                    mouse_x, mouse_y = event.pos
                    if drag:
                        other_blocks=[b for b in self.blocks if b.rect != rectangle]
                        collision_indices=rectangle.collidelistall(other_blocks)
                        if len(collision_indices)!=0:
                            for ind in collision_indices:
                                if self.distance((other_blocks[ind].rect.centerx,other_blocks[ind].rect.centery),event.pos)<\
                                        self.distance((other_blocks[ind].rect.centerx,other_blocks[ind].rect.centery),(rectangle.centerx,rectangle.centery)):
                                        break
                            else:
                                actions_taken['DRAG'].append((time.time() - self.timer_start, event.pos,sel_block_id))
                                rectangle.centerx = mouse_x
                                rectangle.centery = mouse_y
                        elif rectangle:
                            actions_taken['DRAG'].append((time.time() - self.timer_start, event.pos,sel_block_id))
                            rectangle.centerx = mouse_x
                            rectangle.centery = mouse_y
                    else:
                        actions_taken['MOVE'].append((time.time()-self.timer_start,event.pos,sel_block_id))
                '''

            if prev_action:
                self.move_block(prev_action,drag,rectangle,sel_block_id)
            for block in self.blocks:
                self.screen.blit(block.surf, block.rect)

            pygame.display.flip()

            '''
            #TODO: Camera Code
            image = cam.get_image()
            filename = "Snaps/%04d.png" % file_num
            pygame.image.save(image, filename)
            '''

b= BlockWorld(1024,980,5)

