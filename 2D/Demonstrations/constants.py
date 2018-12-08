from enum import Enum

from pygame.locals import K_UP, K_DOWN, K_LEFT, K_RIGHT

Color = {
    "RED": (255, 0, 0),
    "BLUE": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "YELLOW": (255, 255, 0),
    "CYAN": (0, 255, 255),
    "WHITE": (255, 255, 255),
    "GRAY": (50, 50, 50)
}
RED = Color["RED"]
BLUE = Color["BLUE"]
GREEN = Color["GREEN"]
YELLOW = Color["YELLOW"]
CYAN = Color["CYAN"]
WHITE = Color["WHITE"]
GRAY = Color["GRAY"]

CLICKED_COLOR={0:(253,163,163),1:(163,163,253),2:(163,253,163),3:(247,253,163)}
COLORS = [Color["RED"], Color["BLUE"], Color["GREEN"], Color["YELLOW"], Color["CYAN"], Color["WHITE"]]
COLORS_STR = ["RED", "BLUE", "GREEN", "YELLOW", "CYAN", "WHITE"]
FRAME_LOCATION = "Demonstrations"


# might require just for DQN

class Action(Enum):
    MOVE_UP = 'MOVE_UP'
    MOVE_DOWN = 'MOVE_DOWN'
    MOVE_LEFT = 'MOVE_LEFT'
    MOVE_RIGHT = 'MOVE_RIGHT'
    FINISHED = 'FINISHED'
    DROP = 'DROP'
    PICK = 'PICK'


ind_to_action = {0: "MOVE", 1: "RIGHT", 2: "LEFT", 3: "UP", 4: "DOWN", 5: "DROP", 6: "PICK0", 7: "PICK1", 8: "PICK2",
                 9: "PICK3", 10: "PICK4", 11: "FINISHED"}

action_to_ind = {"MOVE": 0, "RIGHT": 1, "LEFT": 2, "UP": 3, "DOWN": 4, "DROP": 5, "PICK0": 6, "PICK1": 7, "PICK2": 8,
                 "PICK3": 9, "PICK4": 10, "FINISHED": 11}

key_to_action = {
    K_UP: Action.MOVE_UP,
    K_DOWN: Action.MOVE_DOWN,
    K_LEFT: Action.MOVE_LEFT,
    K_RIGHT: Action.MOVE_RIGHT
}

step_size = 50
move_action_to_deviation = {
    Action.MOVE_UP: (0, -step_size),
    Action.MOVE_DOWN: (0, step_size),
    Action.MOVE_LEFT: (-step_size, 0),
    Action.MOVE_RIGHT: (step_size, 0),
    Action.DROP: (0, 0),
    Action.FINISHED: (0, 0),
    Action.PICK: (0, 0)
}
