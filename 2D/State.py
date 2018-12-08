import numpy as np

from constants import Action, move_action_to_deviation as action_to_deviation_map
from utilities import euclidean_dist, manhattan_distance, sgn


class State:

    def __init__(self, block_positions, selected_index, goal_config, screen_dims, block_size=50):
        """
        :type block_positions: list[tuple(int)]
        :type goal_positions: list[tuple(int)]
        :type selected_index: nullable int
        :type block_size: int
        """
        self.block_positions = block_positions
        self.block_count = len(block_positions)
        self.selected_index = selected_index
        self.goal_config = goal_config
        self.block_size = block_size
        self.screen_dims = screen_dims

        self.goal_positions = self.compute_goal_positions()

    def compute_goal_positions(self):
        block_count = self.block_count
        median_x = sum(self.get_position(idx)[0] for idx in range(self.block_count)) // self.block_count
        median_x = self.block_size//2 + median_x - median_x % self.block_size
        median_y = sum(self.get_position(idx)[1] for idx in range(self.block_count)) // self.block_count
        median_y = self.block_size//2 + median_y - median_y % self.block_size
        goal_position = [None for _ in range(block_count)]

        if block_count % 2 == 1:
            for idx, i in enumerate(self.goal_config[0]):
                goal_position[i] = (median_x, median_y + self.block_size * (block_count // 2 - idx))
        else:
            for idx, i in enumerate(self.goal_config[0]):
                goal_position[i] = (median_x, median_y + self.block_size * (block_count // 2 - idx))

        for _ in range(self.block_count):
            if not State.is_in_bounding_box(goal_position[self.goal_config[0][-1]], block_size=self.block_size, screen_dims=self.screen_dims):
                # move 50 down i.e. +50
                for idx in range(block_count):
                    goal_position[idx] = (goal_position[idx][0], goal_position[idx][1] + self.block_size)
            elif not State.is_in_bounding_box(goal_position[self.goal_config[0][0]], block_size=self.block_size, screen_dims=self.screen_dims):
                # move 50 up, i.e. -50
                for idx in range(block_count):
                    goal_position[idx] = (goal_position[idx][0], goal_position[idx][1] - self.block_size)

        return goal_position

    def get_position(self, block_index):
        return self.block_positions[block_index]

    def get_selected(self):
        return self.selected_index

    def get_goal_position(self, block_index) -> list:
        return self.goal_positions[block_index]

    def set_goal_positions(self, goal_positions):
        self.goal_positions = goal_positions

    def get_tuple(self):
        return tuple(self.block_positions), tuple(self.goal_positions), self.selected_index

    def update_selection(self, selection):
        self.selected_index = selection

    def update_state(self, idx, position):
        self.block_positions[idx] = position

    def select(self, idx):
        self.selected_index = idx

    def deselect(self):
        self.selected_index = None

    def copy(self):
        return self.__deepcopy__()

    def __deepcopy__(self):
        return State(block_positions=self.block_positions.copy(), goal_config=self.goal_config.copy(), selected_index=self.selected_index, screen_dims=tuple(self.screen_dims))

    def __repr__(self):
        return "Positions: %s, Goal: %s, Selected: %s" % (self.block_positions, self.goal_positions, self.selected_index)

    def goal_reached(self):
        for i in range(self.block_count - 1):
            this_block = self.get_position(self.goal_config[0][i])
            next_block = self.get_position(self.goal_config[0][i + 1])
            val = this_block[0] == next_block[0] and this_block[1] - next_block[1] == self.block_size
            if not val:
                return False
        return True

    def is_action_good(self, move_action, idx):
        def get_next_state(action):
            new_state: State = self.copy()
            old_position: tuple = self.block_positions[idx]
            new_state.block_positions[idx] = (old_position[0] + action_to_deviation_map[action][0], old_position[1] + action_to_deviation_map[action][1])
            return new_state

        new_block_position = get_next_state(move_action).block_positions[idx]
        in_bounding_box = State.is_in_bounding_box(new_block_position, self.block_size, screen_dims=self.screen_dims)
        is_not_colliding = not any([tuple(new_block_position) == tuple(block_position) for block_position in self.block_positions])
        return in_bounding_box and is_not_colliding

    def is_action_allowed(self, move_action, idx):
        return self.is_action_good(move_action, idx) and not self.is_action_blocking_goal(move_action, idx)

    def is_action_blocking_goal(self, move_action, idx):
        def get_next_state(action):
            new_state: State = self.copy()
            old_position: tuple = self.block_positions[idx]
            new_state.block_positions[idx] = (old_position[0] + action_to_deviation_map[action][0], old_position[1] + action_to_deviation_map[action][1])
            return new_state

        new_block_position = get_next_state(move_action).block_positions[idx]
        return self.is_state_blocking_goal(new_block_position, idx)

    def is_state_blocking_goal(self, new_block_position, idx):
        am_blocking_goal = any([tuple(goal_position) == tuple(new_block_position) for goal_position in self.goal_positions])
        am_blocking_my_goal = tuple(new_block_position) == tuple(self.goal_positions[idx])
        return am_blocking_goal and not am_blocking_my_goal

    def all_goals_blocked(self):
        for goalIdx, goalPos in enumerate(self.goal_positions):
            this_goal_blocked = False
            for blockidx, block_position in enumerate(self.block_positions):
                this_goal_blocked = this_goal_blocked or tuple(block_position) == tuple(goalPos)
            if not this_goal_blocked:
                return False
        return True

    def get_target_blocks(self):
        target_blocks = {}
        for i in range(len(self.goal_config[0]) - 1):
            target_blocks[i] = self.goal_config[0][i + 1]
        target_blocks[len(self.goal_config[0]) - 1] = self.goal_config[0][-2]
        return target_blocks

    def get_medial_state_repr(self):
        if self.selected_index is not None:
            pos = self.get_position(self.selected_index)
            transformed_pos = (pos[0] - 25) // 50, (pos[1] - 25) // 50
            goal = self.get_goal_position(self.selected_index)
            transformed_goal = (goal[0] - 25) // 50, (goal[1] - 25) // 50
            return sgn(transformed_pos[0] - transformed_goal[0]), sgn(transformed_pos[1] - transformed_goal[1]), manhattan_distance(transformed_pos, transformed_goal)
        else:
            transformed_x = [(pos[0] - 25) // 50 for pos in self.block_positions]
            transformed_y = [(pos[1] - 25) // 50 for pos in self.block_positions]

            goal_x = [(pos[0] - 25) // 50 for pos in self.goal_positions]
            goal_y = [(pos[1] - 25) // 50 for pos in self.goal_positions]

            transformed_pos = [(sgn(ix - gx), sgn(iy - gy)) for ix, iy, gx, gy in zip(transformed_x, transformed_y, goal_x, goal_y)]

            return tuple(transformed_pos)

    def get_medial_state_repr_old(self):
        transformed_x = [(pos[0] - 25) // 50 for pos in self.block_positions]
        transformed_y = [(pos[1] - 25) // 50 for pos in self.block_positions]

        goal_x = [(pos[0] - 25) // 50 for pos in self.goal_positions]
        goal_y = [(pos[1] - 25) // 50 for pos in self.goal_positions]

        transformed_pos = [(sgn(ix - gx), sgn(iy - gy)) for ix, iy, gx, gy in zip(transformed_x, transformed_y, goal_x, goal_y)]

        return tuple(transformed_pos), tuple(self.goal_config[0]), self.selected_index

    def get_medial_state_repr_older(self):
        transformed_x = [(pos[0] - 25) // 50 for pos in self.block_positions]
        transformed_y = [(pos[1] - 25) // 50 for pos in self.block_positions]

        median_x = np.array(np.median(transformed_x), dtype=int)
        median_y = np.array(np.median(transformed_y), dtype=int)

        def sgn(a):
            if a < 0:
                return -1
            elif a == 0:
                return 0
            else:
                return 1

        transformed_pos = [(pos[0] - median_x, pos[1] - median_y) for pos in zip(transformed_x, transformed_y)]

        return tuple(transformed_pos), tuple(self.goal_config[0]), self.selected_index

    def get_state_as_tuple_pramodith(self):
        target_blocks = self.get_target_blocks()
        some_list = [-1 for _ in range(3)]
        directions = ["-", "-"]
        if self.selected_index is not None:
            if self.selected_index in target_blocks:
                target_id = target_blocks[self.selected_index]
                some_list[0] = np.square(self.block_positions[self.selected_index][0] - self.block_positions[target_id][0]) + np.square(self.block_positions[self.selected_index][1] - self.block_positions[target_id][1])
                if self.block_positions[self.selected_index][0] - self.block_positions[target_id][0] > 0:
                    directions[0] = 'l'
                elif self.block_positions[self.selected_index][0] - self.block_positions[target_id][0] < 0:
                    directions[0] = 'r'

                if self.block_positions[self.selected_index][1] - self.block_positions[target_id][1] > 0:
                    directions[1] = 'u'
                elif self.block_positions[self.selected_index][1] - self.block_positions[target_id][1] < 0:
                    directions[1] = 'd'
            else:
                for key, value in target_blocks.items():
                    if value == self.selected_index:
                        target_id = key
                        some_list[0] = np.square(self.block_positions[self.selected_index][0] - self.block_positions[target_id][0]) + np.square(self.block_positions[self.selected_index].rect.centery - self.block_positions[target_id].rect.centery)
                        if self.block_positions[self.selected_index][0] - self.block_positions[target_id][0] > 0:
                            directions[0] = 'l'
                        elif self.block_positions[self.selected_index][0] - self.block_positions[target_id][0] < 0:
                            directions[0] = 'r'

                        if self.block_positions[self.selected_index].rect.centery - self.block_positions[target_id].rect.centery > 0:
                            directions[1] = 'u'
                        elif self.block_positions[self.selected_index].rect.centery - self.block_positions[target_id].rect.centery < 0:
                            directions[1] = 'd'
        else:
            distances = []
            for key in target_blocks:
                distances.append(euclidean_dist(self.block_positions[key], self.block_positions[target_blocks[key]]))

            some_list[0] = tuple(distances)

        some_list[1] = tuple(directions)
        some_list[-1] = self.selected_index
        some_list.append(tuple([tuple(x) for x in self.goal_config]))
        return tuple(some_list)

    def get_state_as_tuple(self):
        # curr_state is a n-tuple( (x1, y1), (x2, y2), (x3, y3), (x4, y4), selectedBlockId, (goal_config))
        some_list = [0 for _ in range(self.block_count + 1)]
        for block_id in self.block_positions:
            some_list[block_id] = (self.block_positions[block_id][0], self.block_positions[block_id][1])
        some_list[-1] = self.selected_index
        some_list.append(tuple([tuple(x) for x in self.goal_config]))
        return tuple(some_list)

    def get_state_as_dict(self):
        block_pos = self.block_positions
        state = {"positions": {block_id: (block_pos[block_id][0], block_pos[block_id][1]) for block_id in block_pos}, "selected": self.selected_index if self.selected_index is not None else -1}
        return state

    def get_next_state(self, action: tuple, screen_dims):
        # action: [Action, int]
        new_state = self.copy()
        if action[0] == Action.PICK:
            new_state.selected_index = action[1]
        elif action[0] == Action.DROP:
            new_state.selected_index = None
        else:
            new_state.block_positions[self.selected_index] = self.get_transformed_location(action[0], self.selected_index, screen_dims)
        return new_state

    def get_rect(self, center):
        return {"left": center[0] - self.block_size // 2, "right": center[0] + self.block_size // 2, "bottom": center[1] + self.block_size // 2, "top": center[1] - self.block_size // 2}

    @staticmethod
    def are_intersecting(rect1, dx, dy, other_rect):
        return (other_rect["top"] <= rect1["top"] + dy < other_rect["bottom"] and (other_rect["left"] <= rect1["left"] + dx < other_rect["right"] or other_rect["left"] < rect1["right"] + dx <= other_rect["right"])) or (other_rect["top"] < rect1["bottom"] + dy <= other_rect["bottom"] and (other_rect["left"] <= rect1["left"] + dx < other_rect["right"] or other_rect["left"] < rect1["right"] + dx <= other_rect["right"]))

    @staticmethod
    def is_in_bounding_box(next_pos, block_size, screen_dims):
        screen_width, screen_height = screen_dims
        return (block_size / 2) <= next_pos[0] <= (screen_width - block_size / 2) and (block_size / 2) <= next_pos[1] <= (screen_height - block_size / 2)

    def get_transformed_location(self, action, sel_block_id, screen_dims):
        if action in action_to_deviation_map:
            dx, dy = action_to_deviation_map[action]
        else:
            raise IOError("Invalid Action", action)
        rectangle = self.get_position(sel_block_id)
        not_intersections = [not State.are_intersecting(self.get_rect(rectangle), dx, dy, self.get_rect(other_block)) for id, other_block in enumerate(self.block_positions) if sel_block_id != id]
        orig_pos = rectangle
        if all(not_intersections):
            next_pos = (orig_pos[0] + dx, orig_pos[1] + dy)
            if self.is_in_bounding_box(next_pos, self.block_size, screen_dims):
                return next_pos
        return orig_pos


def test_get_goal_position():
    state = State(block_positions=[[75, 25], [125, 25], [175, 25], [225, 25], [275, 25]], selected_index=None, goal_config=[[3, 2, 0, 1, 4]], screen_dims=(350, 350))
    state.compute_goal_positions()
    assert [(175, 125), (175, 75), (175, 175), (175, 225), (175, 25)] == state.goal_positions

    state = State(block_positions=[[75, 325], [125, 325], [175, 325], [225, 325], [275, 325]], selected_index=None, goal_config=[[3, 2, 0, 1, 4]], screen_dims=(350, 350))
    state.compute_goal_positions()
    assert  [(175, 225), (175, 175), (175, 275), (175, 325), (175, 125)] == state.goal_positions


def test_get_medial_position_rep():
    medial_state_rep = State(block_positions=[[10, 20], [20, 10], [30, 30]], selected_index=1, goal_config=[[0, 2, 1]]).get_medial_state_repr()
    print("Medial: ", medial_state_rep)


if __name__ == '__main__':
    test_get_goal_position()