from constants import Action, move_action_to_deviation


class Solver:

    def __init__(self, window_width, window_height, step_size, goal_config, init_state):
        self.window_width = window_width
        self.window_height = window_height
        self.step_size = step_size
        self.goal_config = goal_config
        self.init_state = init_state

    def get_available_actions(self, state, block_idx) -> []:
        return [action for action in move_action_to_deviation
                if 0 <= state[block_idx][0] + move_action_to_deviation[action][0] < self.window_width and
                0 <= state[block_idx][1] + move_action_to_deviation[action][1] < self.window_height]

    def get_next_state(self, state, idx, action) -> dict:
        new_state = state.copy()
        new_state[idx] = (
            state[idx][0] + move_action_to_deviation[action][0], state[idx][1] + move_action_to_deviation[action][1])
        return new_state

    @staticmethod
    def get_all_pairs_max_dist(state) -> int:
        return max([sum([Solver.dist(state, idx, jdx) for jdx in state]) for idx in state])

    @staticmethod
    def get_all_pairs_min_dist(state) -> int:
        return min([sum([Solver.dist(state, idx, jdx) for jdx in state]) for idx in state])

    @staticmethod
    def dist(state, idx, jdx):
        return abs(state[idx][0] - state[jdx][0]) + abs(state[idx][1] - state[jdx][1])

    @staticmethod
    def get_farthest_block_index(state: dict) -> int:
        return max(state.keys(), key=lambda idx: sum(Solver.dist(state, idx, jdx) for jdx in state))

    def run(self):
        curr_state = self.init_state
        n = len(curr_state)
        # bring all blocks together
        actions = []
        selected = None
        while self.get_all_pairs_min_dist(curr_state) > (n * (n - 1) * self.step_size):
            farthest_block = Solver.get_farthest_block_index(curr_state)
            if selected != farthest_block:
                if selected is not None:
                    actions.append((Action.DROP, selected))
                selected = farthest_block
                actions.append((Action.PICK, selected))
            # move_towards_median
            best_action = min(self.get_available_actions(curr_state, selected),
                              key=lambda x: Solver.get_all_pairs_min_dist(self.get_next_state(curr_state, selected, x)))
            actions.append(best_action)
            curr_state = self.get_next_state(curr_state, best_action)
        actions.append((Action.DROP, selected))
