from collections import defaultdict
import time
import numpy as np

from BlockWorld_t import BlockWorld
from constants import Action, COLORS_STR

states_x, states_y = 1000, 900
blocks_count = 4
non_pick_actions = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.DROP]
actions = non_pick_actions.copy()
actions.append(Action.PICK)
action_to_idx = {action: idx for idx, action in enumerate(actions)}
idx_to_action = {idx: action for idx, action in enumerate(actions)}


def get_allowed_actions(prev_actn):
    if not prev_actn or prev_actn[0] == Action.DROP:
        return [(Action.PICK, np.random.randint(0, blocks_count))]
    else:
        return [(a, prev_actn[1]) for a in non_pick_actions]


def get_random_action(prev_actn):
    poss_actions = get_allowed_actions(prev_actn)
    return poss_actions[np.random.randint(0, len(poss_actions))]


def q_learning():
    # curr_state is a n-tuple( (x1, y1), (x2, y2), (x3, y3), (x4, y4), selectedBlockId)
    # q = { (state, action) : value }

    states = ((), (), (), (),)
    gamma = 0.1

    q = defaultdict(lambda: 0)
    episode_count = 2
    prev_action = None
    for ep in range(episode_count):
        blockWorld = BlockWorld(states_x, states_y, blocks_count, 1, record=True)
        print("Goal: ", [COLORS_STR[i] for stack in blockWorld.goal_config for i in stack])
        while blockWorld.get_reward() != 0:

            blockWorld.prerender()
            state = blockWorld.get_state_as_tuple()
            action, block_id = get_random_action(prev_action)
            next_state = blockWorld.get_next_state_based_on_state_tuple(state, (action, block_id))
            q_val = gamma * max([q[next_state, b] for b in get_allowed_actions((action, block_id))])
            q[(blockWorld.get_state_as_tuple(), action)] = blockWorld.get_reward_for_state_tuple(state, blockWorld.goal_config.tolist()) + q_val
            blockWorld.update_state_from_tuple(next_state)
            prev_action = action, block_id
            blockWorld.render()
            # time.sleep(5)

def random_exploration():
    # curr_state is a n-tuple( (x1, y1), (x2, y2), (x3, y3), (x4, y4), selectedBlockId)
    # q = { (state, action) : value }

    states = ((), (), (), (),)
    gamma = 0.1

    q = defaultdict(lambda: 0)
    episode_count = 2
    prev_action = None
    for ep in range(episode_count):
        blockWorld = BlockWorld(states_x, states_y, blocks_count, 1, record=True)
        print("Goal: ", [COLORS_STR[i] for stack in blockWorld.goal_config for i in stack])
        while blockWorld.get_reward() != 0:

            blockWorld.prerender()
            state = blockWorld.get_state_as_tuple()
            action, block_id = get_random_action(prev_action)
            next_state = blockWorld.get_next_state_based_on_state_tuple(state, (action, block_id))
            q_val = gamma * max([q[next_state, b] for b in get_allowed_actions((action, block_id))])
            q[(blockWorld.get_state_as_tuple(), action)] = blockWorld.get_reward_for_state_tuple(state, blockWorld.goal_config.tolist()) + q_val
            blockWorld.update_state_from_tuple(next_state)
            prev_action = action, block_id
            blockWorld.render()
            # time.sleep(5)


def random_exploration2():
    states_x, states_y = 100, 100

    actions = non_pick_actions.copy()
    actions.append(Action.PICK)
    action_to_idx = {action: idx for idx, action in enumerate(actions)}
    idx_to_action = {idx: action for idx, action in enumerate(actions)}

    states = np.zeros((states_x, states_y))
    gamma = 0.1

    q = np.zeros((states_x, states_y, len(non_pick_actions)))
    episode_count = 2
    prev_action = None
    for ep in range(episode_count):
        blockWorld = BlockWorld(1000, 950, blocks_count, 1, record=True)
        print("Goal: ", [COLORS_STR[i] for stack in blockWorld.goal_config for i in stack])
        while blockWorld.get_reward() != 0:
            blockWorld.prerender()
            action, block_id = get_random_action(prev_action)
            print("Action chosen :", action, block_id)
            if action != Action.DROP and action != Action.PICK:
                blockWorld.move_block_by_action(action, block_id)

            prev_action = action, block_id
            blockWorld.render()


if __name__ == '__main__':
    main()
