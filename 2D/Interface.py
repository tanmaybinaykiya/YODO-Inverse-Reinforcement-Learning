import json
import os

from BlockWorld import BlockWorld
from constants import FRAME_LOCATION


def remove_frames_with_no_action():
    with open("state_action_map.json", 'r') as f:
        state_action_map = json.loads(f.read())
    files_with_actions = list(state_action_map.keys())
    file_list = os.listdir(FRAME_LOCATION)

    for file in file_list:
        if file[:-4] not in files_with_actions:
            os.remove(os.path.join(FRAME_LOCATION, file))


def main():
    block_world = BlockWorld(1000, 950, 4, 1)
    block_world.run_environment(record=True)
    # remove_frames_with_no_action()
    # print("FINALLY:", actions_takens)


if __name__ == "__main__":
    main()
