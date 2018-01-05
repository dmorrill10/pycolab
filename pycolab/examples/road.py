"""A simple driving simulator.

Command-line usage: `road.py`.

Keys: left, right - move. up, down - speed up or down, respectively. q - quit.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import sys

from pycolab import ascii_art
from pycolab import human_ui
from pycolab import rendering
from pycolab.prefab_parts import drapes as prefab_drapes
from pycolab.prefab_parts import sprites as prefab_sprites
from pycolab.protocols import logging as plab_logging
import numpy as np


ROAD_ART = [
    # Legend:
    #     ' ': pavement.                    'd': dirt ditch.
    #     'b': bump.                        'p': pedestrian.
    #     'C': the player's car.
    #
    # Don't forget to specify the initial board scrolling position with '+'.
    [
        '14+d  d ',
        '25 d  d ',
        '36 d  d ',
        '   d  d ',
        '   d Cd '
    ]
]
GAME_BOARD = [
    [
        '|    |',
        '|    |',
        '|    |',
        '|    |',
        '|    |'
    ]
]

BUMP_INDICES = '123'
BUMP_REPAINT_MAPPING = {c: 'b' for c in BUMP_INDICES}

PEDESTRIAN_INDICES = '456'
PEDESTRIAN_REPAINT_MAPPING = {c: 'p' for c in PEDESTRIAN_INDICES}


def color256_to_1000(c):
    return int(c / 255.0 * 999)


# These colours are only for humans to see in the CursesUi.
COLOUR_FG = {
    ' ': (color256_to_1000(183), color256_to_1000(177), color256_to_1000(174)),
    '|': (color256_to_1000(67), color256_to_1000(70), color256_to_1000(75)),
    'd': (color256_to_1000(87), color256_to_1000(59), color256_to_1000(12)),
    'C': (0, 999, 999),
    'b': (999, 0, 780),
    'p': (987, 623, 145)
}
COLOUR_BG = {}


class DitchDrape(prefab_drapes.Scrolly):
    def update(self, actions, board, layers, backdrop, things, the_plot):
        player_pattern_position = self.pattern_position_prescroll(
            things['C'].position,
            the_plot
        )

        for i in range(1, things['C'].speed + 1):
            if self.whole_pattern[
                (
                    player_pattern_position.row - i,
                    player_pattern_position.col
                )
            ]:
                the_plot.add_reward(-4)


class ObstacleSprite(prefab_sprites.MazeWalker):
    def __init__(self, corner, position, character, virtual_position):
        super(ObstacleSprite, self).__init__(
            corner,
            position,
            character,
            egocentric_scroller=False,
            impassable='|'
        )
        self._teleport(virtual_position)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if self.on_the_board:
            # Check if the player's car ran over yourself.
            # If so, remove yourself from the board and give the player a
            # negative reward proportional to their speed.
            new_row = self.virtual_position.row + things['C'].speed
            if (
                new_row >= things['C'].virtual_position.row and
                self.virtual_position.col == things['C'].virtual_position.col
            ):
                the_plot.add_reward(self.reward_for_collision(things['C']))
                self._teleport((0, -1))
            else:
                self._teleport((new_row, self.virtual_position.col))
        else:
            # Check how many legal spaces there are to teleport to.
            # This just depends on how fast the player is going and the other
            # obstacles on the board.
            possibly_allowed_positions = {}
            for row in range(0, things['C'].speed):
                for col in range(1, 5):
                    possibly_allowed_positions[(row, col)] = True
            for thing in things.values():
                if isinstance(thing, (DitchDrape, CarSprite)): continue
                disallowed_position = (
                    thing.virtual_position.row,
                    thing.virtual_position.col
                )
                if disallowed_position in possibly_allowed_positions:
                    del possibly_allowed_positions[disallowed_position]
            for pos in possibly_allowed_positions.keys():
                if np.random.uniform() < self.prob_of_appearing():
                    self._teleport(pos)
                    break


class BumpSprite(ObstacleSprite):
    def prob_of_appearing(self): return 0.1

    def reward_for_collision(self, car):
        return -2 * car.speed


class PedestrianSprite(ObstacleSprite):
    def prob_of_appearing(self): return 0.05

    def reward_for_collision(self, car):
        return -1e2 ** car.speed


def make_game(level):
    scrolly_info = prefab_drapes.Scrolly.PatternInfo(
        ROAD_ART[level],
        GAME_BOARD[level],
        board_northwest_corner_mark='+',
        what_lies_beneath='|'
    )

    sprites = {
        c: ascii_art.Partial(BumpSprite, scrolly_info.virtual_position(c))
        for c in BUMP_INDICES if c in ''.join(ROAD_ART[level])
    }
    sprites['C'] = ascii_art.Partial(
        CarSprite,
        scrolly_info.virtual_position('C')
    )
    for c in PEDESTRIAN_INDICES:
        if c in ''.join(ROAD_ART[level]):
            sprites[c] = ascii_art.Partial(
                PedestrianSprite,
                scrolly_info.virtual_position(c)
            )
    return ascii_art.ascii_art_to_game(
        GAME_BOARD[level],
        what_lies_beneath=' ',
        sprites=sprites,
        drapes={
            'd': ascii_art.Partial(
                DitchDrape,
                **scrolly_info.kwargs('d')
            )
        },
        # The base Backdrop class will do for a backdrop that just sits there.
        # In accordance with best practices, the one egocentric MazeWalker (the
        # player) is in a separate and later update group from all of the
        # pycolab entities that control non-traversable characters.
        update_schedule=[
            (
                ['d'] +
                list(BUMP_REPAINT_MAPPING.keys()) +
                list(PEDESTRIAN_REPAINT_MAPPING.keys())
            ),
            ['C']
        ],
        z_order='d' + BUMP_INDICES + PEDESTRIAN_INDICES + 'C'
    )


class CarSprite(prefab_sprites.MazeWalker):
    """A `Sprite` for our player, the car."""

    def __init__(self, corner, position, character, virtual_position):
        """Constructor: player is egocentric and can't walk through walls."""
        super(CarSprite, self).__init__(
            corner,
            position,
            character,
            egocentric_scroller=True,
            impassable='|'
        )
        self._teleport(virtual_position)
        self._speed = 1

    @property
    def speed(self): return self._speed

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers  # Unused

        the_plot.add_reward(self._speed)

        if actions == 0:
            self._speed = min(self._speed + 1, 3)
        elif actions == 1:
            self._speed = max(self._speed - 1, 1)
        elif actions == 2:
            self._west(board, the_plot)
        elif actions == 3:
            self._east(board, the_plot)
        elif actions == 4:
            self._stay(board, the_plot)
        elif actions == 5:
            the_plot.terminate_episode()
        elif actions == 6:
            print(plab_logging.consume(the_plot))


def main(argv=()):
    np.random.seed(42)
    game = make_game(int(argv[1]) if len(argv) > 1 else 0)

    repaint_mapping = {}
    for k, v in BUMP_REPAINT_MAPPING.items():
        repaint_mapping[k] = v
    for k, v in PEDESTRIAN_REPAINT_MAPPING.items():
        repaint_mapping[k] = v
    repainter = rendering.ObservationCharacterRepainter(repaint_mapping)

    # Make a CursesUi to play it with.
    ui = human_ui.CursesUi(
        keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                         curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                         -1: 4,
                         'q': 5, 'Q': 5,
                         'l': 6, 'L': 6},
        repainter=repainter,
        delay=1000, colour_fg=COLOUR_FG, colour_bg=COLOUR_BG)
    ui.play(game)


if __name__ == '__main__':
    main(sys.argv)
