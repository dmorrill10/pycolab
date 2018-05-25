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
from itertools import product, permutations


def valid_meta_configurations(
    num_rows,
    num_bumps,
    num_pedestrians,
    num_speeds
):
    assert num_rows > 1
    num_rows_above_car = num_rows - 1
    num_columns = 4
    num_spaces = num_rows_above_car * num_columns
    for speed in range(num_speeds):
        for car_position in range(num_columns):
            for num_present_bumps in range(num_bumps + 1):
                for num_present_pedestrians in range(num_pedestrians + 1):
                    if (
                        num_present_bumps + num_present_pedestrians <=
                        num_spaces
                    ):
                        yield (
                            speed,
                            car_position,
                            num_present_bumps,
                            num_present_pedestrians
                        )


def determinstic_state_component_generator(
    num_rows,
    num_bumps,
    num_pedestrians,
    num_speeds
):
    assert num_rows > 1
    num_rows_above_car = num_rows - 1
    num_columns = 4
    available_columns = {}
    for i in range(num_rows_above_car):
        for j in range(num_columns):
            available_columns[(i, j)] = True

    def legal_positions():
        return available_columns.keys()

    for (
        speed,
        car_position,
        num_present_bumps,
        num_present_pedestrians
    ) in valid_meta_configurations(
        num_rows,
        num_bumps,
        num_pedestrians,
        num_speeds
    ):
        bump_positions = []
        pedestrian_positions = []
        for bump_positions in combinations(
            legal_positions(),
            num_present_bumps
        ):
            for pos in bump_positions:
                del available_columns[pos]
            for pedestrian_positions in combinations(
                legal_positions(),
                num_present_pedestrians
            ):
                yield (
                    speed,
                    num_rows,
                    bump_positions,
                    pedestrian_positions,
                    car_position)
            for pos in bump_positions:
                available_columns[pos] = True


def determinstic_state_generator(
    num_rows,
    num_bumps,
    num_pedestrians,
    num_speeds
):
    for (
        speed, num_rows, bump_positions, pedestrian_positions, car_position
    ) in determinstic_state_component_generator(
        num_rows, num_bumps, num_pedestrians, num_speeds
    ):
        yield speed, road_state(
            num_rows, bump_positions, pedestrian_positions, car_position)


def game_board(num_rows):
    assert num_rows > 1
    return ['|    |'] * num_rows


def bump_indices(num_bumps):
    assert num_bumps >= 0
    return ''.join([str(i) for i in range(1, num_bumps + 1)])


def pedestrian_indices(num_pedestrians, num_bumps):
    assert num_bumps >= 0
    assert num_pedestrians >= 0
    return ''.join(
        [str(i + num_bumps + 1) for i in range(1, num_pedestrians + 1)])


def car_row_array(position=2, show_walls=True):
    if show_walls:
        row = [' ', 'd', ' ', ' ', 'd', ' ']
    else:
        row = ['d', ' ', ' ', 'd']
    assert position < len(row) - 1
    row[position + int(show_walls)] = 'C'
    return row


def car_row(position=2):
    return ''.join(car_row_array(position))


def road_state(
    num_rows,
    bump_positions=[],
    pedestrian_positions=[],
    car_position=2,
    show_walls=True
):
    assert num_rows > 1

    if show_walls:
        board = (
            [['+', 'd', ' ', ' ', 'd', ' ']] +
            [[' ', 'd', ' ', ' ', 'd', ' '] for _ in range(num_rows - 2)] +
            [car_row_array(car_position, show_walls=show_walls)]
        )
    else:
        board = (
            [['d', ' ', ' ', 'd']] +
            [['d', ' ', ' ', 'd'] for _ in range(num_rows - 2)] +
            [car_row_array(car_position, show_walls=show_walls)]
        )
    for i, j in bump_positions:
        col = j + int(show_walls)
        if i < len(board) and 0 <= col <= len(board[i]) - int(show_walls):
            board[i][col] = 'b'
    for i, j in pedestrian_positions:
        col = j + int(show_walls)
        if i < len(board) and 0 <= col <= len(board[i]) - int(show_walls):
            board[i][col] = 'p'
    return '\n'.join([''.join(row) for row in board])


def road_art(num_rows, num_bumps, num_pedestrians):
    '''
    Legend:
        ' ': pavement.                    'd': dirt ditch.
        'b': bump.                        'p': pedestrian.
        'C': the player's car.
    '''
    assert num_rows > 0
    assert num_bumps >= 0
    assert num_pedestrians >= 0

    wall_to_wall_width = 6
    max_width = max(num_bumps, num_pedestrians, wall_to_wall_width)
    return (
        [
            bump_indices(num_bumps) + ' ' * (max_width - num_bumps),
            (
                pedestrian_indices(num_pedestrians, num_bumps) +
                ' ' * (max_width - num_pedestrians)
            ),
            '+d  d ' + ' ' * (max_width - wall_to_wall_width)
        ] +
        [' d  d ' + ' ' * (max_width - wall_to_wall_width)] * (num_rows - 2) +
        [car_row() + ' ' * (max_width - wall_to_wall_width)])


NUM_ROWS = 5
NUM_BUMPS = 3
NUM_PEDESTRIANS = 3
ROAD_ART = [road_art(NUM_ROWS, NUM_BUMPS, NUM_PEDESTRIANS)]
GAME_BOARD = [game_board(NUM_ROWS)]
BUMP_INDICES = bump_indices(NUM_BUMPS)
BUMP_REPAINT_MAPPING = {c: 'b' for c in BUMP_INDICES}
PEDESTRIAN_INDICES = pedestrian_indices(NUM_PEDESTRIANS, NUM_BUMPS)
PEDESTRIAN_REPAINT_MAPPING = {c: 'p' for c in PEDESTRIAN_INDICES}


def color256_to_1000(c):
    return int(c / 255.0 * 999)


# These colours are only for humans to see in the CursesUi.
COLOUR_FG = {
    ' ': (color256_to_1000(183), color256_to_1000(177), color256_to_1000(174)),
    '|': (color256_to_1000(67), color256_to_1000(70), color256_to_1000(75)),
    'd': (color256_to_1000(87), color256_to_1000(59), color256_to_1000(12)),
    'C': (0, 999, 999),
    'b': (0, 0, 0),
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


class Car(object):
    def __init__(self, row, col, speed):
        self.row = row
        self.col = col
        self.speed = speed

    def position(self): return (self.row, self.col)

    def next(self, action, num_speeds):
        assert num_speeds > 0
        if action == 'u':
            return Car(self.row, self.col, min(self.speed + 1, num_speeds))
        elif action == 'd':
            return Car(self.row, self.col, max(self.speed - 1, 1))
        elif action == 'l':
            return Car(self.row, max(self.col - 1, 0), self.speed)
        elif action == 'r':
            return Car(self.row, min(self.col + 1, 0), self.speed)
        else:
            raise 'Unrecognized action, "{}".'.format(action)


class Obstacle(object):
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def position(self): return (self.row, self.col)

    def prob_of_appearing(self):
        raise 'Abstract. Must be <= 1/num_speeds.'

    def reward_for_collision(self):
        raise 'Abstract'

    def reward(self, car):
        if self.has_collided(car):
            return self.reward_for_collision()
        else:
            return 0

    def has_collided(self, car):
        was_in_front_of_car = self.row < car.row - car.speed
        is_under_or_behind_car = self.row >= car.row
        is_in_car_lane = self.col == car.col
        return (
            was_in_front_of_car and is_under_or_behind_car and is_in_car_lane)

    def next(self, car):
        next_row = self.row + car.speed
        return self.__class__(next_row, self.col)


class Bump(Obstacle):
    def prob_of_appearing(self): return 0.1

    def reward_for_collision(self, car):
        return -2 * car.speed


class Pedestrian(Obstacle):
    def prob_of_appearing(self): return 0.05

    def reward_for_collision(self, car):
        return -1e2 ** car.speed


def combinations(iterable, r, collection=tuple):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    iterable = tuple(iterable)
    n = len(iterable)
    if r > n:
        return
    indices = list(range(r))
    yield collection(iterable[i] for i in indices)
    while True:
        for i in range(r - 1, -1, -1):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield collection(iterable[i] for i in indices)


class Road(object):
    def __init__(self, num_rows, car, obstacles, num_speeds):
        self._num_rows = num_rows
        self._num_columns = 4
        self._car = car
        self._num_speeds = num_speeds
        self._obstacles = obstacles

        self._available_spaces = {}
        for pos in product(range(0, self._car.speed), range(4)):
            self._available_spaces[pos] = True
        for obs in self._obstacles:
            if self.obstacle_visible(obs):
                disallowed_position = obs.position()
                if disallowed_position in self._available_spaces:
                    del self._available_spaces[disallowed_position]

    def obstacle_visible(self, obstacle):
        return 0 <= obstacle.row < self._num_rows

    def successors(self, action):
        '''Generates successor, probability, reward tuples.'''
        next_car = self._car.next(action, self._num_speeds)

        hidden_obstacle_indices = [
            i
            for i in range(len(self._obstacles))
            if not self.obstacle_visible(self._obstacles[i])
        ]
        max_num_revealed_obstacles = min(
            len(hidden_obstacle_indices), len(self._available_spaces))

        for num_newly_visible_obstacles in range(
            max_num_revealed_obstacles + 1
        ):
            position_sets = list(permutations(
                self._available_spaces.keys(), num_newly_visible_obstacles
            ))
            sets_of_obstacle_indices_to_reveal = list(combinations(
                hidden_obstacle_indices,
                num_newly_visible_obstacles,
                collection=set
            ))

            for positions, reveal_indices in product(
                position_sets, sets_of_obstacle_indices_to_reveal
            ):
                assert len(positions) == len(reveal_indices)

                prob = 1.0
                j = 0
                next_obstacles = []
                reward = 0.0
                for i in range(len(self._obstacles)):
                    obs = self._obstacles[i]
                    p = obs.prob_of_appearing()
                    assert p <= 1
                    if i in reveal_indices:
                        next_obstacles.append(obs.__class__(*positions[j]))
                        prob *= p / float(len(self._available_spaces) - j)
                        j += 1
                    else:
                        next_obstacles.append(obs.next(self._car))
                        if j < len(self._available_spaces):
                            prob *= 1.0 - p
                    reward += next_obstacles[-1].reward(self._car)
                yield (
                    self.__class__(
                        self._num_rows,
                        next_car,
                        next_obstacles,
                        self._num_speeds
                    ),
                    prob,
                    reward
                )

    def to_key(self, show_walls=False):
        return (self.to_s(show_walls=show_walls), self._car.speed)

    def to_s(self, show_walls=True):
        bumps = [
            obstacle.position() for obstacle in self._obstacles
            if isinstance(obstacle, Bump)
        ]
        pedestrians = [
            obstacle.position() for obstacle in self._obstacles
            if isinstance(obstacle, Pedestrian)
        ]
        return road_state(
            self._num_rows, bumps, pedestrians, show_walls=show_walls)


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
            # If so, give the player a
            # negative reward proportional to their speed.
            new_row = self.virtual_position.row + things['C'].speed
            if (
                new_row >= things['C'].virtual_position.row and
                self.virtual_position.col == things['C'].virtual_position.col
            ):
                the_plot.add_reward(self.reward_for_collision(things['C']))
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

            allowed_positions = possibly_allowed_positions
            reveal = (
                len(allowed_positions) > 0 and
                np.random.uniform() < self.prob_of_appearing())
            if reveal:
                allowed_positions = tuple(allowed_positions.keys())
                i = np.random.randint(0, len(allowed_positions))
                self._teleport(allowed_positions[i])


class BumpSprite(ObstacleSprite, Bump): pass


class PedestrianSprite(ObstacleSprite, Pedestrian): pass


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
