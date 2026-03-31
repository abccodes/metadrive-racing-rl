"""Racing map variants for local evaluation and training.

Maps are defined as reusable track specs built from straight and curve blocks.
Use set_racing_map() before creating an env to switch the racing map class used
by MetaDrive's racing environment.
"""

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pg_space import Parameter
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.straight import Straight
from metadrive.constants import PGLineType


def _build_track(pg_map, specs):
    """Build a track from a sequence of straight and curve block specs."""
    parent_node_path = pg_map.engine.worldNP
    physics_world = pg_map.engine.physics_world

    lane_num = pg_map.config["lane_num"]
    lane_width = pg_map.config["lane_width"]

    last_block = FirstPGBlock(
        pg_map.road_network,
        lane_width=lane_width,
        lane_num=lane_num,
        render_root_np=parent_node_path,
        physics_world=physics_world,
        remove_negative_lanes=True,
        side_lane_line_type=PGLineType.GUARDRAIL,
        center_line_type=PGLineType.GUARDRAIL,
    )
    pg_map.blocks.append(last_block)

    block_index = 1
    for kind, params in specs:
        block_cls = Straight if kind == "straight" else Curve
        last_block = block_cls(
            block_index,
            last_block.get_socket(0),
            pg_map.road_network,
            1,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.GUARDRAIL,
            center_line_type=PGLineType.GUARDRAIL,
        )
        last_block.construct_from_config(params, parent_node_path, physics_world)
        pg_map.blocks.append(last_block)
        block_index += 1


TRACK_SPECS = {
    "circuit": [
        ("straight", {Parameter.length: 100}),
        ("curve", {Parameter.length: 200, Parameter.radius: 100, Parameter.angle: 90, Parameter.dir: 1}),
        ("straight", {Parameter.length: 100}),
        ("curve", {Parameter.length: 50, Parameter.radius: 20, Parameter.angle: 180, Parameter.dir: 0}),
        ("straight", {Parameter.length: 100}),
        ("curve", {Parameter.length: 100, Parameter.radius: 40, Parameter.angle: 140, Parameter.dir: 0}),
    ],
    "hairpin": [
        ("straight", {Parameter.length: 100}),
        ("curve", {Parameter.length: 40, Parameter.radius: 25, Parameter.angle: 180, Parameter.dir: 1}),
        ("straight", {Parameter.length: 150}),
        ("curve", {Parameter.length: 40, Parameter.radius: 25, Parameter.angle: 180, Parameter.dir: 0}),
        ("straight", {Parameter.length: 150}),
        ("curve", {Parameter.length: 40, Parameter.radius: 30, Parameter.angle: 180, Parameter.dir: 1}),
        ("straight", {Parameter.length: 100}),
        ("curve", {Parameter.length: 60, Parameter.radius: 40, Parameter.angle: 90, Parameter.dir: 0}),
        ("straight", {Parameter.length: 80}),
    ],
    "s_curve": [
        ("straight", {Parameter.length: 100}),
        ("curve", {Parameter.length: 60, Parameter.radius: 45, Parameter.angle: 90, Parameter.dir: 1}),
        ("straight", {Parameter.length: 45}),
        ("curve", {Parameter.length: 60, Parameter.radius: 45, Parameter.angle: 90, Parameter.dir: 0}),
        ("straight", {Parameter.length: 45}),
        ("curve", {Parameter.length: 60, Parameter.radius: 45, Parameter.angle: 90, Parameter.dir: 1}),
        ("straight", {Parameter.length: 90}),
        ("curve", {Parameter.length: 60, Parameter.radius: 50, Parameter.angle: 90, Parameter.dir: 0}),
        ("straight", {Parameter.length: 120}),
    ],
    "chicane": [
        ("straight", {Parameter.length: 140}),
        ("curve", {Parameter.length: 35, Parameter.radius: 35, Parameter.angle: 45, Parameter.dir: 1}),
        ("straight", {Parameter.length: 30}),
        ("curve", {Parameter.length: 35, Parameter.radius: 35, Parameter.angle: 45, Parameter.dir: 0}),
        ("straight", {Parameter.length: 120}),
        ("curve", {Parameter.length: 35, Parameter.radius: 35, Parameter.angle: 45, Parameter.dir: 0}),
        ("straight", {Parameter.length: 30}),
        ("curve", {Parameter.length: 35, Parameter.radius: 35, Parameter.angle: 45, Parameter.dir: 1}),
        ("straight", {Parameter.length: 100}),
    ],
    "sweeper": [
        ("straight", {Parameter.length: 150}),
        ("curve", {Parameter.length: 160, Parameter.radius: 110, Parameter.angle: 120, Parameter.dir: 1}),
        ("straight", {Parameter.length: 80}),
        ("curve", {Parameter.length: 160, Parameter.radius: 100, Parameter.angle: 120, Parameter.dir: 0}),
        ("straight", {Parameter.length: 150}),
        ("curve", {Parameter.length: 120, Parameter.radius: 90, Parameter.angle: 90, Parameter.dir: 1}),
        ("straight", {Parameter.length: 90}),
    ],
    "double_hairpin": [
        ("straight", {Parameter.length: 90}),
        ("curve", {Parameter.length: 40, Parameter.radius: 22, Parameter.angle: 180, Parameter.dir: 1}),
        ("straight", {Parameter.length: 50}),
        ("curve", {Parameter.length: 40, Parameter.radius: 22, Parameter.angle: 180, Parameter.dir: 0}),
        ("straight", {Parameter.length: 90}),
        ("curve", {Parameter.length: 45, Parameter.radius: 24, Parameter.angle: 180, Parameter.dir: 1}),
        ("straight", {Parameter.length: 120}),
    ],
    "technical": [
        ("straight", {Parameter.length: 60}),
        ("curve", {Parameter.length: 50, Parameter.radius: 30, Parameter.angle: 90, Parameter.dir: 1}),
        ("straight", {Parameter.length: 35}),
        ("curve", {Parameter.length: 70, Parameter.radius: 50, Parameter.angle: 120, Parameter.dir: 0}),
        ("straight", {Parameter.length: 30}),
        ("curve", {Parameter.length: 35, Parameter.radius: 40, Parameter.angle: 60, Parameter.dir: 1}),
        ("straight", {Parameter.length: 30}),
        ("curve", {Parameter.length: 45, Parameter.radius: 24, Parameter.angle: 180, Parameter.dir: 0}),
        ("straight", {Parameter.length: 70}),
        ("curve", {Parameter.length: 55, Parameter.radius: 36, Parameter.angle: 90, Parameter.dir: 1}),
    ],
    "mixed_long": [
        ("straight", {Parameter.length: 160}),
        ("curve", {Parameter.length: 130, Parameter.radius: 95, Parameter.angle: 100, Parameter.dir: 1}),
        ("straight", {Parameter.length: 90}),
        ("curve", {Parameter.length: 55, Parameter.radius: 36, Parameter.angle: 90, Parameter.dir: 0}),
        ("straight", {Parameter.length: 80}),
        ("curve", {Parameter.length: 45, Parameter.radius: 24, Parameter.angle: 180, Parameter.dir: 1}),
        ("straight", {Parameter.length: 140}),
        ("curve", {Parameter.length: 120, Parameter.radius: 85, Parameter.angle: 100, Parameter.dir: 0}),
        ("straight", {Parameter.length: 40}),
        ("curve", {Parameter.length: 55, Parameter.radius: 36, Parameter.angle: 90, Parameter.dir: 1}),
        ("straight", {Parameter.length: 90}),
    ],
    # Approximation of the fixed online evaluation maps shared by the course staff.
    "server_map1": [
        ("straight", {Parameter.length: 150}),
        ("curve", {Parameter.length: 110, Parameter.radius: 75, Parameter.angle: 90, Parameter.dir: 1}),
        ("straight", {Parameter.length: 220}),
        ("curve", {Parameter.length: 60, Parameter.radius: 28, Parameter.angle: 170, Parameter.dir: 1}),
        ("straight", {Parameter.length: 95}),
        ("curve", {Parameter.length: 55, Parameter.radius: 26, Parameter.angle: 160, Parameter.dir: 0}),
        ("straight", {Parameter.length: 85}),
        ("curve", {Parameter.length: 50, Parameter.radius: 24, Parameter.angle: 135, Parameter.dir: 1}),
        ("straight", {Parameter.length: 170}),
        ("curve", {Parameter.length: 115, Parameter.radius: 78, Parameter.angle: 105, Parameter.dir: 1}),
        ("straight", {Parameter.length: 120}),
    ],
    "server_map2": [
        ("straight", {Parameter.length: 170}),
        ("curve", {Parameter.length: 115, Parameter.radius: 78, Parameter.angle: 95, Parameter.dir: 1}),
        ("straight", {Parameter.length: 255}),
        ("curve", {Parameter.length: 145, Parameter.radius: 82, Parameter.angle: 185, Parameter.dir: 1}),
        ("straight", {Parameter.length: 120}),
    ],
    "server_map3": [
        ("straight", {Parameter.length: 95}),
        ("curve", {Parameter.length: 55, Parameter.radius: 30, Parameter.angle: 90, Parameter.dir: 1}),
        ("straight", {Parameter.length: 120}),
        ("curve", {Parameter.length: 55, Parameter.radius: 30, Parameter.angle: 90, Parameter.dir: 0}),
        ("straight", {Parameter.length: 130}),
        ("curve", {Parameter.length: 55, Parameter.radius: 30, Parameter.angle: 90, Parameter.dir: 1}),
        ("straight", {Parameter.length: 120}),
        ("curve", {Parameter.length: 55, Parameter.radius: 30, Parameter.angle: 90, Parameter.dir: 0}),
        ("straight", {Parameter.length: 130}),
        ("curve", {Parameter.length: 55, Parameter.radius: 30, Parameter.angle: 90, Parameter.dir: 1}),
        ("straight", {Parameter.length: 135}),
    ],
    "server_map4": [
        ("straight", {Parameter.length: 120}),
        ("curve", {Parameter.length: 65, Parameter.radius: 34, Parameter.angle: 180, Parameter.dir: 1}),
        ("straight", {Parameter.length: 180}),
        ("curve", {Parameter.length: 65, Parameter.radius: 34, Parameter.angle: 180, Parameter.dir: 0}),
        ("straight", {Parameter.length: 185}),
        ("curve", {Parameter.length: 65, Parameter.radius: 34, Parameter.angle: 180, Parameter.dir: 1}),
        ("straight", {Parameter.length: 180}),
        ("curve", {Parameter.length: 75, Parameter.radius: 42, Parameter.angle: 90, Parameter.dir: 0}),
        ("straight", {Parameter.length: 200}),
    ],
}


def _make_racing_map_class(name):
    class _GeneratedRacingMap(PGMap):
        def _generate(self):
            assert len(self.road_network.graph) == 0, "Map is not empty, please create a new map to read config"
            _build_track(self, TRACK_SPECS[name])

    _GeneratedRacingMap.__name__ = f"RacingMap_{name}"
    _GeneratedRacingMap.__qualname__ = _GeneratedRacingMap.__name__
    _GeneratedRacingMap.__doc__ = f"Generated racing map for '{name}'."
    return _GeneratedRacingMap


RACING_MAPS = {name: _make_racing_map_class(name) for name in TRACK_SPECS}


def set_racing_map(map_name):
    """Monkey-patch the RacingMap class used by MultiAgentRacingEnv."""
    import metadrive.envs.marl_envs.marl_racing_env as racing_mod

    if map_name not in RACING_MAPS:
        raise ValueError(f"Unknown map '{map_name}'. Available: {list(RACING_MAPS.keys())}")

    original = racing_mod.RacingMap
    racing_mod.RacingMap = RACING_MAPS[map_name]

    def restore():
        racing_mod.RacingMap = original

    return restore
