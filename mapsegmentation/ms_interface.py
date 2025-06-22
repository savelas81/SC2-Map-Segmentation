from collections import defaultdict
from queue import PriorityQueue
from typing import Union

import numpy as np
from matplotlib import pyplot as plt

# from matplotlib import pyplot as plt
from sc2.units import Units

from mapsegmentation import Region
from sc2.position import Point3, Point2

from sc2.unit import Unit

from mapsegmentation.dataclasses.segmented_map import SegmentedMap
from mapsegmentation.map_segmentation import map_segmentation
from mapsegmentation.dataclasses.passage import Cliff
from dataclasses import dataclass


@dataclass
class ConnectionInfo:
    id: int
    passage_center: Point2  # these are waypoints between
    segment_center: Point2


class MSInterface:
    """
    Strategic-level interface for interacting with the StarCraft II map segmentation system.

    This class provides access to a segmented map structure, allowing bots to reason about
    terrain features like regions, passages, and choke points. It supports segment based
    pathfinding and unit grouping, making strategic movement and control more efficient.

    Key Features:
    - `units_by_segment`: Groups units by their closest segment to facilitate localized analysis and control.
    - `units_in_segment`: Retrieves units from a pre-grouped dictionary that are located within a specific segment.
    - `build_segment_graph`: Constructs a graph of connected regions (segments), allowing for exclusion of specific
       passage types (e.g., cliffs) and regions, or blocking specific directed connections.
    - `find_path_with_segmented_graph`: Computes a waypoint-based path between two segments using the precomputed
       segment graph and passage connections, optimized by passage-to-passage distance (not segment center to center).
    """

    def __init__(self, bot):
        """
        Initialize the MSInterface with a bot AI instance.
        :param bot: BotAI instance
        """
        self.bot = bot
        self.segmented_map = None
        self.passages_grid = None

    def on_start(self):
        """
        Segment the map using the map_segmentation function.
        """
        self.segmented_map: SegmentedMap = map_segmentation(self.bot)

        # self.for_testing()

    def update_connections(self) -> None:
        """
        Updates the passability of the segmented map based on destructables and mineral fields.
        """
        self.segmented_map.update_passability(self.bot.destructables, self.bot.mineral_field)

    def for_testing(self) -> None:
        """
        For testing purposes only.
        It serves as a placeholder to test and demonstrate functionality.
        build_segment_graph filter out Cliff by default
        """
        if self.segmented_map is None:
            raise ValueError("MapData is not initialized.")

        # displays the segmented map and placement grid
        print("Close the plot window to continue")
        plt.figure(figsize=(12, 6))
        plt.axis("off")
        plt.title(f"Map: {self.bot.game_info.map_name}")
        plt.subplot(1, 2, 2)
        self.segmented_map.imshow("Segmented grid")
        plt.subplot(1, 2, 1)
        plt.imshow(self.bot.game_info.placement_grid.data_numpy, origin='lower')
        plt.title("Placement grid")
        plt.show()

        # Get the bots starting position
        position = self.bot.start_location

        # Get the ID of the closest segment. Position can be Unit or Point2
        segment_id = self.closest_segment_id(position=position)

        # Get a list of all connections : ConnectionInfo objects
        # These are (exits) connected to this segment
        # TODO: figure out what more we need in connections in the future
        connections = self.get_connections(segment_id=segment_id)

        # Group the bots units into segments based on proximity
        # {segment_id: [Unit, Unit, ...], ...}
        units_by_segment: dict = self.units_by_segments(units=self.bot.units)

        # Same as above, but for enemy units.
        enemy_units_by_segment: dict = self.units_by_segments(units=self.bot.enemy_units)

        # Units located in segment. (or this is the closest segment for the unit)
        units_in_segment: Units = self.units_in_segment(units_by_segment=enemy_units_by_segment,
                                                        segment_id=segment_id)

        # Builds a segment graph used for calculating pathing.
        # This example removes all Cliff passages,
        # prevents passing through segments with IDs 2 and 5,
        # and avoids connections from segment 6 to 2 and from 7 to 4. Direction matters!
        ground_segmented_graph = self.build_segment_graph(passages_to_avoid=Cliff,
                                                          segments_to_avoid=[2, 5],
                                                          connections_to_avoid={6: 2, 7: 4})

        # basic pathing example
        start_segment_id = self.closest_segment_id(position=self.bot.start_location)
        goal_segment_id = self.closest_segment_id(position=self.bot.enemy_start_locations[0])
        ground_segmented_graph = self.build_segment_graph(passages_to_avoid=Cliff,
                                                          segments_to_avoid=[],
                                                          connections_to_avoid={})
        # segmented path is list of segment IDs that we need to go through to reach the goal segment
        segment_path = self.find_segment_path(segmented_graph=ground_segmented_graph,
                                              start_segment_id=start_segment_id,
                                              goal_segment_id=goal_segment_id)
        # Convert segment path to waypoints.
        # Waypoints are segment centers.
        path = self.convert_segment_path_to_waypoints(segment_path=segment_path)

        if path:
            for a, b in zip(path, path[1:]):
                height_a = self.bot.get_terrain_z_height(a) + 1
                height_b = self.bot.get_terrain_z_height(b) + 1
                self.bot.client.debug_line_out(Point3((a.x + 0.5, a.y + 0.5, height_a)),
                                               Point3((b.x + 0.5, b.y + 0.5, height_b)), color=(255, 0, 0))

        # Creates boolean mask for the specified segment (region).
        # Note: Ramps are not segments so they Units in rams don't show in these masks
        # Is this needed?
        # Could be used with MapAnalyzer?
        mask = self.segment_mask(segment_id=segment_id)

        print("debug")
        plt.figure(figsize=(6, 6))
        plt.imshow(mask, cmap='gray', origin='lower')
        plt.title(f"Segment Mask for Region ID: {segment_id}")
        plt.axis('off')
        plt.show()

    def find_segment_path(self,
                          segmented_graph: dict[int, list[ConnectionInfo]],
                          start_segment_id: int,
                          goal_segment_id: int) -> list[int]:
        """
        Calculates the shortest path from start_segment_id to goal_segment_id using Dijkstra,
        where the cost is the sum of distances between passage centers, including:
        - Distance from start_region.center to first passage center
        - Sum of distances between passage centers
        - Distance from last passage center to goal_region.center

        Returns a list of segment IDs forming the optimal path.

        :param segmented_graph: Segment graph with segment ID as keys and ConnectionInfo lists as values.
        :param start_segment_id: Starting segment ID.
        :param goal_segment_id: Goal segment ID.
        :return: Ordered list of segment IDs forming the shortest path.
        """
        visited = set()
        came_from: dict[int, tuple[int, Point2]] = {}  # segment_id -> (prev_segment_id, passage_center)
        cost_so_far = defaultdict(lambda: float('inf'))
        pq = PriorityQueue()

        start_region = self._segment_by_id(start_segment_id)
        goal_region = self._segment_by_id(goal_segment_id)

        cost_so_far[start_segment_id] = 0
        pq.put((0, start_segment_id, start_region.center))

        while not pq.empty():
            current_cost, current_segment, current_position = pq.get()

            if current_segment == goal_segment_id:
                break

            if current_segment in visited:
                continue
            visited.add(current_segment)

            for conn in segmented_graph.get(current_segment, []):
                next_segment = conn.id
                passage_pos = conn.passage_center
                travel_cost = current_position.distance_to(passage_pos)

                # Estimate final leg to goal
                end_travel_cost = passage_pos.distance_to(goal_region.center) if next_segment == goal_segment_id else 0
                new_cost = current_cost + travel_cost + end_travel_cost

                if new_cost < cost_so_far[next_segment]:
                    cost_so_far[next_segment] = new_cost
                    came_from[next_segment] = (current_segment, passage_pos)
                    pq.put((new_cost, next_segment, passage_pos))

        # Reconstruct path
        if goal_segment_id not in came_from:
            return []

        path = [goal_segment_id]
        current = goal_segment_id
        while current != start_segment_id:
            prev, _ = came_from[current]
            path.append(prev)
            current = prev

        path.reverse()
        return path[1:]

    def convert_segment_path_to_waypoints(self, segment_path: list[int]) -> list[Point2]:
        """
        Converts segmented path (list of segment IDs) to a list of waypoints.
        """
        if not segment_path:
            return []

        waypoints = []
        for segment_id in segment_path:
            segment = self._segment_by_id(segment_id)
            waypoints.append(segment.center)

        return waypoints

    def segment_center(self, segment_id: int) -> Point2:
        """
        Returns the center point of the specified segment (region).

        :param segment_id: The ID of the region/segment.
        :return: Point2 representing the center of the segment.
        """
        if self.segmented_map is None:
            raise ValueError("MapData is not initialized.")

        segment = self._segment_by_id(segment_id)
        return segment.center

    def segment_id(self, segment: Region) -> int:
        """
        Returns the ID of the specified segment (region).

        :param segment: The Region object representing the segment.
        :return: The ID of the segment.
        """
        if self.segmented_map is None:
            raise ValueError("MapData is not initialized.")

        return segment.id

    def units_in_segment(self, units_by_segment: dict[int, list[Unit]], segment_id: int) -> Units:
        """
        Returns a Units object containing units located in the specified segment.
        If the segment ID is not found or has no units, returns an empty Units object.
        """
        units_list = units_by_segment.get(segment_id, [])
        return Units(units_list, self.bot)

    def segment_mask(self, segment_id: int) -> np.ndarray:
        """
        Returns a boolean mask for the specified segment (region).
        The mask has the same shape as the region grid, with True values
        where the segment_id matches and False elsewhere.

        :param segment_id: The ID of the region/segment to mask.
        :return: A boolean np.ndarray mask.
        """
        if self.segmented_map is None:
            raise ValueError("MapData is not initialized.")

        return self.segmented_map.regions_grid.T == segment_id

    from sc2.units import Units

    def units_by_segments(self, units: Units) -> dict[int, Units]:
        """
        Groups a given Units object by the closest segment ID.

        :param units: A Units object (can be enemy_units, allies, or a filtered subset)
        :return: Dictionary mapping segment_id -> Units in that segment
        """
        segment_units: dict[int, list[Unit]] = {}

        for unit in units:
            segment_id = self.closest_segment_id(unit)
            segment_units.setdefault(segment_id, []).append(unit)

        # Convert lists to Units
        grouped_units: dict[int, Units] = {
            segment_id: Units(unit_list, self.bot)
            for segment_id, unit_list in segment_units.items()
        }

        return grouped_units

    def build_segment_graph(self,
                            passages_to_avoid=None,
                            segments_to_avoid: Union[int, list[int]] = None,
                            connections_to_avoid: dict = None
                            ) -> dict[int, list[ConnectionInfo]]:
        if connections_to_avoid is None:
            connections_to_avoid: dict = {}
        segment_graph = {}
        # makes sure that segments_to_avoid is list
        if segments_to_avoid and isinstance(segments_to_avoid, int):
            segments_to_avoid = [segments_to_avoid]
        for from_segment_id in self.segmented_map.regions:
            if segments_to_avoid and from_segment_id in segments_to_avoid:
                continue
            avoid_connection = None
            if from_segment_id in connections_to_avoid.keys():
                avoid_connection = connections_to_avoid[from_segment_id]
            segment_graph[from_segment_id] = self.get_connections(segment_id=from_segment_id,
                                                                  passages_to_avoid=passages_to_avoid,
                                                                  avoid_connection=avoid_connection)
        return segment_graph

    def get_connections(self,
                        segment_id: int,
                        passages_to_avoid=None,  # e.g., Cliff or (Cliff, Ramp)
                        avoid_connection: int | None = None,
                        ) -> list[ConnectionInfo]:
        """
        Returns a list of connections from the specified segment to neighboring segments.

        Each connection includes the position of the passage, the ID of the connected segment,
        and the center of the connected segment. Certain passage types can be excluded.

        :param segment_id: The ID of the segment to find connections from.
        :param passages_to_avoid: Passage types to exclude (e.g. Cliff or tuple of types). None disables this filter.
        :param avoid_connection: Prevent connection to a specific segment ID (directional).
        :return: A list of ConnectionInfo objects representing valid connections.
        """
        segment = self._segment_by_id(segment_id)
        passages = segment.passages
        connections = []

        for passage in passages:
            if passages_to_avoid and isinstance(passage, passages_to_avoid):
                continue
            if not passage.passable:
                continue

            for connection in passage.connections:
                if connection == segment_id or (avoid_connection and connection == avoid_connection):
                    continue

                connected_segment = self._segment_by_id(connection)
                connections.append(ConnectionInfo(
                    id=connection,
                    passage_center=passage.center,
                    segment_center=connected_segment.center
                ))

        return connections

    def _segment_by_id(self, segment_id: int):
        """
        Get a segment by its ID.
        :param segment_id: ID of the segment
        :return: Region object
        """
        if self.segmented_map is None:
            raise ValueError("MapData is not initialized.")
        return self.segmented_map.regions[segment_id]

    def closest_segment_id(self, position: Point2 | Point3 | Unit) -> int:
        """
        Get the segment ID of a unit or position.
        """
        if position is None:
            raise ValueError("Position cannot be None.")
        if isinstance(position, Unit):
            position = position.position
        if isinstance(position, Point3):
            position = Point2((position.x, position.y))
        if self.segmented_map is None:
            raise ValueError("MapData is not initialized.")
        position = position.rounded
        segment = self._closest_segment(position=position)
        return segment.id

    def _closest_segment(self, position: Point2 | Point3 | Unit) -> Region:
        """
        Finds the closest region to the given point.

        Args:
            position: Point2 | Point3 | Unit: The point to find the closest region for.

        Returns:
            Region: The closest region to the given point.
        """
        if position is None:
            raise ValueError("Position cannot be None.")
        if isinstance(position, Unit):
            position = position.position
        if isinstance(position, Point3):
            position = Point2((position.x, position.y))
        if region := self.segmented_map.region_at(position):
            return region

        directions = (
            Point2((1, 0)),  # East
            Point2((1, 1)),  # Northeast
            Point2((0, 1)),  # North
            Point2((-1, 1)),  # Northwest
            Point2((-1, 0)),  # West
            Point2((-1, -1)),  # Southwest
            Point2((0, -1)),  # South
            Point2((1, -1)),  # Southeast
        )
        for distance in range(1, 10):
            for direction in directions:
                neighbor = position + direction * distance

                if region := self.segmented_map.region_at(neighbor):
                    return region

    def get_random_segment_id(self) -> int:
        """
        Returns a random segment ID from the segmented map.
        """
        # Step 1: Check if the segmented map is initialized
        if self.segmented_map is None:
            raise ValueError("MapData is not initialized.")

        # Step 2: Get the dictionary of regions
        regions_dict = self.segmented_map.regions

        # Step 3: Extract the list of segment IDs (the keys)
        segment_ids = list(regions_dict.keys())

        # Step 4: Randomly choose one segment ID
        random_segment_id = int(np.random.choice(segment_ids))

        # Step 5: Return the chosen segment ID
        return random_segment_id
