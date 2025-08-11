def get_curriculum_stage(stage):
    densities = [0.1, 0.2, 0.3, 0.4, 0.5]
    speed_limits = [[5, 10], [10, 15], [15, 20], [20, 25], [25, 30]]
    lanes_count = [2, 3, 4, 5, 6]
    lane = lanes_count[stage] if stage < len(lanes_count) else lanes_count[-1]
    density = densities[stage] if stage < len(densities) else densities[-1]
    speed_limit = speed_limits[stage] if stage < len(speed_limits) else speed_limits[-1]

    return {
        "vehicles_density": density,
        "lanes_count": lane,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "speed_limit": speed_limit,
        "controlled_vehicle": {
            "type": "highway_env.vehicle.behavior.IDMVehicle",
            "initial_speed": 10  # Increase agent's initial speed
        }
    }