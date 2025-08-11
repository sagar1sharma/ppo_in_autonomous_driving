def get_curriculum_stage(stage):
    densities = [0.1, 0.2, 0.3, 0.4, 0.5]
    behaviors = ["highway_env.vehicle.behavior.IDMVehicle", "highway_env.vehicle.behavior.AggressiveVehicle"]
    speed_limits = [[5, 10], [10, 15], [15, 20], [20, 25], [25, 30]]

    density = densities[stage] if stage < len(densities) else densities[-1]
    behavior = behaviors[stage % len(behaviors)]  # Alternate behaviors
    speed_limit = speed_limits[stage] if stage < len(speed_limits) else speed_limits[-1]

    return {
        "vehicles_density": density,
        "other_vehicles_type": behavior,
        "speed_limit": speed_limit,
        "controlled_vehicle": {
            "type": "highway_env.vehicle.behavior.IDMVehicle",
            "initial_speed": 10  # Increase agent's initial speed
        }
    }