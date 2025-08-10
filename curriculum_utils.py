def get_curriculum_stage(stage):
    densities = [0.1, 0.2, 0.3, 0.4, 0.5]
    density = densities[stage] if stage < len(densities) else densities[-1]
    return {
        "vehicles_density": density,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "speed_limit": [5, 10],  # Reduce speed range for other vehicles
        "controlled_vehicle": {
            "type": "highway_env.vehicle.behavior.IDMVehicle",
            "initial_speed": 10  # Increase agent's initial speed
        }
    }