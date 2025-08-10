def get_curriculum_stage(stage):
    densities = [0.1, 0.2, 0.3, 0.4, 0.5]
    density = densities[stage] if stage < len(densities) else densities[-1]
    return {
        "vehicles_density": density,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    }