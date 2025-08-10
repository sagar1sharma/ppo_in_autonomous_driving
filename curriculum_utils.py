def get_curriculum_stage(stage):
    """
    Return environment config dict for given curriculum stage.
    
    Stages increase difficulty by increasing traffic density and vehicle behavior complexity.
    """
    if stage == 0:
        return {
            "vehicles_density": 0.1,               # very sparse traffic
            "other_vehicles_type": "obstacle",    # static obstacles, easy to avoid
        }
    elif stage == 1:
        return {
            "vehicles_density": 0.2,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",  # simple driver model
        }
    elif stage == 2:
        return {
            "vehicles_density": 0.3,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        }
    elif stage == 3:
        return {
            "vehicles_density": 0.4,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        }
    else:
        # Final hard stage
        return {
            "vehicles_density": 0.5,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        }
