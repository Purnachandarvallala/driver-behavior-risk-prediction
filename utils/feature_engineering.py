import numpy as np

def create_features(accx, accy, accz, gyrox, gyroy, gyroz):

    motion_magnitude = np.sqrt(accx**2 + accy**2 + accz**2)

    rotation_magnitude = np.sqrt(gyrox**2 + gyroy**2 + gyroz**2)

    driving_intensity = motion_magnitude + rotation_magnitude

    harsh_braking = 1 if accx < -1 else 0

    sharp_turning = 1 if abs(gyroz) > 1 else 0

    return [
        accx, accy, accz,
        gyrox, gyroy, gyroz,
        motion_magnitude,
        rotation_magnitude,
        driving_intensity,
        harsh_braking,
        sharp_turning
    ]