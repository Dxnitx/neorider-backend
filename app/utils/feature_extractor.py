from typing import List, Dict
import statistics

def extract_features(sensor_window: List[Dict]) -> Dict[str, float]:
    """Extract statistical features from a sensor window.
    
    Computes mean, standard deviation, and max absolute value for each of the 9 sensor channels.
    Returns a flat dictionary with 27 features total.
    """
    channels = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'pitch', 'roll', 'yaw']
    features = {}
    
    for channel in channels:
        values = [reading[channel] for reading in sensor_window]
        
        # Mean
        features[f'{channel}_mean'] = statistics.mean(values)
        
        # Standard deviation
        features[f'{channel}_std'] = statistics.stdev(values)
        
        # Max absolute value
        features[f'{channel}_max'] = max(abs(v) for v in values)
    
    return features