"""
FLIPS Mobility and Network simulation.
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class Vehicle:
    """
    Phase 3: Vehicle with mobility and Kalman Filter prediction.
    """
    def __init__(self, vehicle_id, config):
        self.vehicle_id = vehicle_id
        self.config = config
        
        # Initial position (Manhattan grid-like)
        # Random x, y within [0, grid_size]
        grid_size = config.get('manhattan_grid_size', 1000)
        self.x = np.random.uniform(0, grid_size)
        self.y = np.random.uniform(0, grid_size)
        
        # Velocity (m/s)
        speed_mean = config.get('vehicle_speed_mean', 15)
        speed_std = config.get('vehicle_speed_std', 5)
        speed = max(0.1, np.random.normal(speed_mean, speed_std))
        angle = np.random.uniform(0, 2*np.pi)
        self.vx = speed * np.cos(angle)
        self.vy = speed * np.sin(angle)
        
        # Kalman Filter
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # State transition: x' = x + v*dt
        dt = 1.0 # time step
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # Measurement matrix: we measure x, y
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        # Covariance matrices
        self.kf.P *= 10.
        self.kf.R *= 5.  # Measurement noise
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1, block_size=2)
        
        # Initial state
        self.kf.x = np.array([self.x, self.y, self.vx, self.vy])

    def update_position(self, dt=1.0):
        """Update position and Kalman Filter."""
        # Update real physics
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Boundary checks (bounce)
        grid_size = self.config.get('manhattan_grid_size', 1000)
        if self.x < 0 or self.x > grid_size: self.vx *= -1
        if self.y < 0 or self.y > grid_size: self.vy *= -1
        self.x = np.clip(self.x, 0, grid_size)
        self.y = np.clip(self.y, 0, grid_size)
        
        # KF Predict
        self.kf.predict()
        
        # KF Update (simulate GPS measurement with noise)
        noise = np.random.normal(0, 1.0, 2)
        measurement = np.array([self.x + noise[0], self.y + noise[1]])
        self.kf.update(measurement)

    def estimate_contact_time(self, base_station):
        """Estimate connection time with BS based on KF prediction."""
        # Current predicted state
        px, py, vx, vy = self.kf.x
        
        # Coverage radius
        R = base_station.coverage_radius
        bx, by = base_station.position
        
        # Vector to BS
        dx = px - bx
        dy = py - by
        dist_sq = dx**2 + dy**2
        
        if dist_sq > R**2:
            return 0.0 # Not connected
            
        # Solve when dist = R
        # (px + vx*t - bx)^2 + (py + vy*t - by)^2 = R^2
        # A*t^2 + B*t + C = 0
        
        # A = vx^2 + vy^2
        # B = 2*dx*vx + 2*dy*vy
        # C = dx^2 + dy^2 - R^2
        
        A = vx**2 + vy**2
        if A < 1e-6: return 100.0 # Stationary inside coverage
        
        B = 2 * (dx * vx + dy * vy)
        C = dist_sq - R**2
        
        delta = B**2 - 4*A*C
        
        if delta < 0:
            return 100.0 # Should not happen if inside
            
        t1 = (-B - np.sqrt(delta)) / (2*A)
        t2 = (-B + np.sqrt(delta)) / (2*A)
        
        # We want positive time exiting
        time_exit = max(t1, t2)
        
        if time_exit < 0: return 0.0
        return min(time_exit, 100.0) # Cap at 100s

class BaseStation:
    """Phase 3: Base Station with RSSI and Bandwidth simulation."""
    def __init__(self, position, config):
        self.position = position
        self.coverage_radius = config.get('coverage_radius', 500)
        self.tx_power = config.get('tx_power', 30) # dBm
        
    def compute_rssi(self, vehicle):
        dist = np.sqrt((self.position[0]-vehicle.x)**2 + (self.position[1]-vehicle.y)**2)
        if dist < 1.0: dist = 1.0
        
        # Path loss model (simplified)
        # PL = 20log10(d) + shadowing
        path_loss = 20 * np.log10(dist) + np.random.normal(0, 2)
        rssi = self.tx_power - path_loss
        return rssi

    def estimate_bandwidth(self, rssi):
        # mmWave model (simplified)
        # BW propto SNR
        # Max 100 Mbps
        if rssi < -90: return 0.1 # Very low
        
        bw_max = 100.0 # Mbps
        rssi_min = -90
        rssi_max = -50
        
        factor = (rssi - rssi_min) / (rssi_max - rssi_min)
        factor = np.clip(factor, 0.05, 1.0)
        
        return bw_max * factor
