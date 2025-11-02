#!/usr/bin/env python3
"""
Traffic Analytics Engine
Core module for traffic analysis, counting, and monitoring
"""

import numpy as np
import cv2
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import sqlite3
import json
from pathlib import Path


@dataclass
class Vehicle:
    """Represents a detected vehicle"""
    id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    timestamp: datetime
    center: Tuple[float, float] = field(init=False)
    
    def __post_init__(self):
        """Calculate vehicle center point"""
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class TrafficZone:
    """Defines a monitoring zone in the image"""
    name: str
    polygon: List[Tuple[int, int]]  # Zone boundary points
    vehicle_count: int = 0
    vehicles: List[Vehicle] = field(default_factory=list)
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside the zone"""
        x, y = point
        n = len(self.polygon)
        inside = False
        
        p1x, p1y = self.polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = self.polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside


class TrafficAnalytics:
    """Core traffic analysis and monitoring system"""
    
    def __init__(self, db_path: str = "traffic_monitoring_system/data/traffic_data.db"):
        """Initialize the traffic analytics system"""
        self.db_path = db_path
        self.zones: List[TrafficZone] = []
        self.vehicle_history: deque = deque(maxlen=1000)  # Keep last 1000 detections
        self.traffic_flow_data = defaultdict(list)
        self.congestion_threshold = 10  # vehicles per zone for congestion
        
        # Initialize database
        self._init_database()
        
        print("🚦 Traffic Analytics Engine Initialized")
    
    def _init_database(self):
        """Initialize SQLite database for storing traffic data"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Traffic events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS traffic_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    zone_name TEXT,
                    vehicle_count INTEGER,
                    congestion_level TEXT,
                    average_confidence REAL
                )
            ''')
            
            # Vehicle detections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vehicle_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    zone_name TEXT,
                    bbox TEXT,
                    confidence REAL,
                    vehicle_center TEXT
                )
            ''')
            
            # Hourly statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hourly_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date_hour TEXT UNIQUE,
                    total_vehicles INTEGER DEFAULT 0,
                    peak_congestion INTEGER DEFAULT 0,
                    average_flow REAL DEFAULT 0.0
                )
            ''')
            
            conn.commit()
    
    def add_zone(self, name: str, polygon: List[Tuple[int, int]]):
        """Add a new monitoring zone"""
        zone = TrafficZone(name=name, polygon=polygon)
        self.zones.append(zone)
        print(f"✅ Added monitoring zone: {name}")
    
    def process_detections(self, detections: List[Dict], timestamp: Optional[datetime] = None) -> Dict:
        """
        Process vehicle detections and update analytics
        
        Args:
            detections: List of detection results from RF-DETR
            timestamp: Detection timestamp (defaults to now)
            
        Returns:
            Dict: Analysis results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Reset zone counts
        for zone in self.zones:
            zone.vehicle_count = 0
            zone.vehicles = []
        
        # Process each detection
        vehicles = []
        for i, detection in enumerate(detections):
            vehicle = Vehicle(
                id=i,
                bbox=detection['bbox'],
                confidence=detection['confidence'],
                timestamp=timestamp
            )
            vehicles.append(vehicle)
            
            # Check which zones contain this vehicle
            for zone in self.zones:
                if zone.contains_point(vehicle.center):
                    zone.vehicle_count += 1
                    zone.vehicles.append(vehicle)
        
        # Store in history
        self.vehicle_history.append({
            'timestamp': timestamp,
            'vehicles': vehicles,
            'total_count': len(vehicles)
        })
        
        # Calculate analytics
        results = self._calculate_analytics(timestamp)
        
        # Store in database
        self._store_analytics(results, timestamp)
        
        return results
    
    def _calculate_analytics(self, timestamp: datetime) -> Dict:
        """Calculate traffic analytics"""
        total_vehicles = sum(zone.vehicle_count for zone in self.zones)
        
        # Congestion analysis
        congested_zones = []
        for zone in self.zones:
            if zone.vehicle_count > self.congestion_threshold:
                congested_zones.append({
                    'zone': zone.name,
                    'count': zone.vehicle_count,
                    'level': 'HIGH' if zone.vehicle_count > self.congestion_threshold * 1.5 else 'MEDIUM'
                })
        
        # Traffic flow calculation (vehicles per minute over last 5 minutes)
        recent_history = [
            entry for entry in self.vehicle_history 
            if entry['timestamp'] > timestamp - timedelta(minutes=5)
        ]
        
        flow_rate = len(recent_history) / max(1, len(recent_history)) * 60  # per minute
        
        # Peak detection
        counts = [entry['total_count'] for entry in recent_history]
        peak_traffic = max(counts) if counts else 0
        
        results = {
            'timestamp': timestamp.isoformat(),
            'total_vehicles': total_vehicles,
            'zone_counts': {zone.name: zone.vehicle_count for zone in self.zones},
            'congestion': {
                'detected': len(congested_zones) > 0,
                'zones': congested_zones,
                'overall_level': self._get_overall_congestion_level(congested_zones)
            },
            'flow_rate': round(flow_rate, 2),
            'peak_traffic': peak_traffic,
            'average_confidence': np.mean([v.confidence for zone in self.zones for v in zone.vehicles]) if any(zone.vehicles for zone in self.zones) else 0.0
        }
        
        return results
    
    def _get_overall_congestion_level(self, congested_zones: List[Dict]) -> str:
        """Determine overall congestion level"""
        if not congested_zones:
            return 'LOW'
        
        high_count = sum(1 for zone in congested_zones if zone['level'] == 'HIGH')
        if high_count > 0:
            return 'HIGH'
        elif len(congested_zones) > 1:
            return 'MEDIUM'
        else:
            return 'MEDIUM'
    
    def _store_analytics(self, results: Dict, timestamp: datetime):
        """Store analytics results in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Store traffic event
                cursor.execute('''
                    INSERT INTO traffic_events 
                    (timestamp, zone_name, vehicle_count, congestion_level, average_confidence)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    timestamp.isoformat(),
                    'ALL_ZONES',
                    results['total_vehicles'],
                    results['congestion']['overall_level'],
                    results['average_confidence']
                ))
                
                # Store individual zone data
                for zone_name, count in results['zone_counts'].items():
                    cursor.execute('''
                        INSERT INTO traffic_events 
                        (timestamp, zone_name, vehicle_count, congestion_level, average_confidence)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        timestamp.isoformat(),
                        zone_name,
                        count,
                        'HIGH' if count > self.congestion_threshold else 'LOW',
                        results['average_confidence']
                    ))
                
                # Update hourly statistics
                hour_key = timestamp.strftime('%Y-%m-%d %H:00:00')
                cursor.execute('''
                    INSERT OR REPLACE INTO hourly_stats 
                    (date_hour, total_vehicles, peak_congestion, average_flow)
                    VALUES (?, ?, ?, ?)
                ''', (
                    hour_key,
                    results['total_vehicles'],
                    results['peak_traffic'],
                    results['flow_rate']
                ))
                
                conn.commit()
                
        except Exception as e:
            print(f"❌ Error storing analytics: {e}")
    
    def get_traffic_summary(self, hours: int = 24) -> Dict:
        """Get traffic summary for the last N hours"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent traffic data
                since = datetime.now() - timedelta(hours=hours)
                cursor.execute('''
                    SELECT * FROM traffic_events 
                    WHERE timestamp > ? AND zone_name = 'ALL_ZONES'
                    ORDER BY timestamp DESC
                ''', (since.isoformat(),))
                
                events = cursor.fetchall()
                
                if not events:
                    return {'error': 'No data available'}
                
                # Calculate summary statistics
                total_events = len(events)
                avg_vehicles = np.mean([event[3] for event in events])  # vehicle_count
                congestion_events = sum(1 for event in events if event[4] in ['MEDIUM', 'HIGH'])
                
                summary = {
                    'period_hours': hours,
                    'total_measurements': total_events,
                    'average_vehicles': round(avg_vehicles, 1),
                    'congestion_events': congestion_events,
                    'congestion_percentage': round(congestion_events / total_events * 100, 1) if total_events > 0 else 0,
                    'last_update': events[0][1] if events else None  # timestamp
                }
                
                return summary
                
        except Exception as e:
            print(f"❌ Error getting traffic summary: {e}")
            return {'error': str(e)}
    
    def get_zone_analytics(self) -> List[Dict]:
        """Get current analytics for all zones"""
        zone_data = []
        
        for zone in self.zones:
            zone_info = {
                'name': zone.name,
                'current_count': zone.vehicle_count,
                'congestion_level': 'HIGH' if zone.vehicle_count > self.congestion_threshold else 'LOW',
                'vehicles': [
                    {
                        'bbox': v.bbox,
                        'confidence': round(v.confidence, 3),
                        'center': v.center
                    } for v in zone.vehicles
                ]
            }
            zone_data.append(zone_info)
        
        return zone_data
    
    def export_data(self, start_date: str, end_date: str, format: str = 'json') -> str:
        """Export traffic data for a date range"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM traffic_events 
                    WHERE DATE(timestamp) BETWEEN ? AND ?
                    ORDER BY timestamp
                ''', (start_date, end_date))
                
                events = cursor.fetchall()
                
                if format == 'json':
                    data = []
                    for event in events:
                        data.append({
                            'id': event[0],
                            'timestamp': event[1],
                            'zone_name': event[2],
                            'vehicle_count': event[3],
                            'congestion_level': event[4],
                            'average_confidence': event[5]
                        })
                    
                    return json.dumps(data, indent=2)
                
                # CSV format
                elif format == 'csv':
                    import csv
                    import io
                    
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(['ID', 'Timestamp', 'Zone', 'Vehicle Count', 'Congestion Level', 'Avg Confidence'])
                    writer.writerows(events)
                    
                    return output.getvalue()
                
        except Exception as e:
            return f"Error exporting data: {e}"


def main():
    """Test the traffic analytics system"""
    print("🚦 Testing Traffic Analytics Engine")
    print("=" * 40)
    
    # Initialize analytics
    analytics = TrafficAnalytics()
    
    # Add sample monitoring zones
    analytics.add_zone("Main Intersection", [(100, 100), (300, 100), (300, 300), (100, 300)])
    analytics.add_zone("Highway Entry", [(400, 150), (600, 150), (600, 250), (400, 250)])
    
    # Simulate some detections
    sample_detections = [
        {'bbox': [150, 150, 200, 200], 'confidence': 0.95, 'class_name': 'vehicle'},
        {'bbox': [250, 180, 290, 220], 'confidence': 0.87, 'class_name': 'vehicle'},
        {'bbox': [450, 180, 490, 220], 'confidence': 0.92, 'class_name': 'vehicle'},
    ]
    
    # Process detections
    results = analytics.process_detections(sample_detections)
    
    print("\\n📊 Analysis Results:")
    print(f"Total Vehicles: {results['total_vehicles']}")
    print(f"Zone Counts: {results['zone_counts']}")
    print(f"Congestion Detected: {results['congestion']['detected']}")
    print(f"Flow Rate: {results['flow_rate']} vehicles/min")
    
    # Get summary
    summary = analytics.get_traffic_summary(1)
    print(f"\\n📈 Traffic Summary: {summary}")
    
    print("\\n✅ Traffic Analytics Engine Ready!")


if __name__ == "__main__":
    main()