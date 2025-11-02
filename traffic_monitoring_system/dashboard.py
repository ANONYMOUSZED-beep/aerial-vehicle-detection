#!/usr/bin/env python3
"""
Smart Traffic Monitoring Dashboard
Web-based interface for real-time traffic monitoring and analytics
"""

import dash
from dash import dcc, html, Input, Output, callback, dash_table, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import json
import threading
import time
from pathlib import Path
import sys
import os
import base64
import numpy as np

# Add parent directory to import our modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from traffic_monitoring_system.realtime_processor import RealTimeTrafficProcessor
from traffic_monitoring_system.traffic_analytics import TrafficAnalytics

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Smart Traffic Monitoring Dashboard"

# Global variables
traffic_processor = None
analytics = None
is_live_monitoring = False
live_data_store = {'latest_results': None, 'update_count': 0}

# Video processing progress tracking
video_processing_state = {
    'is_processing': False,
    'progress': 0,
    'status': 'Ready',
    'current_frame': 0,
    'total_frames': 0,
    'results': None
}

def initialize_system():
    """Initialize the traffic monitoring system"""
    global traffic_processor, analytics
    
    try:
        print("🚦 Initializing Traffic Monitoring System...")
        
        # Initialize processor
        traffic_processor = RealTimeTrafficProcessor()
        analytics = TrafficAnalytics()
        
        # Load RF-DETR model
        if not traffic_processor.load_model():
            print("❌ Failed to load RF-DETR model")
            return False
        
        # Setup default monitoring zones
        default_zones = {
            "Main_Intersection": [(150, 150), (450, 150), (450, 350), (150, 350)],
            "Highway_Entry": [(500, 200), (750, 200), (750, 300), (500, 300)],
            "Parking_Area": [(100, 400), (300, 400), (300, 500), (100, 500)]
        }
        traffic_processor.setup_monitoring_zones(default_zones)
        
        # Set callback for live updates
        traffic_processor.set_results_callback(update_live_data)
        
        print("✅ Traffic Monitoring System Ready!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return False

def update_live_data(results):
    """Callback function to update live data"""
    global live_data_store
    live_data_store['latest_results'] = results
    live_data_store['update_count'] += 1

def process_video_with_progress(video_path):
    """Process video with progress updates"""
    global video_processing_state, traffic_processor
    
    try:
        import cv2
        
        # Clear any previous results
        global live_data_store
        live_data_store['latest_results'] = None
        
        video_processing_state['is_processing'] = True
        video_processing_state['progress'] = 0
        video_processing_state['status'] = 'Analyzing video...'
        video_processing_state['results'] = None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            video_processing_state['status'] = 'Error: Could not open video'
            video_processing_state['is_processing'] = False
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        video_processing_state['total_frames'] = total_frames
        video_processing_state['status'] = f'Processing {total_frames} frames ({duration:.1f}s video)...'
        
        # Process every N frames based on processing interval
        frame_skip = max(1, int(fps * traffic_processor.processing_interval)) if fps > 0 else 1
        frames_to_process = list(range(0, total_frames, frame_skip))
        
        # Initialize results
        results_summary = {
            'video_path': video_path,
            'total_frames': total_frames,
            'frames_processed': 0,
            'total_detections': 0,
            'peak_traffic': 0,
            'congestion_events': 0,
            'frame_results': [],
            'analytics': {
                'total_vehicles': 0,
                'congestion': {'detected': False, 'overall_level': 'Low'}
            }
        }
        
        temp_dir = Path("traffic_monitoring_system/data/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        for i, frame_num in enumerate(frames_to_process):
            if not video_processing_state['is_processing']:  # Check if cancelled
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            try:
                # Update progress
                progress = (i / len(frames_to_process)) * 100
                video_processing_state['progress'] = progress
                video_processing_state['current_frame'] = frame_num
                video_processing_state['status'] = f'Processing frame {frame_num}/{total_frames} ({progress:.1f}%)'
                
                # Save frame temporarily
                temp_frame_path = temp_dir / f"temp_frame_{frame_num}.jpg"
                cv2.imwrite(str(temp_frame_path), frame)
                
                # Process frame
                frame_results = traffic_processor.process_single_image(str(temp_frame_path))
                
                if 'error' not in frame_results:
                    results_summary['frames_processed'] += 1
                    results_summary['total_detections'] += len(frame_results['detections'])
                    results_summary['peak_traffic'] = max(
                        results_summary['peak_traffic'], 
                        frame_results['analytics']['total_vehicles']
                    )
                    
                    if frame_results['analytics']['congestion']['detected']:
                        results_summary['congestion_events'] += 1
                    
                    # Update live data with latest frame results
                    live_data_store['latest_results'] = frame_results
                    live_data_store['update_count'] += 1
                
                # Cleanup temp frame
                if temp_frame_path.exists():
                    temp_frame_path.unlink()
                
                # Small delay to allow UI updates
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")
        
        cap.release()
        
        # Finalize results
        results_summary['analytics']['total_vehicles'] = results_summary['peak_traffic']
        results_summary['analytics']['congestion']['detected'] = results_summary['congestion_events'] > 0
        
        video_processing_state['results'] = results_summary
        video_processing_state['progress'] = 100
        video_processing_state['status'] = f'Complete! Processed {results_summary["frames_processed"]} frames, found {results_summary["total_detections"]} vehicles'
        video_processing_state['is_processing'] = False
        
    except Exception as e:
        video_processing_state['status'] = f'Error: {str(e)}'
        video_processing_state['is_processing'] = False

def get_traffic_data(hours=24):
    """Get traffic data from database"""
    try:
        db_path = "traffic_monitoring_system/data/traffic_data.db"
        if not Path(db_path).exists():
            return pd.DataFrame()
        
        with sqlite3.connect(db_path) as conn:
            since = datetime.now() - timedelta(hours=hours)
            query = """
                SELECT timestamp, zone_name, vehicle_count, congestion_level, average_confidence
                FROM traffic_events 
                WHERE timestamp > ? AND zone_name != 'ALL_ZONES'
                ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, conn, params=(since.isoformat(),))
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    except Exception as e:
        print(f"Error getting traffic data: {e}")
        return pd.DataFrame()

# Dashboard Layout
def create_layout():
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🚦 Smart Traffic Monitoring Dashboard", className="text-center mb-4"),
                html.Hr()
            ])
        ]),
        
        # Control Panel
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🎛️ Control Panel"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("📸 Process Image", id="process-image-btn", color="primary", className="me-2"),
                                dbc.Button("🎬 Process Video", id="process-video-btn", color="secondary", className="me-2"),
                                dbc.Button("📹 Start Live", id="start-live-btn", color="success", className="me-2"),
                                dbc.Button("🛑 Stop Live", id="stop-live-btn", color="danger", disabled=True)
                            ], width=8),
                            dbc.Col([
                                html.Div(id="system-status", children="🟢 System Ready")
                            ], width=4)
                        ]),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col([
                                html.H6("📸 Image Upload", className="mb-2"),
                                dcc.Upload(
                                    id='upload-image',
                                    children=html.Div([
                                        '📁 Drag & Drop or ',
                                        html.A('Select Image', style={'color': '#007bff'})
                                    ]),
                                    style={
                                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                        'borderWidth': '2px', 'borderStyle': 'dashed',
                                        'borderRadius': '10px', 'textAlign': 'center', 
                                        'margin': '5px', 'borderColor': '#007bff',
                                        'backgroundColor': '#f8f9fa'
                                    },
                                    multiple=False,
                                    accept='image/*'
                                ),
                                html.Div(id='upload-image-status', style={'margin': '5px', 'textAlign': 'center', 'fontSize': '12px'})
                            ], width=4),
                            dbc.Col([
                                html.H6("🎬 Video Upload", className="mb-2"),
                                dcc.Upload(
                                    id='upload-video',
                                    children=html.Div([
                                        '📁 Drag & Drop or ',
                                        html.A('Select Video', style={'color': '#28a745'})
                                    ]),
                                    style={
                                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                        'borderWidth': '2px', 'borderStyle': 'dashed',
                                        'borderRadius': '10px', 'textAlign': 'center', 
                                        'margin': '5px', 'borderColor': '#28a745',
                                        'backgroundColor': '#f8f9fa'
                                    },
                                    multiple=False,
                                    accept='video/*'
                                ),
                                html.Div(id='upload-video-status', style={'margin': '5px', 'textAlign': 'center', 'fontSize': '12px'}),
                                # Progress bar for video processing
                                html.Div(id='video-progress-container', children=[
                                    dbc.Progress(id='video-progress-bar', value=0, style={'height': '20px', 'margin': '10px 0'}, color="success"),
                                    html.Div(id='video-progress-text', style={'textAlign': 'center', 'fontSize': '10px', 'color': '#666'})
                                ], style={'display': 'none'})
                            ], width=4),
                            dbc.Col([
                                html.H6("⚙️ Settings", className="mb-2"),
                                dcc.Dropdown(
                                    id='confidence-dropdown',
                                    options=[
                                        {'label': 'Low (0.1)', 'value': 0.1},
                                        {'label': 'Medium (0.15)', 'value': 0.15},
                                        {'label': 'High (0.25)', 'value': 0.25},
                                        {'label': 'Very High (0.4)', 'value': 0.4}
                                    ],
                                    value=0.15,
                                    placeholder="Select Confidence Threshold"
                                ),
                                html.Div([
                                    dbc.Button("🧪 Process Sample", id="process-sample-btn", 
                                              color="info", size="sm", className="mt-2")
                                ])
                            ], width=4)
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Real-time Stats Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("🚗 Total Vehicles", className="card-title"),
                        html.H2(id="total-vehicles", children="0", className="text-primary")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("🚨 Congestion Level", className="card-title"),
                        html.H2(id="congestion-level", children="LOW", className="text-success")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("⏱️ Processing Time", className="card-title"),
                        html.H2(id="processing-time", children="0.0s", className="text-info")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("🎯 Detection Rate", className="card-title"),
                        html.H2(id="detection-rate", children="0/min", className="text-warning")
                    ])
                ])
            ], width=3)
        ], className="mb-4"),
        
        # Charts Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 Vehicle Count Over Time"),
                    dbc.CardBody([
                        dcc.Graph(id="traffic-timeline-chart")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🎯 Zone Analysis"),
                    dbc.CardBody([
                        dcc.Graph(id="zone-analysis-chart")
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        # Detection Results and Image Display
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🔍 Latest Detection Results"),
                    dbc.CardBody([
                        html.Div(id="detection-results"),
                        html.Hr(),
                        html.Div(id="processed-image-display")
                    ])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📈 Zone Statistics"),
                    dbc.CardBody([
                        html.Div(id="zone-stats")
                    ])
                ])
            ], width=4)
        ], className="mb-4"),
        
        # Data Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📋 Traffic Events Log"),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id="traffic-table",
                            columns=[
                                {"name": "Timestamp", "id": "timestamp"},
                                {"name": "Zone", "id": "zone_name"},
                                {"name": "Vehicles", "id": "vehicle_count"},
                                {"name": "Congestion", "id": "congestion_level"},
                                {"name": "Confidence", "id": "average_confidence", "type": "numeric", "format": {"specifier": ".2f"}}
                            ],
                            page_size=10,
                            sort_action="native",
                            style_cell={'textAlign': 'left'},
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{congestion_level} = HIGH'},
                                    'backgroundColor': '#ffebee',
                                    'color': 'black',
                                },
                                {
                                    'if': {'filter_query': '{congestion_level} = MEDIUM'},
                                    'backgroundColor': '#fff3e0',
                                    'color': 'black',
                                }
                            ]
                        )
                    ])
                ])
            ])
        ]),
        
        # Auto-refresh component
        dcc.Interval(
            id='interval-component',
            interval=2000,  # Update every 2 seconds
            n_intervals=0
        ),
        
        # Store for live data
        dcc.Store(id='live-data-store')
        
    ], fluid=True)

app.layout = create_layout()

# Callbacks

# Progress bar callback for video processing
@app.callback(
    [Output('video-progress-container', 'style'),
     Output('video-progress-bar', 'value'),
     Output('video-progress-text', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_video_progress(n_intervals):
    """Update video processing progress bar"""
    global video_processing_state
    
    if video_processing_state['is_processing']:
        return (
            {'display': 'block'},
            video_processing_state['progress'],
            video_processing_state['status']
        )
    else:
        return ({'display': 'none'}, 0, '')

@app.callback(
    [Output('total-vehicles', 'children'),
     Output('congestion-level', 'children'),
     Output('congestion-level', 'className'),
     Output('processing-time', 'children'),
     Output('detection-rate', 'children'),
     Output('traffic-timeline-chart', 'figure'),
     Output('zone-analysis-chart', 'figure'),
     Output('detection-results', 'children'),
     Output('zone-stats', 'children'),
     Output('traffic-table', 'data'),
     Output('processed-image-display', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    """Update dashboard with latest data"""
    global live_data_store, traffic_processor, video_processing_state
    
    # Get latest results (prioritize video results if available)
    latest_results = live_data_store.get('latest_results')
    
    # Check if video processing is complete and has results
    if (not video_processing_state['is_processing'] and 
        video_processing_state['results'] is not None):
        # Use video results for enhanced dashboard display
        video_results = video_processing_state['results']
        if video_results:
            # Ensure analytics data is properly set
            analytics_data = video_results.get('analytics', {})
            analytics_data['total_vehicles'] = video_results.get('peak_traffic', 0)
            analytics_data['flow_rate'] = video_results.get('total_detections', 0)
            
            latest_results = {
                'detections': [{'confidence': 0.9}] * video_results.get('total_detections', 0),
                'analytics': analytics_data,
                'processing_time': 0.1,  # Video processing time per frame
                'image_path': 'video_processing'
            }
            # Clear the results after showing
            video_processing_state['results'] = None
    
    # Default values
    total_vehicles = "0"
    congestion_level = "LOW"
    congestion_class = "text-success"
    processing_time = "0.0s"
    detection_rate = "0/min"
    
    # Update if we have results
    if latest_results and 'analytics' in latest_results:
        analytics_data = latest_results['analytics']
        total_vehicles = str(analytics_data.get('total_vehicles', 0))
        congestion_level = analytics_data.get('congestion', {}).get('overall_level', 'LOW')
        processing_time = f"{latest_results.get('processing_time', 0):.2f}s"
        
        # Debug info (remove in production)
        print(f"DEBUG: Analytics data: {analytics_data}")
        print(f"DEBUG: Total vehicles: {total_vehicles}")
        
        # Set congestion level styling
        if congestion_level == 'HIGH':
            congestion_class = "text-danger"
        elif congestion_level == 'MEDIUM':
            congestion_class = "text-warning"
        else:
            congestion_class = "text-success"
        
        detection_rate = f"{analytics_data.get('flow_rate', 0)}/min"
    
    # Get historical data for charts
    df = get_traffic_data(hours=6)
    
    # Timeline chart
    timeline_fig = go.Figure()
    if not df.empty:
        for zone in df['zone_name'].unique():
            zone_data = df[df['zone_name'] == zone]
            timeline_fig.add_trace(go.Scatter(
                x=zone_data['timestamp'],
                y=zone_data['vehicle_count'],
                mode='lines+markers',
                name=zone,
                line=dict(width=2)
            ))
    
    timeline_fig.update_layout(
        title="Vehicle Count Timeline (Last 6 Hours)",
        xaxis_title="Time",
        yaxis_title="Vehicle Count",
        height=300
    )
    
    # Zone analysis chart
    zone_fig = go.Figure()
    if not df.empty:
        zone_counts = df.groupby('zone_name')['vehicle_count'].sum().reset_index()
        zone_fig = px.bar(
            zone_counts, 
            x='zone_name', 
            y='vehicle_count',
            title="Total Vehicle Count by Zone",
            color='vehicle_count',
            color_continuous_scale='Blues'
        )
    zone_fig.update_layout(height=300)
    
    # Detection results
    detection_results = html.Div("No recent detections")
    if latest_results and 'detections' in latest_results:
        detections = latest_results['detections']
        detection_results = html.Div([
            html.P(f"🚗 {len(detections)} vehicles detected"),
            html.Ul([
                html.Li(f"Vehicle {i+1}: {det['confidence']:.1%} confidence")
                for i, det in enumerate(detections[:5])
            ])
        ])
    
    # Zone statistics
    zone_stats = html.Div("No zone data available")
    if latest_results and 'zones' in latest_results:
        zones = latest_results['zones']
        zone_stats = html.Div([
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Div([
                        html.H6(zone['name'], className="mb-1"),
                        html.P(f"Vehicles: {zone['current_count']}", className="mb-1"),
                        html.Small(f"Level: {zone['congestion_level']}")
                    ])
                ]) for zone in zones
            ])
        ])
    
    # Table data
    table_data = []
    if not df.empty:
        table_data = df.head(20).to_dict('records')
        for record in table_data:
            record['timestamp'] = record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    
    # Processed image display
    processed_image_display = html.Div("No processed image available")
    if latest_results and 'image_path' in latest_results:
        # Check if there's a detection result image
        image_path = latest_results['image_path']
        image_name = Path(image_path).name
        
        # Look for annotated image in detection_results folder
        annotated_image_path = Path("detection_results") / f"detected_{image_name}"
        
        if annotated_image_path.exists():
            # Convert image to base64 for display
            import base64
            with open(annotated_image_path, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode()
            
            processed_image_display = html.Div([
                html.H6("📸 Processed Image with Detections:", className="mb-2"),
                html.Img(
                    src=f"data:image/jpeg;base64,{encoded_image}",
                    style={'width': '100%', 'max-width': '600px', 'height': 'auto', 'border': '1px solid #ddd', 'borderRadius': '5px'}
                ),
                html.P(f"Image: {image_name}", className="text-muted mt-2")
            ])
    
    return (total_vehicles, congestion_level, congestion_class, processing_time, 
            detection_rate, timeline_fig, zone_fig, detection_results, 
            zone_stats, table_data, processed_image_display)

@app.callback(
    Output('upload-image-status', 'children'),
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')],
    prevent_initial_call=True
)
def update_image_upload_status(contents, filename):
    """Update image upload status"""
    if contents and filename:
        return html.Div([
            "✅ " + filename[:30] + ("..." if len(filename) > 30 else "")
        ], style={'color': 'green'})
    return ""

@app.callback(
    Output('upload-video-status', 'children'),
    [Input('upload-video', 'contents')],
    [State('upload-video', 'filename')],
    prevent_initial_call=True
)
def update_video_upload_status(contents, filename):
    """Update video upload status"""
    if contents and filename:
        return html.Div([
            "✅ " + filename[:30] + ("..." if len(filename) > 30 else "")
        ], style={'color': 'green'})
    return ""

@app.callback(
    [Output('system-status', 'children'),
     Output('start-live-btn', 'disabled'),
     Output('stop-live-btn', 'disabled')],
    [Input('start-live-btn', 'n_clicks'),
     Input('stop-live-btn', 'n_clicks'),
     Input('process-image-btn', 'n_clicks'),
     Input('process-video-btn', 'n_clicks'),
     Input('upload-image', 'contents'),
     Input('upload-video', 'contents')],
    [State('upload-image', 'filename'),
     State('upload-video', 'filename'),
     State('confidence-dropdown', 'value')],
    prevent_initial_call=True
)
def control_system(start_clicks, stop_clicks, process_clicks, video_process_clicks, upload_contents, video_upload_contents, upload_filename, video_upload_filename, confidence):
    """Control live monitoring and image processing"""
    global is_live_monitoring, traffic_processor, live_data_store
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return "🟢 System Ready", False, True
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-live-btn' and traffic_processor:
        if not is_live_monitoring:
            # Test webcam availability with different backends
            import cv2
            camera_found = False
            backend_name = "Unknown"
            
            # Try different backends
            backends = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
                (cv2.CAP_ANY, "Default")
            ]
            
            for backend_id, name in backends:
                test_cap = cv2.VideoCapture(0, backend_id)
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    if ret:
                        camera_found = True
                        backend_name = name
                        test_cap.release()
                        break
                test_cap.release()
            
            if camera_found:
                # Start live monitoring with working backend
                threading.Thread(
                    target=traffic_processor.start_live_monitoring,
                    args=(0,),  # Use webcam
                    daemon=True
                ).start()
                is_live_monitoring = True
                return f"🔴 Live Monitoring Active - {backend_name} Camera", True, False
            else:
                return "❌ Error: No camera detected. Try running as Administrator or check camera permissions in Windows Settings", False, True
    
    elif button_id == 'stop-live-btn' and traffic_processor:
        traffic_processor.stop_live_monitoring()
        is_live_monitoring = False
        return "🟡 Monitoring Stopped", False, True
    
    elif button_id == 'process-image-btn' or button_id == 'upload-image':
        if upload_contents and upload_filename and traffic_processor:
            try:
                # Decode uploaded file
                import base64
                content_type, content_string = upload_contents.split(',')
                decoded = base64.b64decode(content_string)
                
                # Save temporary file
                temp_dir = Path("traffic_monitoring_system/data/temp")
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_file_path = temp_dir / f"uploaded_{upload_filename}"
                
                with open(temp_file_path, 'wb') as f:
                    f.write(decoded)
                
                # Update confidence threshold if provided
                if confidence:
                    traffic_processor.confidence_threshold = confidence
                    traffic_processor.detector.confidence_threshold = confidence
                
                # Process the uploaded image
                results = traffic_processor.process_single_image(str(temp_file_path))
                
                # Update live data store with results
                live_data_store['latest_results'] = results
                live_data_store['update_count'] += 1
                
                # Clean up temp file
                if temp_file_path.exists():
                    temp_file_path.unlink()
                
                if 'error' not in results:
                    return f"✅ Processed {upload_filename} - Found {len(results['detections'])} vehicles", False, True
                else:
                    return f"❌ Error processing {upload_filename}: {results['error']}", False, True
                    
            except Exception as e:
                return f"❌ Upload error: {str(e)}", False, True
    
    elif button_id == 'process-video-btn' or button_id == 'upload-video':
        if video_upload_contents and video_upload_filename and traffic_processor:
            try:
                # Decode uploaded video file
                import base64
                content_type, content_string = video_upload_contents.split(',')
                decoded = base64.b64decode(content_string)
                
                # Save temporary file
                temp_dir = Path("traffic_monitoring_system/data/temp")
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_file_path = temp_dir / f"uploaded_{video_upload_filename}"
                
                with open(temp_file_path, 'wb') as f:
                    f.write(decoded)
                
                # Update confidence threshold if provided
                if confidence:
                    traffic_processor.confidence_threshold = confidence
                    traffic_processor.detector.confidence_threshold = confidence
                
                # Start video processing in background thread
                processing_thread = threading.Thread(
                    target=process_video_with_progress, 
                    args=(str(temp_file_path),)
                )
                processing_thread.start()
                
                return f"🎬 Started processing {video_upload_filename}. Check progress bar above!", False, True
                    
            except Exception as e:
                return f"❌ Video upload error: {str(e)}", False, True
        else:
            return "⚠️ Please select an image file first", False, True
    
    return "🟢 System Ready", False, True

def main():
    """Run the dashboard application"""
    print("🚦 Smart Traffic Monitoring Dashboard")
    print("=" * 50)
    
    # Initialize system
    if not initialize_system():
        print("❌ Failed to initialize system")
        return
    
    print("🌐 Starting web dashboard...")
    print("📱 Open your browser and go to: http://localhost:8050")
    print("🛑 Press Ctrl+C to stop the dashboard")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=8050)
    except KeyboardInterrupt:
        print("\\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

if __name__ == "__main__":
    main()