from flask import Flask, request, render_template, send_file, jsonify, Response, after_this_request
from utils import read_video, save_video
from trackers import Tracker
from team_assignment import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
import os
import uuid
import tempfile
import cv2
import re

app = Flask(__name__)
UPLOAD_FOLDER = tempfile.gettempdir()  # Use OS temp dir
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video/<filename>')
def stream_video(filename):
    """Stream video with Range request support for proper browser playback"""
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(video_path):
        return "Video not found", 404
    
    range_header = request.headers.get('Range', None)
    byte1, byte2 = 0, None
    
    if range_header:
        match = re.search(r'(\d+)-(\d*)', range_header)
        groups = match.groups()
        
        if groups[0]:
            byte1 = int(groups[0])
        if groups[1]:
            byte2 = int(groups[1])
    
    chunk, start, length, file_size = get_chunk(video_path, byte1, byte2)
    
    resp = Response(chunk, 
                    206 if range_header else 200,  # Partial content or OK
                    mimetype='video/mp4',
                    direct_passthrough=True)
    
    if range_header:
        resp.headers.add('Content-Range', f'bytes {start}-{start + length - 1}/{file_size}')
    
    resp.headers.add('Accept-Ranges', 'bytes')
    resp.headers.add('Content-Length', str(length))
    
    return resp

def get_chunk(full_path, byte1=None, byte2=None):
    """Get video chunk for streaming"""
    file_size = os.path.getsize(full_path)
    start = 0
    
    if byte1 < file_size:
        start = byte1
    if byte2:
        length = byte2 + 1 - byte1
    else:
        length = file_size - start
    
    with open(full_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(length)
    
    return chunk, start, len(chunk), file_size

@app.route('/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'message': 'No video file uploaded.'}), 400

    video = request.files['video']
    unique_name = str(uuid.uuid4()) + "_" + video.filename
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    base_name = os.path.splitext(unique_name)[0]
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{base_name}.mp4")

    try:
        video.save(input_path)

        # Step 1: Read and process video
        video_frames = read_video(input_path)
        # video_frames = video_frames[:10] #sample frames for testing
        tracker = Tracker('models/best.pt')
        tracks = tracker.get_object_tracks(video_frames)
        tracks['ball'] = tracker.interpolate_ball(tracks['ball'])

        team_assigner = TeamAssigner()
        team_assigner.assign_teams(video_frames[0], tracks['player'][0])
        tracks = team_assigner.get_team_colour(tracks, video_frames)

        player_ball_assigner = PlayerBallAssigner()
        team_ball_control = player_ball_assigner.assign_possession_to_team(tracks)

        output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)


        save_video(output_frames, output_path)
        @after_this_request
        def remove_file(response):
            try:
                os.remove(output_path)
                print(f"Deleted: {output_path}")
            except Exception as e:
                print(f"Failed to delete {output_path}: {e}")
            return response
        # Step 3: Return JSON with video URL instead of direct file
        video_filename = os.path.basename(output_path)
        return send_file(output_path, as_attachment=True, mimetype='video/mp4')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Error: {str(e)}'}), 500

    finally:
        # Clean up original video file after processing
        if os.path.exists(input_path):
            os.remove(input_path)

if __name__ == "__main__":
    app.run(debug=True)
