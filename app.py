from flask import Flask, request, render_template, send_file, jsonify, after_this_request
from utils import read_video, save_video
from trackers import Tracker
from team_assignment import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
import os
import uuid
import tempfile
import cv2
import traceback

app = Flask(__name__)
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

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
        # video_frames=video_frames[:10] #uncomment to test sample
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

        return send_file(output_path, as_attachment=True, mimetype='video/mp4')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'message': f'Error: {str(e)}'}), 500

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

if __name__ == "__main__":
    app.run(debug=True)
