import cv2
import os
def read_video(video_path):
    capture = cv2.VideoCapture(video_path)
    frames=[]
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    return frames  

# def save_video(output_video_frames, output_path):
#     # Choose codec based on file extension
    
#     ext = os.path.splitext(output_path)[1].lower()
#     if ext == '.mp4':
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Better for .mp4
#     else:
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Works well for .avi

#     height, width = output_video_frames[0].shape[:2]
#     out = cv2.VideoWriter(output_path, fourcc, 24, (width, height))

#     for frame in output_video_frames:
#         out.write(frame)
#     out.release()
import cv2

def save_video(frames, output_path, fps=15):
    if not frames:
        raise ValueError("No frames to save.")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # ✅ CORRECT CODEC
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

