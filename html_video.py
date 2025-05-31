import glob
import os
import re

# Configuration from training pipeline
N_EPISODE = 25
TRAIN_VALID_RATIO = 0.9
N_TRAIN = int(N_EPISODE * TRAIN_VALID_RATIO)  # 22
# Training episodes: 0-21, Validation episodes: 22-24

def extract_episode_number(filename):
    """Extract episode number from filename like 'prediction_3.mp4'"""
    match = re.search(r'prediction_(\d+)\.mp4', filename)
    if match:
        return int(match.group(1))
    return None

# HTML 头部
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Simulation Videos - Training vs Validation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f9f9f9;
            color: #333;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #2c3e50;
        }
        h2 {
            text-align: center;
            margin: 30px 0 20px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .training {
            background-color: #e8f5e8;
            color: #2d5a2d;
        }
        .validation {
            background-color: #e8f0ff;
            color: #1a4480;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: flex-start;
            margin-bottom: 40px;
        }
        .case {
            background-color: #fff;
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
            box-sizing: border-box;
            flex: 0 0 23%;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .training-case {
            border-left: 4px solid #4caf50;
        }
        .validation-case {
            border-left: 4px solid #2196f3;
        }
        .videos {
            display: flex;
            justify-content: center;
            gap: 10px;
            width: 100%;
        }
        .video-container {
            width: 100%;
            text-align: center;
        }
        video {
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .case-name {
            margin-top: 10px;
            font-weight: bold;
            font-size: 1.1em;
            color: #2c3e50;
        }
        .episode-type {
            font-size: 0.9em;
            margin-bottom: 10px;
            padding: 4px 8px;
            border-radius: 12px;
            display: inline-block;
        }
        .stats {
            background-color: #fff;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>GNN Simulation Videos - Training vs Validation Split</h1>
    
    <div class="stats">
        <strong>Dataset Split:</strong> Episodes 0-21 (Training) | Episodes 22-24 (Validation) | Ratio: 90%-10%
    </div>
"""

# List all mp4 files in data/video
videos_dir = "data/video"
os.makedirs(videos_dir, exist_ok=True)
video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))

# Separate videos into training and validation
training_videos = []
validation_videos = []

for video_path in video_files:
    filename = os.path.basename(video_path)
    episode_num = extract_episode_number(filename)
    
    if episode_num is not None:
        if episode_num < N_TRAIN:  # Episodes 0-21
            training_videos.append((video_path, filename, episode_num))
        else:  # Episodes 22-24
            validation_videos.append((video_path, filename, episode_num))

# Sort by episode number
training_videos.sort(key=lambda x: x[2])
validation_videos.sort(key=lambda x: x[2])

# Add training videos section
if training_videos:
    html_content += f"""
    <h2 class="training">🎯 Training Episodes (0-{N_TRAIN-1}) - {len(training_videos)} videos</h2>
    <div class="container">
"""
    
    for video_path, filename, episode_num in training_videos:
        rel_video_path = os.path.join("video", filename)
        html_content += f"""
            <div class="case training-case">
                <div class="case-name">Episode {episode_num}</div>
                <div class="videos">
                    <div class="video-container">
                        <video src="{rel_video_path}" controls autoplay muted loop></video>
`                    </div>
                </div>
            </div>
    """
    
    html_content += """
    </div>
"""

# Add validation videos section  
if validation_videos:
    html_content += f"""
    <h2 class="validation">📊 Validation Episodes ({N_TRAIN}-{N_EPISODE-1}) - {len(validation_videos)} videos</h2>
    <div class="container">
"""
    
    for video_path, filename, episode_num in validation_videos:
        rel_video_path = os.path.join("video", filename)
        html_content += f"""
            <div class="case validation-case">
                <div class="case-name">Episode {episode_num}</div>
                <div class="videos">
                    <div class="video-container">
                        <video src="{rel_video_path}" controls autoplay muted loop></video>
                    </div>
                </div>
            </div>
    """
    
    html_content += """
    </div>
"""

# End of the HTML structure
html_content += """
</body>
</html>
"""

output_path = "data/index.html"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as file:
    file.write(html_content)

print(f"HTML file saved to {output_path}")
print(f"Training videos: {len(training_videos)}")
print(f"Validation videos: {len(validation_videos)}")
if training_videos:
    print(f"Training episodes: {[v[2] for v in training_videos]}")
if validation_videos:
    print(f"Validation episodes: {[v[2] for v in validation_videos]}")