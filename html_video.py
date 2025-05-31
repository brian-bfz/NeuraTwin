import glob
import os

# HTML 头部
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simulation Videos Visualizer</title>
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
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: flex-start;
        }
        .case {
            background-color: #fff;
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
            box-sizing: border-box;
            flex: 0 0 23%;
            margin-bottom: 20px;
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
        }
        .case-name {
            margin-top: 10px;
            font-weight: bold;
            font-size: 1.2em;
            color: #2c3e50;
        }
    </style>
</head>
<body>
    <h1>Simulation Videos Visualizer</h1>
    <div class="container">
"""

# List all mp4 files in data/video
videos_dir = "data/video"
os.makedirs(videos_dir, exist_ok=True)
video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))

for video_path in sorted(video_files):
    filename = os.path.basename(video_path)
    rel_video_path = os.path.join("video", filename)
    html_content += f"""
                <div class="case">
                    <div class="case-name">{filename}</div>
                    <div class="videos">
                        <div class="video-container">
                            <video src="{rel_video_path}" controls autoplay muted loop></video>
                            <div>Simulation Video</div>
                        </div>
                    </div>
                </div>
        """

# End of the HTML structure
html_content += """
    </div>
</body>
</html>
"""

output_path = "data/index.html"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as file:
    file.write(html_content)

print(f"HTML file saved to {output_path}")