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
            justify-content: center;
        }
        .case {
            width: 100%;
            background-color: #fff;
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
            box-sizing: border-box;
        }
        .videos {
            display: flex;
            justify-content: center;
            gap: 10px;
            width: 100%;
        }
        .video-container {
            width: 48%;
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

# Get all case directories in generated_data
base_path = "generated_data"
dir_names = glob.glob(f"{base_path}/*")

for dir_name in dir_names:
    case_name = os.path.basename(dir_name)
    
    # Get the video path - assuming it's named 'output.mp4' in each case directory
    video_path = os.path.join(dir_name, "output.mp4")
    # Make the path relative to generated_data (where the HTML will be)
    rel_video_path = os.path.join(case_name, "output.mp4")
    
    # Only add to HTML if video exists
    if os.path.exists(video_path):
        html_content += f"""
                    <div class="case">
                        <div class="case-name">{case_name}</div>
                        <div class="videos">
                            <div class="video-container">
                                <video src="{rel_video_path}" controls></video>
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

output_path = "generated_data/index.html"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as file:
    file.write(html_content)

print(f"HTML file saved to {output_path}")