import glob
import os
import re
import yaml
import argparse
import json

def extract_episode_number(filename):
    """Extract episode number from filename like 'prediction_3.mp4'"""
    match = re.search(r'prediction_(\d+)\.mp4', filename)
    if match:
        return int(match.group(1))
    return None

def format_config_for_html(config, indent=0):
    """Convert config dictionary to formatted HTML"""
    html = ""
    for key, value in config.items():
        spacing = "&nbsp;" * (indent * 4)
        if isinstance(value, dict):
            html += f"{spacing}<strong>{key}:</strong><br>\n"
            html += format_config_for_html(value, indent + 1)
        else:
            html += f"{spacing}<strong>{key}:</strong> {value}<br>\n"
    return html

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate HTML page for GNN simulation videos')
    parser.add_argument('--name', type=str, required=True,
                       help='Model name to display (e.g., 2025-05-31-21-01-09-427982 or custom_model_name)')
    args = parser.parse_args()
    
    model_name = args.name
    
    # Use model-specific configuration
    model_dir = f"data/gnn_dyn_model/{model_name}"
    config_path = os.path.join(model_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Error: Model config file '{config_path}' does not exist!")
        print(f"Please check if the model directory exists and contains config.yaml")
        return
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    N_EPISODE = config['dataset']['n_episode']
    TRAIN_VALID_RATIO = config['train']['train_valid_ratio']
    N_TRAIN = int(N_EPISODE * TRAIN_VALID_RATIO)

    # Convert config to formatted HTML
    config_html = format_config_for_html(config)

    # HTML 头部
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Simulation Videos - Model {model_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f9f9f9;
            color: #333;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 20px;
            color: #2c3e50;
        }}
        h2 {{
            text-align: center;
            margin: 30px 0 20px 0;
            padding: 10px;
            border-radius: 5px;
        }}
        .training {{
            background-color: #e8f5e8;
            color: #2d5a2d;
        }}
        .validation {{
            background-color: #e8f0ff;
            color: #1a4480;
        }}
        .config {{
            background-color: #fff9e6;
            color: #8b6914;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: flex-start;
            margin-bottom: 40px;
        }}
        .case {{
            background-color: #fff;
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
            box-sizing: border-box;
            flex: 0 0 23%;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .training-case {{
            border-left: 4px solid #4caf50;
        }}
        .validation-case {{
            border-left: 4px solid #2196f3;
        }}
        .videos {{
            display: flex;
            justify-content: center;
            gap: 10px;
            width: 100%;
        }}
        .video-container {{
            width: 100%;
            text-align: center;
        }}
        video {{
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .case-name {{
            margin-top: 10px;
            font-weight: bold;
            font-size: 1.1em;
            color: #2c3e50;
        }}
        .episode-type {{
            font-size: 0.9em;
            margin-bottom: 10px;
            padding: 4px 8px;
            border-radius: 12px;
            display: inline-block;
        }}
        .stats {{
            background-color: #fff;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .model-info {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            color: #856404;
        }}
        .config-section {{
            background-color: #fff;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: left;
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
        }}
        .config-toggle {{
            background-color: #fff9e6;
            border: 1px solid #f0e68c;
            padding: 10px 20px;
            margin: 20px auto;
            border-radius: 5px;
            cursor: pointer;
            text-align: center;
            display: block;
            width: 200px;
            font-weight: bold;
            color: #8b6914;
        }}
        .config-toggle:hover {{
            background-color: #fff3cd;
        }}
        .config-content {{
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.4;
        }}
        #config-details {{
            display: none;
        }}
    </style>
    <script>
        function toggleConfig() {{
            var element = document.getElementById('config-details');
            var button = document.getElementById('config-toggle-btn');
            if (element.style.display === 'none' || element.style.display === '') {{
                element.style.display = 'block';
                button.textContent = 'Hide Configuration';
            }} else {{
                element.style.display = 'none';
                button.textContent = 'Show Configuration';
            }}
        }}
    </script>
</head>
<body>
    <h1>GNN Simulation Videos - Model {model_name}</h1>
            
    <button id="config-toggle-btn" class="config-toggle" onclick="toggleConfig()">Show Configuration</button>
    <div id="config-details" class="config-section">
        <h3>Model Configuration</h3>
        <div class="config-content">
            {config_html}
        </div>
    </div>
"""

    # List all mp4 files in the model-specific video directory
    videos_dir = f"data/video/{model_name}"
    
    if not os.path.exists(videos_dir):
        print(f"Error: Video directory '{videos_dir}' does not exist!")
        print(f"Please run gnn_inference.py with model name '{model_name}' first.")
        return
    
    video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))

    if not video_files:
        print(f"No videos found in '{videos_dir}'")
        print(f"Please run gnn_inference.py with model name '{model_name}' first.")
        return

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
            rel_video_path = os.path.join("..", "video", model_name, filename)
            html_content += f"""
                <div class="case training-case">
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

    # Add validation videos section  
    if validation_videos:
        html_content += f"""
        <h2 class="validation">📊 Validation Episodes ({N_TRAIN}-{N_EPISODE-1}) - {len(validation_videos)} videos</h2>
        <div class="container">
    """
        
        for video_path, filename, episode_num in validation_videos:
            rel_video_path = os.path.join("..", "video", model_name, filename)
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

    # Save HTML file in data/html directory
    html_dir = "data/html"
    os.makedirs(html_dir, exist_ok=True)
    output_path = os.path.join(html_dir, f"{model_name}_videos.html")
    
    with open(output_path, "w") as file:
        file.write(html_content)

    print(f"HTML file saved to {output_path}")
    print(f"Model name: {model_name}")
    print(f"Model directory: {model_dir}")
    print(f"Training videos: {len(training_videos)}")
    print(f"Validation videos: {len(validation_videos)}")
    if training_videos:
        print(f"Training episodes: {[v[2] for v in training_videos]}")
    if validation_videos:
        print(f"Validation episodes: {[v[2] for v in validation_videos]}")

if __name__ == "__main__":
    main()