"""
HTML/Jinja2 renderer for storyboard output.
Styled for mobile with 9:16 aspect ratio containers.
"""

from pathlib import Path

from jinja2 import Environment, select_autoescape

from models import Storyboard


STORYBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TikTok Storyboard: {{ logline[:50] }}...</title>
    <style>
        :root {
            --bg: #0a0a0a;
            --card: #141414;
            --text: #f5f5f5;
            --muted: #a0a0a0;
            --accent: #ff0050;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 1rem;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { font-size: 1.25rem; margin-bottom: 0.5rem; line-height: 1.3; }
        .logline { color: var(--muted); font-size: 0.9rem; margin-bottom: 1rem; }
        .world { color: var(--muted); font-size: 0.85rem; margin-bottom: 1.5rem; line-height: 1.4; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; }
        th, td { border: 1px solid #333; padding: 0.75rem; vertical-align: top; }
        th {
            background: var(--card);
            color: var(--accent);
            font-weight: 600;
            text-align: left;
        }
        .col-image { width: 30%; }
        .col-desc { width: 35%; }
        .col-narration { width: 35%; }
        .scene-img {
            aspect-ratio: 9 / 16;
            width: 100%;
            max-width: 200px;
            object-fit: cover;
            display: block;
            border-radius: 8px;
        }
        .scene-num { color: var(--accent); font-size: 0.8rem; margin-bottom: 0.25rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ logline }}</h1>
        <p class="logline">Storyboard · ~30 sec</p>
        {% if world_building %}
        <p class="world">{{ world_building }}</p>
        {% endif %}
        <table>
            <thead>
                <tr>
                    <th class="col-image">Image</th>
                    <th class="col-desc">Scene Description</th>
                    <th class="col-narration">Narration</th>
                </tr>
            </thead>
            <tbody>
            {% for scene in scenes %}
            <tr>
                <td class="col-image">
                    <div class="scene-num">Scene {{ scene.scene_num }} · {{ scene.camera_angle }} · {{ scene.duration }}s</div>
                    {% if scene.image_path %}
                    <img src="{{ scene.image_path }}" alt="Scene {{ scene.scene_num }}" class="scene-img" loading="lazy">
                    {% else %}
                    <div class="scene-img" style="background:#222;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:0.8rem;">Image not available</div>
                    {% endif %}
                </td>
                <td class="col-desc">{{ scene.visual_prompt }}</td>
                <td class="col-narration">{{ scene.narration }}</td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
"""


def render_storyboard_html(storyboard: Storyboard, output_path: str) -> str:
    """
    Render the Storyboard to an HTML file.
    Images are referenced by path - use relative paths for portability.
    """
    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    template = env.from_string(STORYBOARD_TEMPLATE)

    # Image paths: use filename only when image is in same dir as HTML (for file:// URLs)
    out_dir = Path(output_path).parent.resolve()
    scenes_for_render = []
    for s in storyboard.scenes:
        img_path = Path(s.image_path).resolve()
        try:
            rel = img_path.relative_to(out_dir)
            # Use forward slashes for URLs; if same dir, use just filename
            path_str = rel.as_posix() if len(rel.parts) > 1 else rel.name
        except ValueError:
            path_str = img_path.name
        scenes_for_render.append({
            **s.model_dump(),
            "image_path": path_str,
        })

    html = template.render(
        logline=storyboard.logline,
        world_building=storyboard.world_building,
        scenes=scenes_for_render,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    return output_path
