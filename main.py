#!/usr/bin/env python3
"""
Agentic Video Maker - Single entry point for the full pipeline.

Edit the CONFIG section below, then run: python main.py
  1. Discovery chat - agent asks for your idea, style, character, anchor images
  2. Storyboard creation (storyboard.json + storyboard.html)
  3. Edit loop - change image or narration per scene
  4. When you say 'like': narration -> video -> merge -> movie_final.mp4
"""

import re
import sys
from datetime import datetime
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIG - Edit these
# -----------------------------------------------------------------------------

# Shortcuts (optional). Leave empty/False for normal interactive flow.
IDEA: str = ""  # If set, skip chat and extract from this text
SKIP_DISCOVERY: bool = False  # If True, use default context (no chat)
PROJECT_DIR: str = "projects/movie_2026-03-23_011548"  # If set to existing path (e.g. projects/movie_xxx), skip to edit loop

# Fallback anchors when using IDEA or SKIP_DISCOVERY and agent doesn't extract paths
ANCHOR_IMAGE_PATHS_FALLBACK: list[str] = ['anchor_images/zlisto.jpeg']  # e.g. ["images/lisa.jpg"]

# Chat/LLM model (Gemini)
CHAT_MODEL = "gemini-3-flash-preview"

# Image generation model (Gemini)
IMAGE_MODEL = "gemini-3.1-flash-image-preview"  #"gemini-3-pro-image-preview" or "gemini-3.1-flash-image-preview"

# Narration: elevenlabs or gemini
NARRATION_PROVIDER = "elevenlabs"
VOICE_ID = "dR1Ptm3rjBUIbHiaywdJ"  # Lisa voice id
NARRATION_MODEL_ID = "eleven_v3"
GEMINI_VOICE_NAME = "Kore"
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"

# Video: runway or veo
VIDEO_PROVIDER = "veo"
RUNWAY_MODEL = "gen4_turbo"
VEO_MODEL = "veo-3.1-fast-generate-preview"

PROJECTS_ROOT = Path(__file__).resolve().parent / "projects"
OUTPUT_FILENAME = "movie_final.mp4"

# -----------------------------------------------------------------------------


def resolve_anchors(paths: list[str]) -> list[str]:
    """Resolve anchor paths to absolute. Tries base, images/, projects/ when paths are relative."""
    base = Path(__file__).resolve().parent
    resolved = []
    for p in paths:
        path = Path(p)
        if path.is_absolute():
            if path.exists():
                resolved.append(str(path.resolve()))
            else:
                print(f"Warning: Anchor not found: {p}")
            continue
        candidates = [
            base / p,
            base / "images" / path.name,
            base / "images" / p,
        ]
        found = any(c.exists() for c in candidates)
        if found:
            for c in candidates:
                if c.exists():
                    resolved.append(str(c.resolve()))
                    break
        else:
            print(f"Warning: Anchor not found: {p}")
    return resolved


def main():
    from dotenv import load_dotenv
    load_dotenv()

    from utils import (
        run_storyboard_agent,
        run_storyboard_discovery_chat,
        extract_video_brief,
        update_storyboard_scene,
        run_narration_agent,
        run_video_agent,
        run_merge_agent,
        describe_anchor_images,
        get_storyboard_outline,
        revise_storyboard_outline,
        play_video,
    )
    from models import MovieContext

    base = Path(__file__).resolve().parent

    # Resolve project dir
    if PROJECT_DIR:
        project_dir = None
        for candidate in (Path(PROJECT_DIR), base / PROJECT_DIR, PROJECTS_ROOT / PROJECT_DIR):
            if candidate.exists():
                project_dir = candidate.resolve()
                break
        if project_dir is None:
            print(f"Error: PROJECT_DIR not found: {PROJECT_DIR}")
            return 1
        print(f"Continuing in project: {project_dir}\n")
    else:
        project_name = f"movie_{datetime.now():%Y-%m-%d_%H%M%S}"
        project_dir = PROJECTS_ROOT / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        print(f"Project folder: {project_dir}\n")

        # Discovery + storyboard
        if IDEA:
            print(f"Extracting from idea: {IDEA[:60]}...")
            history = [("Movie idea: " + IDEA, "")]
            pre_anchors = resolve_anchors(ANCHOR_IMAGE_PATHS_FALLBACK)
            context = extract_video_brief(
                history, CHAT_MODEL,
                anchor_image_paths=pre_anchors if pre_anchors else None,
            )
            anchors = pre_anchors if pre_anchors else resolve_anchors(
                getattr(context, "anchor_image_paths", []) or []
            )
        elif SKIP_DISCOVERY:
            context = MovieContext(
                narrative_goal="A day in the life",
                audience="Gen Z",
                characters="One person",
                vibe="Casual",
                video_style="social_media",
                num_scenes=4,
                anchor_image_paths=ANCHOR_IMAGE_PATHS_FALLBACK,
                narrator="",
                narrator_speaking_style="conversational",
            )
            anchors = resolve_anchors(ANCHOR_IMAGE_PATHS_FALLBACK)
            print("Using default context (SKIP_DISCOVERY)")
        else:
            print("\n" + "=" * 50)
            print("  Jarvis: Good evening. I'm Jarvis, your creative assistant.")
            print("  I'll help you craft a video storyboard. Simply answer as we go.")
            print("  Type 'done', 'proceed', 'proceed with storyboard', or 'ok' when ready.")
            print("=" * 50 + "\n")

            pre_anchors = resolve_anchors(ANCHOR_IMAGE_PATHS_FALLBACK)
            image_description = ""
            if pre_anchors:
                print("Jarvis: Reviewing your reference images...\n")
                image_description = describe_anchor_images(pre_anchors, CHAT_MODEL)
                if image_description:
                    print("Jarvis:", image_description)
                    print()
                print("Jarvis: I've noted what I see. Tell me more when you're ready.\n")

            history = run_storyboard_discovery_chat(
                CHAT_MODEL,
                anchor_image_paths=pre_anchors if pre_anchors else None,
                image_description=image_description or None,
            )
            context = extract_video_brief(
                history, CHAT_MODEL,
                anchor_image_paths=pre_anchors if pre_anchors else None,
            )
            anchors = pre_anchors if pre_anchors else resolve_anchors(
                getattr(context, "anchor_image_paths", []) or ANCHOR_IMAGE_PATHS_FALLBACK
            )

        if not anchors:
            print("Error: No anchor images. Provide paths in chat or set ANCHOR_IMAGE_PATHS_FALLBACK in config.")
            return 1

        print("\n--- Storyboard outline ---")
        overview, script = get_storyboard_outline(context, CHAT_MODEL)
        print(f"\nLogline: {overview.logline}")
        print(f"Scenes: {len(script.scenes)}")
        for s in script.scenes:
            print(f"  Scene {s.scene_num}: {s.visual_prompt[:70]}...")
            print(f"    Narration: {s.narration[:60]}...")
        print("\nJarvis: Here is your storyboard outline. Type 'yes'/'proceed' to generate images, or say what to change (e.g. '2 scenes').")
        outline_confirm = ("yes", "proceed", "ok", "go", "looks good", "confirmed")
        while True:
            confirm = input("\nYou: ").strip()
            if not confirm:
                continue
            confirm_lower = confirm.lower()
            if confirm_lower in outline_confirm:
                break
            # Try to parse "2 scenes" or "make it 2 scenes"
            m = re.search(r"(\d+)\s*scenes?", confirm_lower)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 10:
                    context.num_scenes = n
                    print("Jarvis: Revising outline for", n, "scenes...")
                    overview, script = get_storyboard_outline(context, CHAT_MODEL)
                    print(f"\nLogline: {overview.logline}\nScenes: {len(script.scenes)}")
                    for s in script.scenes:
                        print(f"  Scene {s.scene_num}: {s.visual_prompt[:70]}...")
                        print(f"    Narration: {s.narration[:60]}...")
                    print("\nJarvis: Updated. Type 'yes' to proceed.")
                    continue
            # Any other feedback: revise the outline with user's requested change
            print("Jarvis: Revising outline based on your feedback...")
            overview, script = revise_storyboard_outline(
                overview, script, confirm, context, CHAT_MODEL
            )
            print(f"\nLogline: {overview.logline}\nScenes: {len(script.scenes)}")
            for s in script.scenes:
                print(f"  Scene {s.scene_num}: {s.visual_prompt[:70]}...")
                print(f"    Narration: {s.narration[:60]}...")
            print("\nJarvis: Updated. Type 'yes' to proceed.")

        print("\n--- Creating storyboard ---")
        run_storyboard_agent(
            anchor_image_paths=anchors,
            output_dir=str(project_dir),
            chat_model=CHAT_MODEL,
            image_model=IMAGE_MODEL,
            context=context,
            vertical=True,
            generate_images=True,
        )
        print(f"\nStoryboard: {project_dir / 'storyboard.html'}")
        print("Open storyboard.html in your browser to preview.\n")

    # Edit loop
    print("=" * 60)
    print("  Edit loop - Ask to change an image or narration in a scene.")
    print("  Examples: 'change scene 2 narration to ...' or 'change scene 3 image to ...'")
    print("  Type 'like' or 'done' when you approve the storyboard.")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue

        lower = user_input.lower()
        if lower in ("like", "done", "ok", "approve", "generate", "go"):
            print("\nGenerating movie...")
            break

        match = re.search(
            r"change\s+scene\s+(\d+)\s+(narration|image)\s+to\s+(.+)",
            user_input,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            scene_num = int(match.group(1))
            field = match.group(2).lower()
            value = match.group(3).strip()
            try:
                if field == "narration":
                    update_storyboard_scene(
                        project_dir,
                        scene_num,
                        narration=value,
                        image_model=IMAGE_MODEL,
                    )
                    print(f"  Updated scene {scene_num} narration.")
                else:
                    if Path(value).exists():
                        update_storyboard_scene(
                            project_dir,
                            scene_num,
                            image_file_path=value,
                            image_model=IMAGE_MODEL,
                        )
                    else:
                        update_storyboard_scene(
                            project_dir,
                            scene_num,
                            new_image_prompt=value,
                            image_model=IMAGE_MODEL,
                        )
                    print(f"  Updated scene {scene_num} image.")
            except Exception as e:
                print(f"  Error: {e}")
            continue

        print("  (Say 'change scene N narration/image to ...' or 'like' when done)")

    # Run production pipeline
    print("\n" + "=" * 60)
    print("  Step 1: Narration")
    print("=" * 60)
    run_narration_agent(
        project_dir,
        provider=NARRATION_PROVIDER,
        voice_id=VOICE_ID,
        model_id=NARRATION_MODEL_ID,
        voice_name=GEMINI_VOICE_NAME,
        tts_model=GEMINI_TTS_MODEL,
    )

    print("\n" + "=" * 60)
    print("  Step 2: Video")
    print("=" * 60)
    run_video_agent(
        project_dir,
        provider=VIDEO_PROVIDER,
        model=RUNWAY_MODEL if VIDEO_PROVIDER == "runway" else VEO_MODEL,
        chat_model=CHAT_MODEL,
    )

    print("\n" + "=" * 60)
    print("  Step 3: Merge")
    print("=" * 60)
    final_path = run_merge_agent(project_dir, OUTPUT_FILENAME)
    print("\nJarvis: Your video is ready.")
    print(f"  {final_path}")
    play_prompt = input("\nJarvis: Want me to play it for you? (yes/no): ").strip().lower()
    if play_prompt in ("yes", "y", "play", "sure"):
        try:
            play_video(final_path)
            print("Jarvis: Playing.")
        except Exception as e:
            print(f"Jarvis: Couldn't launch player: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())





