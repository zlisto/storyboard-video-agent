"""
Agentic Video Maker utilities: image generation, video (Runway/Veo), narration (ElevenLabs/Gemini TTS).
"""

import base64
import mimetypes
import os
import platform
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# API keys - support both RUNWAY_API_KEY and RUNWAYML_API_SECRET
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY") or os.getenv("RUNWAYML_API_SECRET")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def play_video(video_path: str | Path) -> None:
    """Open the video file with the system's default player (cross-platform)."""
    path = Path(video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    sys_name = platform.system()
    if sys_name == "Windows":
        os.startfile(str(path))
    elif sys_name == "Darwin":  # macOS
        subprocess.run(["open", str(path)], check=True)
    else:
        subprocess.run(["xdg-open", str(path)], check=True)


def get_image_as_data_uri(image_path: str) -> str:
    """Convert a local image file to a base64 data URI for API uploads."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    content_type, _ = mimetypes.guess_type(str(path))
    content_type = content_type or "image/jpeg"
    return f"data:{content_type};base64,{base64_image}"


def create_image_to_video(
    prompt_text: str,
    image_path: str,
    model: str = "gen3a_turbo",
    ratio: str = "16:9",
    duration: int = 5,
    output_dir: str | None = None,
    output_path: str | None = None,
    wait: bool = True,
):
    """
    Create a video from an image using Runway's image-to-video API.

    Args:
        prompt_text: Description of what should happen in the video.
        image_path: Path to the anchor/reference image.
        model: Runway model. Options: gen3a_turbo, gen4_turbo, gen4.5, act_two, veo3, veo3.1, veo3.1_fast.
        ratio: Output aspect ratio (e.g., "1280:720", "720:1280" for vertical).
        duration: Video duration in seconds (2-10).
        output_dir: Directory to save the output video URL/file.
        wait: If True, wait for task to complete and return output.

    Returns:
        Task output with video URL(s), or task ID if wait=False.
    """
    from runwayml import RunwayML, TaskFailedError, TaskTimeoutError

    if not RUNWAY_API_KEY:
        raise ValueError("RUNWAY_API_KEY or RUNWAYML_API_SECRET must be set in .env")

    client = RunwayML(api_key=RUNWAY_API_KEY.strip())
    image_uri = get_image_as_data_uri(image_path)

    kwargs = {
        "model": model,
        "prompt_image": image_uri,
        "prompt_text": prompt_text,
        "ratio": ratio,
        "duration": duration,
    }

    task = client.image_to_video.create(**kwargs)

    if not wait:
        return {"task_id": task.id, "status": "PENDING"}

    try:
        result = task.wait_for_task_output(timeout=600)
        output_url = result.output[0] if result.output else None

        if output_url:
            if output_path:
                out_path = _save_video_from_url(output_url, "", output_path=output_path)
            elif output_dir:
                out_path = _save_video_from_url(output_url, output_dir, "image_to_video")
            else:
                return {"output": result.output, "task_id": result.id}
            return {"output": result.output, "task_id": result.id, "output_path": str(out_path)}

        return {"output": result.output, "task_id": result.id}
    except TaskFailedError as e:
        raise RuntimeError(f"Runway task failed: {e.task_details}") from e
    except TaskTimeoutError as e:
        raise TimeoutError(f"Runway task timed out: {e}") from e


def create_anchor_image_with_gemini(
    prompt_text: str,
    reference_image_path: str,
    model: str = "gemini-3.1-flash-image-preview",
    output_path: str | None = None,
    aspect_ratio: str = "9:16",
) -> str:
    """
    Generate an anchor image using Gemini Nano Banana 2 (image editing).
    Puts the person from the reference image into a new setting with desired clothes.

    Uses gemini-3.1-flash-image-preview (Nano Banana 2) - better than Runway for
    character-in-scene generation.

    Args:
        prompt_text: Scene description (e.g. "Put this person in a coffee shop wearing a red sweater").
        reference_image_path: Path to base/character photo.
        model: Gemini image model. Options: gemini-3.1-flash-image-preview (Nano Banana 2, fast),
               gemini-3-pro-image-preview (Pro, higher quality).
        output_path: Where to save the generated image.
        aspect_ratio: "9:16" (vertical) or "16:9" (landscape). Also supports 1:1, 3:4, 4:3, etc.

    Returns:
        Path to saved image or URL if output_path not given.
    """
    from google import genai
    from PIL import Image

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY must be set in .env")

    ref_path = Path(reference_image_path).resolve()
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_image_path}")

    client = genai.Client(api_key=GEMINI_API_KEY.strip())
    ref_image = Image.open(ref_path)

    # Ensure prompt asks for photorealistic output suitable for video first frame
    enhanced_prompt = (
        f"{prompt_text.strip()} "
        "Photorealistic, cinematic lighting, high quality. "
        "The person should be clearly visible and well-lit, suitable as a video frame."
    )

    from google.genai import types

    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
    )
    response = client.models.generate_content(
        model=model,
        contents=[enhanced_prompt, ref_image],
        config=config,
    )

    out_image = None
    for part in response.parts:
        if getattr(part, "inline_data", None) is not None:
            out_image = part.as_image()
            break

    if out_image is None:
        raise RuntimeError("Gemini image generation failed: no image in response")

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_image.save(str(out_path))
        return str(out_path)
    raise ValueError("output_path is required for create_anchor_image_with_gemini")


def create_veo_video(
    prompt_text: str,
    image_path: str | list[str] | None = None,
    first_frame_image: str | None = None,
    model: str = "veo-3.1-generate-preview",
    aspect_ratio: str = "9:16",
    duration_seconds: int = 8,
    output_dir: str | None = None,
    generate_audio: bool = True,
    resolution: str = "720p",
    timeout: int = 600,
):
    """
    Create a video using Google Veo 3.1 via Gemini API.
    Uses reference images to preserve subject appearance while placing them in new scenes.

    Args:
        prompt_text: Description of the video (e.g. "Lisa walking through Yale campus").
        image_path: Path(s) for reference images (limited Gemini API support). Prefer first_frame_image.
        first_frame_image: Path to image to animate as first frame (reliable; use with Runway-generated image).
        model: Options: veo-3.1-generate-preview (full Veo 3), veo-3.1-fast-generate-preview (faster).
        aspect_ratio: "16:9" or "9:16".
        duration_seconds: 8 when using reference images (required by API).
        output_dir: Directory to save the output video.
        generate_audio: Ignored (Gemini API doesn't support it; Vertex AI does).
        resolution: "720p", "1080p", or "4k".
        timeout: Max seconds to wait for generation.

    Returns:
        Dict with output path and task info.
    """
    import time

    from google import genai
    from google.genai import types

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY must be set in .env")

    client = genai.Client(api_key=GEMINI_API_KEY.strip())

    paths = []
    if image_path:
        paths = [image_path] if isinstance(image_path, str) else list(image_path)
        paths = [str(Path(p).resolve()) for p in paths[:3]]

    reference_images = [
        types.VideoGenerationReferenceImage(
            image=types.Image.from_file(location=p),
            reference_type="asset",
        )
        for p in paths
    ]

    config_kwargs = dict(
        aspect_ratio=aspect_ratio,
        number_of_videos=1,
        duration_seconds=duration_seconds,
        resolution=resolution,
    )
    if reference_images:
        config_kwargs["reference_images"] = reference_images
        config_kwargs["duration_seconds"] = 8  # Required when using reference images

    gen_kwargs = dict(model=model, prompt=prompt_text, config=types.GenerateVideosConfig(**config_kwargs))
    if first_frame_image:
        gen_kwargs["image"] = types.Image.from_file(location=str(Path(first_frame_image).resolve()))

    operation = client.models.generate_videos(**gen_kwargs)

    elapsed = 0
    while not operation.done and elapsed < timeout:
        time.sleep(10)
        elapsed += 10
        operation = client.operations.get(operation)

    if not operation.done:
        raise TimeoutError(f"Veo video generation timed out after {timeout}s")

    if getattr(operation, "error", None):
        raise RuntimeError(f"Veo generation failed: {operation.error}")

    result = getattr(operation, "result", None) or getattr(operation, "response", None)
    if not result or not getattr(result, "generated_videos", None):
        err_parts = ["Veo generation failed: no video in response."]
        if getattr(operation, "error", None):
            err_parts.append(f"Error: {operation.error}")
        if getattr(operation, "metadata", None):
            err_parts.append(f"Metadata: {operation.metadata}")
        # Common causes: content policy rejection, model not enabled, quota
        err_parts.append(
            "Possible causes: content policy blocked the prompt/image; Veo model needs allowlist; "
            "or try Runway instead (VIDEO_PROVIDER=runway in main.py)."
        )
        raise RuntimeError(" ".join(err_parts))

    generated_video = result.generated_videos[0]
    client.files.download(file=generated_video)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        prefix = "veo_simple" if not reference_images and not first_frame_image else "veo_video"
        out_path = Path(output_dir) / f"{prefix}.mp4"
        generated_video.video.save(str(out_path))
        return {"output_path": str(out_path), "video": generated_video}
    else:
        return {"video": generated_video}


def generate_audio_with_elevenlabs(
    text: str,
    voice_id: str | None = None,
    output_path: str | None = None,
    model_id: str = "eleven_v3",
    stability: float = 0.5,
    similarity_boost: float = 0.8,
) -> bytes:
    """
    Generate speech audio from text using ElevenLabs API.

    Args:
        model_id: TTS model (eleven_multilingual_v2, eleven_v3, etc.).

    Returns:
        Audio bytes (MP3).
    """
    import httpx

    voice_id = voice_id or ELEVENLABS_VOICE_ID
    if not ELEVENLABS_API_KEY or not voice_id:
        raise ValueError("ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID must be set in .env")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
        },
    }

    with httpx.Client() as client:
        resp = client.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        audio_bytes = resp.content

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

    return audio_bytes


def _save_video_from_url(
    url: str, output_dir: str, prefix: str = "video", output_path: str | None = None
) -> Path:
    """Download video from URL and save to output_dir or output_path."""
    import httpx

    if output_path:
        out_path = Path(output_path)
    else:
        from urllib.parse import urlparse
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path_part = urlparse(url).path
        stem = Path(path_part).stem or "output"
        out_path = Path(output_dir) / f"{prefix}_{stem}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client() as client:
        resp = client.get(url, timeout=120)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
    return out_path


def generate_audio_with_gemini_tts(
    text: str,
    voice_name: str = "Kore",
    model: str = "gemini-2.5-flash-preview-tts",
    output_path: str | None = None,
) -> bytes:
    """
    Generate speech audio from text using Gemini TTS API.

    Args:
        text: Text to speak. You can prefix with style hints, e.g. "Say cheerfully: Hello!"
        voice_name: Prebuilt voice. Options: Kore, Puck, Charon, Fenrir, Aoede, Zephyr,
                    Orus, Leda, Adam, etc. See Gemini TTS voices docs.
        model: TTS model. Options: gemini-2.5-flash-preview-tts (fast, low latency),
               gemini-2.5-flash-lite-preview-tts (budget), gemini-2.5-pro-preview-tts (high quality).
        output_path: Where to save (saved as MP3 for compatibility with merge).

    Returns:
        Audio bytes (MP3).
    """
    import subprocess
    import tempfile
    import wave

    from google import genai
    from google.genai import types

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY must be set in .env")

    client = genai.Client(api_key=GEMINI_API_KEY.strip())
    response = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name,
                    )
                )
            ),
        ),
    )

    data = response.candidates[0].content.parts[0].inline_data.data
    if isinstance(data, str):
        import base64
        pcm = base64.b64decode(data)
    else:
        pcm = data

    # PCM is 24kHz, 16-bit, mono. Save as WAV, then convert to MP3 for merge compatibility
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_f:
        wav_path = wav_f.name
    try:
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm)

        mp3_path = Path(output_path) if output_path else Path(wav_path).with_suffix(".mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-acodec", "libmp3lame", "-q:a", "2", str(mp3_path)],
            check=True,
            capture_output=True,
        )
        mp3_bytes = mp3_path.read_bytes()
        return mp3_bytes
    finally:
        Path(wav_path).unlink(missing_ok=True)


# =============================================================================
# AGENTIC VIDEO PIPELINE - All agents consolidated
# =============================================================================

# Chat model options (Gemini): gemini-3-flash-preview, gemini-2.0-flash, gemini-2.5-flash-preview, gemini-1.5-pro
# Image model options (Gemini): gemini-3.1-flash-image-preview, gemini-3-pro-image-preview
# Narration: elevenlabs (voice_id + model_id) or gemini (voice_name + model)
# Video: runway (model) or veo (model)


def describe_anchor_images(image_paths: list[str], model: str = "gemini-2.0-flash") -> str:
    """
    Use Gemini vision to describe what's in the anchor images.
    Returns a concise description: people (who they might be), scenes, setting, etc.
    """
    if not image_paths:
        return ""
    from google import genai
    from PIL import Image

    if not GEMINI_API_KEY:
        return ""
    client = genai.Client(api_key=GEMINI_API_KEY.strip())
    parts = []
    for p in image_paths[:5]:  # Limit to 5 images
        path = Path(p)
        if not path.exists():
            continue
        img = Image.open(path)
        parts.append(img)
    if not parts:
        return ""
    prompt = (
        "Describe these images concisely for a video producer. "
        "If they show people: who they might be, appearance, expression, what they're doing. "
        "If they show scenes/locations: what place, setting, mood. "
        "Keep it to 2-3 sentences per image. Be specific and observant."
    )
    response = client.models.generate_content(
        model=model,
        contents=[prompt] + parts,
    )
    return (response.text or "").strip()


def _rel_path(path: str, base: Path) -> str:
    """Return path relative to base, or filename if not under base."""
    p = Path(path)
    if not p.exists():
        return p.name
    try:
        return str(p.resolve().relative_to(base.resolve()))
    except ValueError:
        return p.name


def _get_gemini_chat_agent(model_name: str):
    """Create PydanticAI chat agent with Gemini."""
    from pydantic_ai import Agent
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in .env")
    provider = GoogleProvider(api_key=api_key)
    model = GoogleModel(model_name, provider=provider)
    return Agent(model, system_prompt="")


def _get_discovery_agent(model_name: str, anchor_image_paths: list[str] | None = None):
    """Discovery agent - extracts MovieContext from conversation."""
    from pydantic_ai import Agent
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    from models import MovieContext

    api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in .env")
    provider = GoogleProvider(api_key=api_key)
    model = GoogleModel(model_name, provider=provider)

    anchor_note = ""
    if anchor_image_paths:
        anchor_note = (
            f" IMPORTANT: Anchor image paths are already provided: {anchor_image_paths}. "
            "Use ONLY these for anchor_image_paths. Do not extract different paths from the conversation."
        )

    return Agent(
        model,
        deps_type=None,
        output_type=MovieContext,
        system_prompt=(
            "You are the Executive Producer for a short video storyboard. "
            "Given a conversation with a user about their video idea, extract and populate "
            "MovieContext with: narrative_goal, audience, characters, vibe, vertical, "
            "video_style (social_media, corporate_ad, political_campaign, or informational), "
            "num_scenes (infer from user: e.g. '1st scene, 2nd scene' = 2; default 4 if unclear), "
            "outfits_per_scene (list of outfit descriptions, one per scene or one for all), "
            "anchor_image_paths (list of file paths - only if not already provided), "
            "narrator (who is the narrator - e.g. Lisa, a young professional, the host), "
            "and narrator_speaking_style (e.g. casual, punchy, formal, conversational, energetic, calm). "
            "Be concise. Default vertical=True."
            + anchor_note
        ),
    )


def run_storyboard_discovery_chat(
    model_name: str = "gemini-2.0-flash",
    triggers: tuple[str, ...] = (
        "done", "generate", "go", "create", "like", "ok", "approve",
        "proceed", "ready", "let's go", "proceed with storyboard",
        "create storyboard", "make storyboard",
    ),
    anchor_image_paths: list[str] | None = None,
    image_description: str | None = None,
) -> list[tuple[str, str]]:
    """
    Interactive discovery chat with Jarvis. Returns chat history.
    User types one of triggers when ready.
    """
    from pydantic_ai import Agent

    def _wants_to_proceed(text: str) -> bool:
        lower = text.lower().strip()
        if lower in triggers:
            return True
        # Phrase triggers: "proceed with storyboard", "go ahead", etc.
        proceed_phrases = ("proceed with", "create storyboard", "make storyboard", "go ahead")
        return any(p in lower for p in proceed_phrases)
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in .env")
    provider = GoogleProvider(api_key=api_key)
    model = GoogleModel(model_name, provider=provider)

    image_context = ""
    if image_description and image_description.strip():
        image_context = (
            f"\n\nYou have already reviewed the user's reference images. Your analysis: {image_description}\n"
            "Use this to ask natural follow-up questions (e.g. who are the people, what scene is this)."
        )
    if anchor_image_paths:
        image_context += (
            "\n\nAnchor image paths are already provided. Do NOT ask for or suggest different images. "
            "Use only the provided anchor images."
        )

    chat_agent = Agent(
        model,
        system_prompt=(
            "You are Jarvis, Tony Stark's AI assistant from Iron Man. Sophisticated, dry wit, efficient. "
            "You're helping create a short video storyboard. Speak as Jarvis would—brief, clever, never verbose. "
            "CRITICAL: Ask ONE question at a time. Never list multiple questions. Never overwhelm with text. "
            "Keep every response to 1-3 short sentences. "
            "Gather over the conversation: narrative_goal, audience, characters, vibe, video_style "
            "(social_media/corporate_ad/political_campaign/informational), num_scenes (3-5), "
            "outfits_per_scene, narrator, narrator_speaking_style. "
            "If you've seen their images: ask naturally (e.g. 'Who is the person in the first image?' or 'What scene is this?'). "
            "When they say 'done', 'generate', 'proceed', 'proceed with storyboard', 'ok', etc., one brief confirmation only."
            + image_context
        ),
    )
    history: list[tuple[str, str]] = []
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if _wants_to_proceed(user_input):
            convo = "\n\n".join(f"User: {u}\nAssistant: {a}" for u, a in history)
            prompt = (
                f"The user said '{user_input}'. Based on this conversation:\n\n{convo}\n\n"
                "Briefly confirm you're ready to create the storyboard (1-2 sentences, Jarvis style)."
            )
            result = chat_agent.run_sync(prompt)
            history.append((user_input, str(result.output)))
            print(f"\nJarvis: {result.output}\n")
            return history
        convo = "\n\n".join(f"User: {u}\nAssistant: {a}" for u, a in history)
        prompt = f"{convo}\n\nUser: {user_input}" if history else user_input
        result = chat_agent.run_sync(prompt)
        history.append((user_input, str(result.output)))
        print(f"\nJarvis: {result.output}\n")


def extract_video_brief(
    chat_history: list[tuple[str, str]],
    model_name: str = "gemini-2.0-flash",
    anchor_image_paths: list[str] | None = None,
):
    """Extract MovieContext from discovery chat history."""
    from models import MovieContext

    agent = _get_discovery_agent(model_name, anchor_image_paths=anchor_image_paths)
    convo = "\n\n".join(f"User: {u}\nAssistant: {a}" for u, a in chat_history)
    result = agent.run_sync(f"Extract the MovieContext from this conversation:\n\n{convo}")
    ctx = result.output
    if anchor_image_paths:
        ctx.anchor_image_paths = anchor_image_paths
    return ctx


def _run_synthesis(context, model_name: str):
    from models import MovieOverview
    from pydantic_ai import Agent
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
    provider = GoogleProvider(api_key=api_key)
    model = GoogleModel(model_name, provider=provider)
    agent = Agent(
        model,
        deps_type=None,
        output_type=MovieOverview,
        system_prompt=(
            "You are the Creative Lead. Given MovieContext, produce MovieOverview with "
            "logline (one sentence) and world_building (setting, atmosphere, visual style)."
        ),
    )
    ctx_str = (
        f"narrative_goal: {context.narrative_goal}\n"
        f"audience: {context.audience}\n"
        f"characters: {context.characters}\n"
        f"vibe: {context.vibe}\n"
        f"video_style: {getattr(context, 'video_style', 'social_media')}\n"
        f"num_scenes: {getattr(context, 'num_scenes', 4)}\n"
        f"narrator: {getattr(context, 'narrator', '')}\n"
        f"narrator_speaking_style: {getattr(context, 'narrator_speaking_style', '')}"
    )
    return agent.run_sync(f"Create MovieOverview for:\n{ctx_str}").output


def _run_screenwriter(overview, context, model_name: str):
    from models import Script
    from pydantic_ai import Agent
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
    provider = GoogleProvider(api_key=api_key)
    model = GoogleModel(model_name, provider=provider)
    agent = Agent(
        model,
        deps_type=None,
        output_type=Script,
        system_prompt=(
            "You are a screenwriter. Given MovieOverview and video_style, write a Script with "
            "exactly 3-5 scenes. Each scene: scene_num, visual_prompt, narration, duration (2-10s), "
            "camera_angle. Total duration under 30s. Match the video_style. "
            "If outfits_per_scene is provided, include outfit in visual_prompt for each scene. "
            "Narration should be written in the narrator's voice and speaking style."
        ),
    )
    num_scenes = getattr(context, "num_scenes", 4)
    outfits = getattr(context, "outfits_per_scene", []) or []
    narrator = getattr(context, "narrator", "") or ""
    speaking_style = getattr(context, "narrator_speaking_style", "") or ""
    extra = f"\nCRITICAL: Create EXACTLY {num_scenes} scenes—no more, no fewer. If user said 2 scenes, output only 2."
    if outfits:
        extra += f"\nOutfits per scene: {outfits}"
    if narrator or speaking_style:
        extra += f"\nNarrator: {narrator}. Speaking style: {speaking_style}. Write narration in this voice."
    return agent.run_sync(
        f"logline: {overview.logline}\nworld_building: {overview.world_building}{extra}"
    ).output


def revise_storyboard_outline(
    overview: "MovieOverview",
    script: "Script",
    user_feedback: str,
    context: "MovieContext",
    chat_model: str = "gemini-2.0-flash",
) -> tuple["MovieOverview", "Script"]:
    """
    Revise the script based on user feedback (e.g. "3rd person narration", "shorter narration").
    Returns (overview, revised_script).
    """
    from models import Script
    from pydantic_ai import Agent
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
    provider = GoogleProvider(api_key=api_key)
    model = GoogleModel(chat_model, provider=provider)
    agent = Agent(
        model,
        deps_type=None,
        output_type=Script,
        system_prompt=(
            "You are a screenwriter. Given an existing Script and user feedback, output a revised Script. "
            "Apply the requested changes exactly. Keep the same number of scenes, same structure "
            "(scene_num, visual_prompt, narration, duration, camera_angle). "
            "Preserve logline and world_building intent. Output only the revised Script as structured data."
        ),
    )
    scenes_text = "\n".join(
        f"Scene {s.scene_num}: visual={s.visual_prompt[:120]}... | narration={s.narration} | duration={s.duration}s"
        for s in script.scenes
    )
    prompt = (
        f"Current outline:\nLogline: {overview.logline}\nWorld: {overview.world_building}\n\n"
        f"Scenes:\n{scenes_text}\n\n"
        f"User requested change: {user_feedback}\n\n"
        f"Output the full revised Script with {len(script.scenes)} scenes, applying the change."
    )
    revised = agent.run_sync(prompt).output
    return overview, revised


def get_storyboard_outline(
    context: "MovieContext",
    chat_model: str = "gemini-2.0-flash",
) -> tuple["MovieOverview", "Script"]:
    """Run synthesis + screenwriter only. Returns (overview, script) for user to confirm before image gen."""
    overview = _run_synthesis(context, chat_model)
    script = _run_screenwriter(overview, context, chat_model)
    return overview, script


def run_storyboard_agent(
    anchor_image_paths: list[str],
    output_dir: str,
    chat_model: str = "gemini-2.0-flash",
    image_model: str = "gemini-3.1-flash-image-preview",
    context: "MovieContext | None" = None,
    chat_history: list[tuple[str, str]] | None = None,
    vertical: bool = True,
    generate_images: bool = True,
) -> str:
    """
    Create storyboard (storyboard.json + storyboard.html) from anchor images and context.
    Uses anchor[i] for scene i, or anchor[0] for all if single anchor.
    Returns path to storyboard.html.
    """
    import json

    from models import MovieContext, MovieOverview, SceneWithImage, Storyboard
    from renderer import render_storyboard_html

    if context is None and chat_history:
        context = extract_video_brief(chat_history, chat_model)
    if context is None:
        context = MovieContext(
            narrative_goal="A day in the life",
            audience="Gen Z",
            characters="One person",
            vibe="Casual",
            video_style="social_media",
            num_scenes=4,
        )

    overview = _run_synthesis(context, chat_model)
    script = _run_screenwriter(overview, context, chat_model)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    anchors = [Path(p).resolve() for p in anchor_image_paths]
    if not anchors:
        raise ValueError("At least one anchor image required")

    aspect = "9:16" if vertical else "16:9"
    scenes_with_images: list[SceneWithImage] = []

    for i, scene in enumerate(script.scenes):
        anchor = anchors[i % len(anchors)]
        out_name = f"scene_{scene.scene_num:02d}.png"
        out_path = str(Path(output_dir) / out_name)

        if generate_images:
            print(f"  Generating scene {scene.scene_num}/{len(script.scenes)}...", flush=True)
            create_anchor_image_with_gemini(
                prompt_text=scene.visual_prompt,
                reference_image_path=str(anchor),
                output_path=out_path,
                aspect_ratio=aspect,
                model=image_model,
            )

        scenes_with_images.append(
            SceneWithImage(
                scene_num=scene.scene_num,
                visual_prompt=scene.visual_prompt,
                narration=scene.narration,
                duration=scene.duration,
                camera_angle=scene.camera_angle,
                image_path=out_path if generate_images else "",
            )
        )

    storyboard = Storyboard(
        logline=overview.logline,
        world_building=overview.world_building,
        scenes=scenes_with_images,
    )
    html_path = str(Path(output_dir) / "storyboard.html")
    render_storyboard_html(storyboard, html_path)

    payload = {
        "logline": storyboard.logline,
        "world_building": storyboard.world_building,
        "anchor_image_path": str(anchors[0]),
        "vertical": vertical,
        "narrator": getattr(context, "narrator", "") or "",
        "narrator_speaking_style": getattr(context, "narrator_speaking_style", "") or "",
        "scenes": [
            {
                "scene_num": s.scene_num,
                "visual_prompt": s.visual_prompt,
                "narration": s.narration,
                "duration": s.duration,
                "camera_angle": s.camera_angle,
                "image_path": _rel_path(s.image_path, Path(output_dir)) if s.image_path else "",
            }
            for s in storyboard.scenes
        ],
    }
    json_path = str(Path(output_dir) / "storyboard.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return html_path


def update_storyboard_scene(
    project_dir: str | Path,
    scene_num: int,
    *,
    new_image_prompt: str | None = None,
    image_file_path: str | None = None,
    narration: str | None = None,
    image_model: str = "gemini-3.1-flash-image-preview",
    vertical: bool = True,
) -> None:
    """
    Update a single scene's image or narration in storyboard.json.
    - narration: new narration text
    - new_image_prompt: new visual description to generate image from (uses anchor)
    - image_file_path: path to existing image file to use instead (copy into project)
    """
    import json
    import shutil

    project_dir = Path(project_dir)
    json_path = project_dir / "storyboard.json"
    if not json_path.exists():
        raise FileNotFoundError(f"storyboard.json not found in {project_dir}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    scenes = data["scenes"]
    scene = next((s for s in scenes if s["scene_num"] == scene_num), None)
    if not scene:
        raise ValueError(f"Scene {scene_num} not found")

    if narration is not None:
        scene["narration"] = narration

    if new_image_prompt is not None:
        anchor = data.get("anchor_image_path", "")
        if not anchor or not Path(anchor).exists():
            for base in (project_dir.parent, Path.cwd()):
                cand = base / str(anchor)
                if cand.exists():
                    anchor = str(cand)
                    break
        out_name = f"scene_{scene_num:02d}.png"
        out_path = project_dir / out_name
        create_anchor_image_with_gemini(
            prompt_text=new_image_prompt,
            reference_image_path=anchor,
            output_path=str(out_path),
            aspect_ratio="9:16" if vertical else "16:9",
            model=image_model,
        )
        scene["image_path"] = out_name
    elif image_file_path is not None:
        src = Path(image_file_path)
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {image_file_path}")
        out_name = f"scene_{scene_num:02d}.png"
        out_path = project_dir / out_name
        shutil.copy2(str(src), str(out_path))
        scene["image_path"] = out_name

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    from renderer import render_storyboard_html
    from models import Storyboard, SceneWithImage

    out_dir = project_dir.resolve()
    storyboard = Storyboard(
        logline=data["logline"],
        world_building=data["world_building"],
        scenes=[
            SceneWithImage(
                scene_num=s["scene_num"],
                visual_prompt=s["visual_prompt"],
                narration=s["narration"],
                duration=s["duration"],
                camera_angle=s["camera_angle"],
                image_path=str(out_dir / s.get("image_path", "")) if s.get("image_path") else "",
            )
            for s in scenes
        ],
    )
    render_storyboard_html(storyboard, str(project_dir / "storyboard.html"))


def run_narration_agent(
    project_dir: str | Path,
    *,
    provider: str = "elevenlabs",
    voice_id: str | None = None,
    model_id: str | None = None,
    voice_name: str = "Kore",
    tts_model: str | None = None,
) -> None:
    """
    Generate narration audio for each scene. Saves scene_01_narration.mp3, etc.

    provider: "elevenlabs" or "gemini"
    - elevenlabs: needs voice_id, model_id (eleven_multilingual_v2, eleven_v3, etc.)
    - gemini: needs voice_name (Kore, Puck, etc.), tts_model (gemini-2.5-flash-preview-tts, etc.)
    """
    import json

    project_dir = Path(project_dir)
    json_path = project_dir / "storyboard.json"
    if not json_path.exists():
        raise FileNotFoundError(f"storyboard.json not found in {project_dir}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    scenes = data["scenes"]

    voice_id = voice_id or ELEVENLABS_VOICE_ID
    model_id = model_id or "eleven_multilingual_v2"
    tts_model = tts_model or "gemini-2.5-flash-preview-tts"

    for scene in scenes:
        n = scene["scene_num"]
        narration = scene.get("narration", "").strip()
        if not narration:
            continue
        out_path = project_dir / f"scene_{n:02d}_narration.mp3"
        if provider == "elevenlabs":
            if not voice_id:
                raise ValueError("voice_id required for ElevenLabs")
            generate_audio_with_elevenlabs(
                text=narration,
                voice_id=voice_id,
                model_id=model_id,
                output_path=str(out_path),
            )
        else:
            generate_audio_with_gemini_tts(
                text=narration,
                voice_name=voice_name,
                model=tts_model,
                output_path=str(out_path),
            )
        print(f"  Scene {n}: {out_path.name}")


def _runway_motion_prompt(visual_prompt: str, camera_angle: str, model_name: str) -> str:
    """Convert scene to Runway-safe camera/motion-only prompt."""
    from models import RunwayMotionPrompt
    from pydantic_ai import Agent
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return f"{camera_angle}. Subtle, natural movement."
    provider = GoogleProvider(api_key=api_key)
    model = GoogleModel(model_name, provider=provider)
    agent = Agent(
        model,
        deps_type=None,
        output_type=RunwayMotionPrompt,
        system_prompt=(
            "Convert scene description to SHORT motion prompt for Runway. "
            "Output ONLY camera angle, movement, lighting. NEVER include person/scene content."
        ),
    )
    result = agent.run_sync(
        f"Camera: {camera_angle}\nScene (context only): {visual_prompt}\nOutput motion prompt."
    )
    return result.output.motion_prompt.strip()[:100]


def run_video_agent(
    project_dir: str | Path,
    *,
    provider: str = "runway",
    model: str | None = None,
    chat_model: str = "gemini-2.0-flash",
) -> None:
    """
    Generate video clips from scene images. Saves scene_01.mp4, scene_02.mp4, etc.

    provider: "runway" or "veo"
    - runway: model options - gen3a_turbo, gen4_turbo, gen4.5, act_two, veo3, veo3.1, veo3.1_fast
    - veo: model options - veo-3.1-generate-preview, veo-3.1-fast-generate-preview
    """
    import json
    import time

    project_dir = Path(project_dir)
    json_path = project_dir / "storyboard.json"
    if not json_path.exists():
        raise FileNotFoundError(f"storyboard.json not found in {project_dir}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    scenes = data["scenes"]
    vertical = data.get("vertical", True)
    anchor = data.get("anchor_image_path", "")

    if provider == "runway":
        model = model or "gen4_turbo"
        ratio = "720:1280" if vertical else "1280:720"
        for scene in scenes:
            n = scene["scene_num"]
            img_path = project_dir / (scene.get("image_path") or f"scene_{n:02d}.png")
            if not img_path.exists():
                raise FileNotFoundError(f"Scene image not found: {img_path}")
            motion = _runway_motion_prompt(scene["visual_prompt"], scene.get("camera_angle", "Medium shot"), chat_model)
            out_path = project_dir / f"scene_{n:02d}.mp4"
            dur = min(10, max(5, int(scene.get("duration", 5))))
            result = create_image_to_video(
                prompt_text=motion,
                image_path=str(img_path),
                model=model,
                ratio=ratio,
                duration=dur,
                output_path=str(out_path),
            )
            print(f"  Scene {n}: {out_path.name}")
    else:
        model = model or "veo-3.1-generate-preview"
        aspect = "9:16" if vertical else "16:9"
        for scene in scenes:
            n = scene["scene_num"]
            img_path = project_dir / (scene.get("image_path") or f"scene_{n:02d}.png")
            if not img_path.exists():
                raise FileNotFoundError(f"Scene image not found: {img_path}")
            out_dir = str(project_dir)
            result = create_veo_video(
                prompt_text=scene.get("visual_prompt", "Subtle movement"),
                first_frame_image=str(img_path),
                model=model,
                aspect_ratio=aspect,
                duration_seconds=8,
                output_dir=out_dir,
            )
            out_path = Path(result["output_path"])
            target = project_dir / f"scene_{n:02d}.mp4"
            if out_path != target:
                import shutil
                shutil.move(str(out_path), str(target))
            print(f"  Scene {n}: {target.name}")


def run_merge_agent(project_dir: str | Path, output_filename: str = "movie_final.mp4") -> str:
    """
    Mux scene videos with narration audio and concatenate to final video.
    Returns path to movie_final.mp4.
    """
    import json
    import subprocess

    project_dir = Path(project_dir)
    json_path = project_dir / "storyboard.json"
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        scene_nums = [s["scene_num"] for s in data["scenes"]]
    else:
        scene_nums = []
        for p in sorted(project_dir.glob("scene_??.mp4")):
            try:
                n = int(p.stem.split("_")[1])
                scene_nums.append(n)
            except (IndexError, ValueError):
                pass
        scene_nums.sort()

    def get_media_duration(p: Path) -> float:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())

    def mux(video: Path, audio: Path, out: Path) -> Path:
        vd = get_media_duration(video)
        ad = get_media_duration(audio)
        if ad > vd:
            stretch = min(ad / vd * 1.02, 1.5)
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(video), "-i", str(audio),
                 "-filter:v", f"setpts=PTS*{stretch:.4f}", "-c:a", "aac",
                 "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(out)],
                check=True, capture_output=True,
            )
        else:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(video), "-i", str(audio),
                 "-c:v", "copy", "-c:a", "aac",
                 "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(out)],
                check=True, capture_output=True,
            )
        return out

    merged = []
    for n in scene_nums:
        video = project_dir / f"scene_{n:02d}.mp4"
        audio = project_dir / f"scene_{n:02d}_narration.mp3"
        out = project_dir / f"scene_{n:02d}_with_audio.mp4"
        if not video.exists() or not audio.exists():
            continue
        mux(video, audio, out)
        merged.append(out)

    if not merged:
        raise RuntimeError("No scenes to merge. Ensure scene_XX.mp4 and scene_XX_narration.mp3 exist.")

    final = project_dir / output_filename
    list_file = project_dir / "concat_list.txt"
    with open(list_file, "w") as f:
        for p in merged:
            f.write(f"file '{p.resolve()}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c", "copy", str(final)],
        check=True, capture_output=True,
    )
    # On Windows, ffmpeg may hold the file handle briefly—retry or ignore
    import time
    for _ in range(3):
        try:
            list_file.unlink(missing_ok=True)
            break
        except PermissionError:
            time.sleep(0.5)
    for p in merged:
        try:
            p.unlink(missing_ok=True)
        except PermissionError:
            pass
    return str(final)
