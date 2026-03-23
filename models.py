"""
Pydantic models for the AI Movie Agentic Pipeline.
"""

from pydantic import BaseModel, Field


# Video style options for storyboard agent
VIDEO_STYLES = ("social_media", "corporate_ad", "political_campaign", "informational")


class MovieContext(BaseModel):
    """Context gathered during Discovery phase - enough to produce a storyboard."""

    narrative_goal: str = Field(description="What the story is about or aims to achieve")
    audience: str = Field(description="Target audience (e.g. Gen Z, teens, young adults)")
    characters: str = Field(description="Main characters or subjects")
    vibe: str = Field(description="Mood, tone, aesthetic (e.g. dreamy, chaotic, nostalgic)")
    vertical: bool = Field(
        default=True,
        description="True for vertical (9:16), False for normal (16:9 landscape)",
    )
    # Extended fields for agentic video tool
    video_style: str = Field(
        default="social_media",
        description="Style: social_media, corporate_ad, political_campaign, informational",
    )
    num_scenes: int = Field(default=4, ge=1, le=10, description="Number of scenes")
    outfits_per_scene: list[str] = Field(
        default_factory=list,
        description="Outfit description per scene (e.g. ['beige hoodie, grey cap'] for consistency)",
    )
    anchor_image_paths: list[str] = Field(
        default_factory=list,
        description="Paths to anchor/reference images for character consistency (one per scene or one for all)",
    )
    narrator: str = Field(
        default="",
        description="Who is the narrator (e.g. Lisa, a young professional, the host)",
    )
    narrator_speaking_style: str = Field(
        default="",
        description="Speaking style: casual, punchy, formal, conversational, energetic, calm, etc.",
    )


class MovieOverview(BaseModel):
    """Creative synthesis from chat history - logline and world-building."""

    logline: str = Field(description="One-sentence summary of the story")
    world_building: str = Field(
        description="Setting, atmosphere, visual style, and world details"
    )


class Scene(BaseModel):
    """A single scene in the storyboard."""

    scene_num: int = Field(ge=1, description="Scene number (1-indexed)")
    visual_prompt: str = Field(
        description="Detailed description for image generation (what to show visually)"
    )
    narration: str = Field(
        description="Voice-over or narration text (keep short for TikTok)"
    )
    duration: float = Field(
        ge=2.0, le=10.0,
        description="Duration in seconds (2-10 per scene)"
    )
    camera_angle: str = Field(
        description="Camera angle or shot type (e.g. close-up, wide shot, over-shoulder)"
    )


class Script(BaseModel):
    """Full script for a short video."""

    scenes: list[Scene] = Field(
        min_length=1, max_length=10,
        description="Scenes (count matches user's num_scenes). Total duration under 30 seconds."
    )


class SceneWithImage(Scene):
    """Scene with generated image path attached."""

    image_path: str = Field(description="Path to the generated scene image")


class Storyboard(BaseModel):
    """Final storyboard with all scenes and their generated images."""

    logline: str = Field(description="One-sentence story summary")
    world_building: str = Field(description="World and style context")
    scenes: list[SceneWithImage] = Field(description="Scenes with embedded images")


# ---------------------------------------------------------------------------
# Video Template - learned style from a reference video for emulation
# ---------------------------------------------------------------------------


class VideoTemplate(BaseModel):
    """
    A reusable template describing the visual and audio style of a reference video.
    Used with an anchor image + topic to generate a new video script in this style.
    """

    name: str = Field(
        default="",
        description="Short identifier for this template (e.g. 'miami_bro_style')",
    )

    # Format
    vertical: bool = Field(
        default=True,
        description="True for 9:16 vertical, False for 16:9 landscape",
    )
    total_duration_seconds: float = Field(
        ge=5.0, le=120.0,
        description="Typical total duration in seconds",
    )

    # Visual style
    composition: str = Field(
        description="Subject framing: e.g. 'single person talking head', 'close-up face', 'medium shot'",
    )
    camera_angles: list[str] = Field(
        default_factory=list,
        description="Typical shot types used: close-up, medium, wide, etc.",
    )
    lighting: str = Field(
        description="Lighting style: natural, soft, harsh, studio, etc.",
    )
    background: str = Field(
        description="Typical background or setting (e.g. indoors casual, outdoor, plain wall)",
    )
    visual_style_summary: str = Field(
        description="One-paragraph summary of the overall visual aesthetic for image generation",
    )

    # Speaking / audio style
    speaking_style: str = Field(
        description="Tone and energy: casual, energetic, conversational, direct-to-camera, etc.",
    )
    pacing: str = Field(
        description="Pacing: fast cuts, single take, slow, punchy, etc.",
    )
    narration_guidance: str = Field(
        description="How narration should sound: short punchy sentences, casual slang, etc.",
    )

    # Content structure
    content_structure: str = Field(
        description="How the video is organized: single take, direct monologue, cuts between scenes, etc.",
    )
    segment_count: int = Field(
        ge=1, le=20,
        default=1,
        description="Typical number of distinct segments or shots",
    )

    # Reference transcript (for style, not content)
    sample_transcript: str = Field(
        default="",
        description="Transcribed text from the reference video for style/tone reference",
    )


class RunwayMotionPrompt(BaseModel):
    """
    Camera/motion-only prompt for Runway image-to-video.
    NO subject, person, or scene content - only camera angle, movement, and style.
    """

    motion_prompt: str = Field(
        max_length=100,
        description="2-15 words. Camera angle, movement, lighting style only. No person/scene details.",
    )


class TemplateStoryboardPlan(BaseModel):
    """
    Plan for a storyboard derived from a VideoTemplate + topic.
    Used by storyboard_from_template agent.
    """

    logline: str = Field(description="One-sentence summary of the video about the topic")
    world_building: str = Field(
        description="Setting, atmosphere, visual style from the template"
    )
    scenes: list[Scene] = Field(
        min_length=3, max_length=5,
        description="3-5 scenes. Total duration under 30 seconds. Match template style.",
    )
