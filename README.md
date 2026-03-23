# Agentic Video Maker

Create short narrated videos with AI. An interactive pipeline guides you through your idea, generates a storyboard, then produces narration-only videos. Run: `python main.py`.

---

## Agentic Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. DISCOVERY (Jarvis)                                                   │
│     Chat about your idea, character, narrator, style, scenes.              │
│     Optionally reviews your anchor images first.                         │
│     Type "done" or "proceed with storyboard" when ready.                │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  2. OUTLINE CONFIRMATION                                                 │
│     Logline + scene list shown. Confirm, or say what to change          │
│     (e.g. "2 scenes", "3rd person narration").                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  3. STORYBOARD                                                          │
│     Synthesis → Screenwriter → Gemini image gen per scene.              │
│     Output: storyboard.json, storyboard.html, scene_01.png, ...          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  4. EDIT LOOP                                                           │
│     Adjust narration or images per scene. Type "like" when done.        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  5. NARRATION AGENT                                                      │
│     ElevenLabs or Gemini TTS → scene_01_narration.mp3, ...              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  6. VIDEO AGENT                                                         │
│     Runway or Veo → scene_01.mp4, scene_02.mp4, ...                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  7. MERGE AGENT                                                         │
│     Mux video + narration → movie_final.mp4. Option to play when done.  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Runway Credit Costs

Runway charges credits per second of video (1 credit ≈ $0.01). Set `VIDEO_PROVIDER` and `RUNWAY_MODEL` in `main.py`.

| Model      | Credits/sec | 3 scenes × 5 sec | 4 scenes × 5 sec |
|------------|-------------|------------------|------------------|
| gen4_turbo | 5           | 75 (~$0.75)      | 100 (~$1.00)     |
| gen4.5     | 12          | 180 (~$1.80)     | 240 (~$2.40)     |
| act_two    | 5           | 75 (~$0.75)      | 100 (~$1.00)     |
| veo3.1_fast| 15          | 225 (~$2.25)     | 300 (~$3.00)     |
| veo3 / veo3.1 | 40       | 600 (~$6.00)    | 800 (~$8.00)     |

See [Runway docs](https://docs.dev.runwayml.com/guides/models) for current pricing.

---

## Runway Account & API Key

If you use Runway for video generation, create an account and API key:

1. **Sign up** at [dev.runwayml.com](https://dev.runwayml.com/)
2. **Create an organization** when prompted (this holds your API keys)
3. **Create a key** – Go to the **API Keys** tab → Create new key → give it a name
4. **Copy the key immediately** – it’s shown only once; if lost, disable it and create a new one
5. **Add credits** – Go to the **Billing** tab and add credits (minimum $10, at $0.01/credit)

Add the key to your `.env` as `RUNWAY_API_KEY=key_xxxxx`. See [Runway setup docs](https://docs.dev.runwayml.com/guides/setup) for details.

---

## Setup

```bash
pip install -r requirements.txt
```

Install **ffmpeg** (required for merging video and audio).

Create a `.env` file:

```
GEMINI_API_KEY=your-gemini-api-key
RUNWAY_API_KEY=key_xxxxx
ELEVENLABS_API_KEY=sk_xxxxx
ELEVENLABS_VOICE_ID=your_voice_id
```

| Key                  | Purpose                  |
|----------------------|--------------------------|
| `GEMINI_API_KEY`     | Chat, images, optional TTS |
| `RUNWAY_API_KEY`     | Video generation         |
| `ELEVENLABS_API_KEY` | Narration/TTS only (no lipsync) |
| `ELEVENLABS_VOICE_ID`| Narration voice          |

Place anchor images in `anchor_images/` and set `ANCHOR_IMAGE_PATHS_FALLBACK` in `main.py`, or let Jarvis ask for paths in chat.

---

## Usage

```bash
python main.py
```

Edit the CONFIG in `main.py`:
- **IDEA** – Skip chat; extract from this text
- **SKIP_DISCOVERY** – Use default context
- **PROJECT_DIR** – Resume from existing project (edit loop only)
- **ANCHOR_IMAGE_PATHS_FALLBACK** – Anchor images when not in chat

Output: `projects/movie_YYYY-MM-DD_HHMMSS/` with `storyboard.json`, `storyboard.html`, scene images, scene videos, and `movie_final.mp4`.
