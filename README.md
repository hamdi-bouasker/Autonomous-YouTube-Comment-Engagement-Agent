# Autonomous-YouTube-Comment-Engagement-Agent
A production-grade, containerized AI agent that monitors YouTube video comments in real-time, leverages Google Gemini AI to analyze viewer sentiment and draft context-aware responses, and routes drafts to Telegram for a 1-click human-in-the-loop approval workflow before publishing live via the YouTube Data API v3.

![Intelligent_YouTube_AI_Engagement_Agent.jpg](https://github.com/hamdi-bouasker/Autonomous-YouTube-Comment-Engagement-Agent/blob/master/Intelligent_YouTube_AI_Engagement_Agent.jpg)

## 🌟 Key Features
🧠 AI-Powered Sentiment & Response Generation: Uses Google Gemini (gemini-3.8-flash with automatic fallback to gemini-3.5-flash-lite during high-demand spikes) enforced via strict Pydantic validation schemas.

## 🛡️ Smart Filtering: 
Automatically detects and filters out spam, trolls, and negative abuse without bothering your Telegram chat.

## 👥 Human-in-the-Loop Workflow: 
Sends rich interactive approval cards directly to your Telegram chat with [✅ Approve & Post] and [❌ Reject] inline buttons.

## 📺 Video Context Injection: 
Scrapes the target video's subtitles/transcript using youtube-transcript-api to ensure Gemini's draft replies are deeply contextual and accurate.

##  Rate-Limit Resilient: 
Features automatic backoff retries, local comment deduplication caching (processed_comments.json), and a 10-second delay between Telegram notification batches to prevent flood limits (429 Too Many Requests).

## 🐳 Dockerized Architecture: 
Built for seamless 24/7 background execution using Docker and Docker Compose with persistent volume mapping.

## 🏗️ Architecture & Tech Stack

![master/project_directory_structure.jpg](https://github.com/hamdi-bouasker/Autonomous-YouTube-Comment-Engagement-Agent/blob/master/project_directory_structure.jpg)

## Technology Ecosystem & Why We Use Them:

- Python 3.11+: Core runtime for asynchronous execution and data handling.

- Google GenAI SDK (google-genai): Interfaces directly with Gemini models for structured AI analysis.

- YouTube Data API v3 & OAuth 2.0: Securely fetches comment threads and posts approved replies under your channel.

- youtube-transcript-api: Lightweight caption scraping to feed video context into the AI prompt.

- python-telegram-bot: Asynchronous framework handling polling, interactive callbacks, and background job scheduling (JobQueue).

- Pydantic: Guarantees predictable JSON outputs from Gemini.

- Docker & Docker Compose: Isolates dependencies and keeps the bot running continuously in the background.


## 🛠️ Step-by-Step Installation & Setup

### Step 1: Clone the Repository & Setup Environment

```
git clone https://github.com/your-username/yt-gemini-bot.git
cd yt-gemini-bot
mkdir data
```

### Step 2: Configure Environment Variables (.env)

Create a .env file in your root directory and fill in your actual keys:

```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_personal_numeric_chat_id_here
GEMINI_API_KEY=your_google_gemini_api_key_here
YOUTUBE_VIDEO_ID=your_target_youtube_video_id_here
SCAN_INTERVAL_SECONDS=900
```

#### How to Obtain Your Credentials:

1. TELEGRAM_BOT_TOKEN: Message @BotFather on Telegram, send /newbot, and follow the prompts to get your HTTP API token.

2. TELEGRAM_CHAT_ID: Message @userinfobot on Telegram to get your unique numeric user ID. (Note: Make sure to open a direct chat with your bot and press /start first so it is permitted to message you).

3. GEMINI_API_KEY: Generate your API key from Google AI Studio.

4. YOUTUBE_VIDEO_ID: Extract the identifier from your target video URL (e.g., for [https://youtube.com/watch?v=n4iEI-7zHM0](https://youtube.com/watch?v=n4iEI-7zHM0), the ID is n4iEI-7zHM0).

### Step 3: YouTube API OAuth Setup (client_secret.json & token.json)

Because automated posting requires write access on behalf of your channel:

1. Go to the Google Cloud Console, create a project, and enable the YouTube Data API v3.

2. Configure your OAuth consent screen (External).

3. Create OAuth client ID credentials for a Desktop application.

4. Download the resulting credentials JSON file, rename it to client_secret.json, and place it inside your ./data/ folder.

5. Create and run the local OAuth generator script (generate_token.py) in your project root to authorize headlessly:

``` 
  import os
  from google_auth_oauthlib.flow import InstalledAppFlow

 SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

 def main():
    flow = InstalledAppFlow.from_client_secrets_file("data/client_secret.json", SCOPES)
    creds = flow.run_local_server(port=0)
    
    os.makedirs("data", exist_ok=True)
    with open("data/token.json", "w") as token:
        token.write(creds.to_json())
    print("Success! data/token.json generated.")

if __name__ == "__main__":
    main() 
```

#### Run it locally:

1. pip install google-auth-oauthlib
2. python generate_token.py

Sign in with your channel account in the pop-up browser window. Your token.json will be saved securely inside ./data/.

### Step 4: Python Dependencies (requirements.txt)

```
google-genai==1.3.0
google-api-python-client==2.160.0
google-auth-oauthlib==1.2.1
youtube-transcript-api==0.6.3
python-telegram-bot==21.10
pydantic==2.10.6
httpx==0.28.1
```

### Step 5: Core Application Script (main.py)

Place the production-ready script into your root directory as main.py (refer to the full implementation provided in the guide earlier).

```
import os
import time
import json
import logging
from pydantic import BaseModel
from google.api_core.exceptions import ResourceExhausted
from google.genai.errors import ServerError
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)
import google.genai as genai
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from telegram.error import BadRequest

# Configure Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Environment Configuration ---
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
YOUTUBE_VIDEO_ID = os.getenv("YOUTUBE_VIDEO_ID", "")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL_SECONDS", "900"))

# Initialize Gemini Client
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# In-memory storage for pending drafts and processed comment tracking
PENDING_DRAFTS = {}
PROCESSED_COMMENTS_FILE = "/app/data/processed_comments.json"

def load_processed_comments() -> set:
    if os.path.exists(PROCESSED_COMMENTS_FILE):
        try:
            with open(PROCESSED_COMMENTS_FILE, "r") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()

def save_processed_comments(processed_set: set):
    os.makedirs("/app/data", exist_ok=True)
    try:
        with open(PROCESSED_COMMENTS_FILE, "w") as f:
            json.dump(list(processed_set), f)
    except Exception as e:
        logger.error(f"Failed to save processed comments cache: {e}")

PROCESSED_COMMENTS = load_processed_comments()

# --- YouTube API Authentication & Client ---
def get_youtube_client():
    token_path = "/app/data/token.json"
    if not os.path.exists(token_path):
        logger.error("YouTube token.json not found in ./data/! OAuth authentication required.")
        return None
    try:
        creds = Credentials.from_authorized_user_file(token_path, ["https://www.googleapis.com/auth/youtube.force-ssl"])
        return build("youtube", "v3", credentials=creds)
    except Exception as e:
        logger.error(f"Failed to build YouTube API client: {e}")
        return None

def fetch_real_youtube_comments(video_id: str):
    """Fetches ALL recent unreplied comment threads for the video."""
    youtube = get_youtube_client()
    if not youtube:
        return []
    
    try:
        # Fetch up to 100 recent comment threads for the video
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            order="time"
        ).execute()
        
        unreplied_comments = []
        
        for item in response.get("items", []):
            top_comment = item["snippet"]["topLevelComment"]
            comment_id = top_comment["id"]
            snippet = top_comment["snippet"]
            
            # 1. Skip if already processed in our persistent database
            if comment_id in PROCESSED_COMMENTS:
                continue

            # 2. Skip if YouTube itself reports existing replies on this thread
            total_replies = item["snippet"].get("totalReplyCount", 0)
            if total_replies > 0:
                # Mark locally so we don't query it again
                PROCESSED_COMMENTS.add(comment_id)
                save_processed_comments(PROCESSED_COMMENTS)
                continue
            
            # Clean text extracted from HTML format
            comment_text = snippet.get("textDisplay", "").strip()
            author_name = snippet.get("authorDisplayName", "Viewer")

            unreplied_comments.append({
                "id": comment_id,
                "author": author_name,
                "text": comment_text
            })

        logger.info(f"Found {len(unreplied_comments)} total unreplied comments needing review.")
        return unreplied_comments

    except Exception as e:
        logger.error(f"Error fetching comment threads from YouTube: {e}")
        return []

def post_youtube_reply(parent_id: str, reply_text: str):
    """Executes real YouTube API call to post a reply to a comment thread."""
    youtube = get_youtube_client()
    if not youtube:
        logger.error("Cannot post reply: YouTube client not authenticated.")
        return False
    
    try:
        youtube.comments().insert(
            part="snippet",
            body={
                "snippet": {
                    "parentId": parent_id,
                    "textOriginal": reply_text
                }
            }
        ).execute()
        logger.info(f"[YouTube API] Successfully posted reply to comment ID {parent_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to post YouTube reply via API: {e}")
        return False

# --- Gemini Analysis & Transcript Logic ---
class CommentAnalysis(BaseModel):
    should_reply: bool
    category: str  # "question", "positive_feedback", "spam", "troll"
    draft_reply: str

def get_video_context(video_id: str) -> str:
    if not video_id:
        return "No video ID configured."
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript[:50]])
    except (TranscriptsDisabled, NoTranscriptFound):
        logger.warning(f"Subtitles disabled for video {video_id}. Proceeding without transcript context.")
        return "No transcript available."
    except Exception as e:
        logger.warning(f"Could not retrieve transcript for {video_id}: {e}")
        return "No transcript available."

def analyze_and_draft(user_comment: str, video_context: str) -> CommentAnalysis:
    prompt = f"""
    Video Context: {video_context}
    User Comment: "{user_comment}"
    
    Task:
    1. Categorize the comment (question, positive_feedback, spam, or troll).
    2. Set `should_reply` to true ONLY if it is a genuine question or positive feedback.
    3. If `should_reply` is true, write a polite, concise reply using the video context.
    """
    
    models_to_try = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    
    for model_name in models_to_try:
        max_retries = 2
        backoff_factor = 3

        for attempt in range(max_retries):
            try:
                response = ai_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": CommentAnalysis,
                    }
                )
                return CommentAnalysis.model_validate_json(response.text)
                
            except (ResourceExhausted, ServerError) as e:
                if attempt == max_retries - 1:
                    break
                sleep_time = backoff_factor * (2 ** attempt)
                time.sleep(sleep_time)
                
    raise ServerError(503, {"message": "All fallback models experiencing high demand."})

# --- Core Scanner Logic ---
async def check_youtube_comments(context: ContextTypes.DEFAULT_TYPE):
    """Runs periodically via JobQueue or manually via /check to scan real comments."""
    logger.info("Scanning YouTube comments for new viewer messages...")
    if not YOUTUBE_VIDEO_ID:
        logger.warning("YOUTUBE_VIDEO_ID is not configured in .env!")
        return

    video_context = get_video_context(YOUTUBE_VIDEO_ID)
    real_comments = fetch_real_youtube_comments(YOUTUBE_VIDEO_ID)

    if not real_comments:
        logger.info("No new unreplied comments found.")
        return

    for index, item in enumerate(real_comments):
        comment_id = item["id"]
        
        # Skip if already processed or currently pending approval
        if comment_id in PROCESSED_COMMENTS or comment_id in PENDING_DRAFTS:
            continue

        try:
            analysis = analyze_and_draft(item["text"], video_context)
        except Exception as err:
            logger.error(f"Skipping comment {comment_id} due to Gemini API failure: {err}")
            continue
        
        if analysis and analysis.should_reply:
            PENDING_DRAFTS[comment_id] = {
                "author": item["author"],
                "comment": item["text"],
                "draft": analysis.draft_reply
            }

            message_text = (
                f"💬 *New YouTube Comment Approval Needed*\n\n"
                f"👤 *Author:* `{item['author']}`\n"
                f"📝 *Comment:* \"{item['text']}\"\n\n"
                f"🤖 *Gemini Draft Reply:*\n\"{analysis.draft_reply}\""
            )

            keyboard = [
                [
                    InlineKeyboardButton("✅ Approve & Post", callback_data=f"approve:{comment_id}"),
                    InlineKeyboardButton("❌ Reject", callback_data=f"reject:{comment_id}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await context.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

            # Wait 10 seconds before sending the next draft card to Telegram
            if index < len(real_comments) - 1:
                logger.info("Waiting 10 seconds before processing the next comment...")
                await asyncio.sleep(10)
        else:
            # NEW: If Gemini says should_reply is False (spam, troll, negative feedback), 
            # mark it as processed so the bot ignores it on future scans!
            logger.info(f"Skipping comment {comment_id} (Category: {analysis.category}) - no reply needed.")
            PROCESSED_COMMENTS.add(comment_id)
            save_processed_comments(PROCESSED_COMMENTS)

# --- Telegram Event Handlers ---
async def handle_approval_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    try:
        await query.answer()
    except BadRequest:
        return

    action, comment_id = query.data.split(":", 1)
    draft_info = PENDING_DRAFTS.get(comment_id)

    if not draft_info:
        try:
            await query.edit_message_text(
                text=f"{query.message.text}\n\n-----------------------------\n⚠️ *Draft expired or bot was restarted.*",
                parse_mode="Markdown"
            )
        except Exception:
            pass
        return

    if action == "approve":
        success = post_youtube_reply(comment_id, draft_info["draft"])
        if success:
            updated_text = (
                f"{query.message.text}\n\n"
                f"-----------------------------\n"
                f"✅ *APPROVED & POSTED LIVE TO YOUTUBE*"
            )
        else:
            updated_text = (
                f"{query.message.text}\n\n"
                f"-----------------------------\n"
                f"⚠️ *APPROVED, BUT YOUTUBE API POST FAILED (Check logs)*"
            )
            
        await query.edit_message_text(text=updated_text, parse_mode="Markdown")
        
        # Mark as permanently processed so it never triggers again
        PROCESSED_COMMENTS.add(comment_id)
        save_processed_comments(PROCESSED_COMMENTS)
        del PENDING_DRAFTS[comment_id]

    elif action == "reject":
        updated_text = (
            f"{query.message.text}\n\n"
            f"-----------------------------\n"
            f"❌ *REJECTED*"
        )
        await query.edit_message_text(text=updated_text, parse_mode="Markdown")
        
        # Mark as processed so it won't keep re-prompting
        PROCESSED_COMMENTS.add(comment_id)
        save_processed_comments(PROCESSED_COMMENTS)
        if comment_id in PENDING_DRAFTS:
            del PENDING_DRAFTS[comment_id]

async def manual_check_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔎 Triggering manual real-time YouTube comment scan...")
    try:
        await check_youtube_comments(context)
        await update.message.reply_text("✅ Scan completed.")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Scan error: {e}")

# --- Application Entry Point ---
def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    job_queue = app.job_queue
    job_queue.run_repeating(
        check_youtube_comments,
        interval=SCAN_INTERVAL,
        first=10
    )

    app.add_handler(CommandHandler("check", manual_check_command))
    app.add_handler(CallbackQueryHandler(handle_approval_buttons))

    logger.info(f"Bot started. Monitoring YouTube video ID: {YOUTUBE_VIDEO_ID}")
    app.run_polling()

if __name__ == "__main__":
    main()    
```

### 🐳 Containerization & Deployment

Build and spin up your stack instantly using Docker Compose:

`
docker compose up -d --build
`

#### Docker Configuration Files

##### Dockerfile:

```
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "main.py"]
```

##### docker-compose.yml:

```
services:
  yt-gemini-bot:
    build: .
    container_name: yt_telegram_gemini_bot
    restart: unless-stopped
    env_file:
      - .env
    volumes:
      # Mount data directory to persist OAuth tokens or database files if needed
      - ./data:/app/data
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    ```

  ### 📱 Telegram Commands & Usage

  - Background Polling: The bot automatically scans your configured YouTube video every 15 minutes (SCAN_INTERVAL_SECONDS).

  - Manual Scan (/check): Send /check directly to your Telegram bot at any time to trigger an instant real-time comment scan.

  - Interactive Approvals: Review the incoming cards in Telegram:

  - Click ✅ Approve & Post to publish the AI response instantly to YouTube.

  - Click ❌ Reject to archive and dismiss the comment.

 ### 📄 License & Credits

   Developed as an open-source automation utility for creators and engineers. Feel free to fork, modify, and adapt for your     own workflows!

