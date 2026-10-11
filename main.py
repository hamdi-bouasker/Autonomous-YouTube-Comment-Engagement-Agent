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