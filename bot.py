# bot.py
import os
import tempfile
import traceback
import requests
from telegram import Update, Voice, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    CallbackQueryHandler,
)

# provider helpers (unified adapter)
from llm_utils import provider_generate, provider_transcribe

# Environment
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ADMIN_CHAT = os.getenv("ADMIN_CHAT", "")
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS", "")
BOT_PROXY_KEY = os.getenv("BOT_PROXY_KEY", "")  # optional header shared with proxy

# In-memory state
platforms = {}
plugin_scores = {}
task_log = []
auto_learn_enabled = False
auto_retrain_enabled = False
approved_plugins = set()
blocked_plugins = set()


def is_admin(chat_id):
    return str(chat_id) == str(ADMIN_CHAT)


# --- UI and routing handlers

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_chat.id):
        return
    keyboard = [
        [
            InlineKeyboardButton("🧠 Learned Plugins", callback_data="platforms"),
            InlineKeyboardButton("🔍 Inspect Plugin", callback_data="recall"),
        ],
        [
            InlineKeyboardButton("📊 Scorecards", callback_data="score"),
            InlineKeyboardButton("📋 Recent Tasks", callback_data="tasks"),
        ],
        [
            InlineKeyboardButton("💰 Earnings", callback_data="earnings"),
            InlineKeyboardButton("🔁 Auto Learn", callback_data="learn_auto_toggle"),
        ],
        [
            InlineKeyboardButton("🔄 Retrain", callback_data="retrain_auto_toggle"),
            InlineKeyboardButton("🏛️ System Status", callback_data="status"),
        ],
        [InlineKeyboardButton("🧹 Reset Memory", callback_data="reset")],
    ]
    reply = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("AC LIVE COMMANDS", reply_markup=reply)


async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    cmd = query.data
    global auto_learn_enabled, auto_retrain_enabled

    if cmd == "learn_auto_toggle":
        auto_learn_enabled = not auto_learn_enabled
        await query.message.reply_text(f"Auto-learning {'ON' if auto_learn_enabled else 'OFF'}")
        return

    if cmd == "retrain_auto_toggle":
        auto_retrain_enabled = not auto_retrain_enabled
        await query.message.reply_text(f"Auto-retraining {'ON' if auto_retrain_enabled else 'OFF'}")
        return

    mapping = {
        "platforms": "/platforms",
        "recall": "/recall",
        "score": "/score",
        "tasks": "/tasks",
        "earnings": "/earnings",
        "status": "/status",
        "reset": "/reset",
    }
    fake_text = mapping.get(cmd, "")
    if fake_text:
        class FakeMessage:
            def __init__(self, text, chat_id, reply_fn):
                self.text = text
                self.chat = type("C", (), {"id": chat_id})
                self.reply_text = reply_fn
                self.from_user = query.from_user

        fake_update = type("U", (), {"message": FakeMessage(fake_text, query.message.chat.id, query.message.reply_text)})
        await handle_text(fake_update, context)


# --- Natural language routing via the configured provider

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        chat = getattr(update, "message", None)
        if not chat:
            return
        cid = getattr(chat.chat, "id", None)
        if not is_admin(cid):
            return

        user_input = chat.text or ""
        # Use the provider to interpret free text commands reliably
        prompt = (
            "You are Animous Core's command interpreter. Map the user message to one of these exact commands:\n"
            "- /platforms\n- /recall [platform]\n- /score [platform]\n- /earnings\n- /tasks\n- /learn_auto on/off\n- /retrain_auto on/off\n- /reset\n- /status\n\n"
            f"Respond ONLY with the command string (e.g., \"/platforms\" or \"/recall LabelNet\").\nUser message: \"{user_input}\""
        )
        command = provider_generate(prompt, max_tokens=120, temperature=0.0).strip()
        parts = command.split()
        cmd = parts[0] if parts else ""
        args = parts[1:]

        class Ctx:
            args = args

        if cmd == "/platforms":
            await cmd_platforms(update, Ctx)
        elif cmd == "/recall":
            chat.args = args
            await cmd_recall(update, Ctx)
        elif cmd == "/score":
            chat.args = args
            await cmd_score(update, Ctx)
        elif cmd == "/earnings":
            await cmd_earnings(update, Ctx)
        elif cmd == "/tasks":
            await cmd_tasks(update, Ctx)
        elif cmd == "/learn_auto":
            chat.args = args
            await cmd_learn_auto(update, Ctx)
        elif cmd == "/retrain_auto":
            chat.args = args
            await cmd_retrain_auto(update, Ctx)
        elif cmd == "/reset":
            await cmd_reset(update, Ctx)
        elif cmd == "/status":
            await cmd_status(update, Ctx)
        else:
            await update.message.reply_text("Unknown or unsupported command interpretation.")
    except Exception:
        traceback.print_exc()
        await update.message.reply_text("Command interpretation failed.")


# --- Voice handling (transcription via provider or proxy)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not is_admin(update.message.chat.id):
            return

        # Fetch the voice file
        voice: Voice = update.message.voice
        file = await context.bot.get_file(voice.file_id)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_path = f.name
            await file.download_to_drive(file_path)

        # Read bytes and call provider_transcribe (supports proxy or direct providers)
        with open(file_path, "rb") as af:
            audio_bytes = af.read()

        transcript = provider_transcribe(audio_bytes, filename=os.path.basename(file_path))
        if transcript:
            # Put transcript into message.text and re-run text handler for consistent flow
            update.message.text = transcript
            await handle_text(update, context)
        else:
            await update.message.reply_text("Transcription returned no text.")
    except Exception:
        traceback.print_exc()
        await update.message.reply_text("Transcription failed.")


# --- Command handlers (admin-protected)

async def cmd_platforms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.message.chat.id):
        return
    text = "🧠 Learned Platforms:\n" + ("\n".join(f"- {n}" for n in platforms) or "No platforms yet.")
    await update.message.reply_text(text)


async def cmd_recall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.message.chat.id):
        return
    name = " ".join(getattr(update.message, "args", []))
    plugin = platforms.get(name)
    await update.message.reply_text(f"🧠 Plugin for {name}:\n{plugin}" if plugin else "❌ Platform not found.")


async def cmd_score(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.message.chat.id):
        return
    name = " ".join(getattr(update.message, "args", []))
    score = plugin_scores.get(name)
    if not score:
        await update.message.reply_text("❌ No score found.")
        return
    await update.message.reply_text(f"📊 Score for {name}:\n✅ {score['success']} | ❌ {score['fail']} | 💰 ${score['earned']}")


async def cmd_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.message.chat.id):
        return
    msg = "📋 Recent Tasks:\n" + ("\n".join(f"- {t['title']} (${t.get('pay', 0)}) from {t['platform']}" for t in task_log[-10:]) or "📭 No tasks completed yet.")
    await update.message.reply_text(msg)


async def cmd_earnings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.message.chat.id):
        return
    total = sum(t.get("pay", 0) for t in task_log if t.get("status") == "completed")
    await update.message.reply_text(f"💰 Total earned: ${total}")


async def cmd_learn_auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.message.chat.id):
        return
    global auto_learn_enabled
    arg = " ".join(getattr(update.message, "args", [])).lower()
    auto_learn_enabled = arg == "on" if arg else not auto_learn_enabled
    await update.message.reply_text(f"🧠 Auto-learning is now {'ON' if auto_learn_enabled else 'OFF'}.")


async def cmd_retrain_auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.message.chat.id):
        return
    global auto_retrain_enabled
    arg = " ".join(getattr(update.message, "args", [])).lower()
    auto_retrain_enabled = arg == "on" if arg else not auto_retrain_enabled
    await update.message.reply_text(f"🔁 Auto-retraining is now {'ON' if auto_retrain_enabled else 'OFF'}.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.message.chat.id):
        return
    msg = (
        f"🏛️ Animous Core Status:\n- Auto-learning: {'ON' if auto_learn_enabled else 'OFF'}\n"
        f"- Auto-retraining: {'ON' if auto_retrain_enabled else 'OFF'}\n- Plugins learned: {len(platforms)}\n- Wallet: {WALLET_ADDRESS}"
    )
    await update.message.reply_text(msg)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.message.chat.id):
        return
    plugin_scores.clear()
    platforms.clear()
    await update.message.reply_text("🔄 Memory and scores reset.")


# --- Application setup

def build_application():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("menu", cmd_menu))
    application.add_handler(CallbackQueryHandler(callback_router))
    application.add_handler(CommandHandler("platforms", cmd_platforms))
    application.add_handler(CommandHandler("recall", cmd_recall))
    application.add_handler(CommandHandler("score", cmd_score))
    application.add_handler(CommandHandler("tasks", cmd_tasks))
    application.add_handler(CommandHandler("earnings", cmd_earnings))
    application.add_handler(CommandHandler("learn_auto", cmd_learn_auto))
    application.add_handler(CommandHandler("retrain_auto", cmd_retrain_auto))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("reset", cmd_reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    return application


if __name__ == "__main__":
    app = build_application()
    app.run_polling()
