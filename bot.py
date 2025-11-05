import os
import tempfile
from telegram import Update, Voice
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import openai
import requests

# 🔐 Environment
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_CHAT = os.getenv("ADMIN_CHAT")
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")
openai.api_key = OPENAI_API_KEY

# 🧠 Memory + GPT-5
embedding_model = OpenAIEmbeddings()
docs = [Document(page_content="initial memory", metadata={"source": "bootstrap"})]
memory_db = FAISS.from_documents(docs, embedding_model)
llm = ChatOpenAI(model="gpt-5", temperature=0.3)

platforms = {}
plugin_scores = {}
task_log = []
auto_learn_enabled = False
auto_retrain_enabled = False
approved_plugins = set()
blocked_plugins = set()

def is_admin(chat_id):
    return str(chat_id) == str(ADMIN_CHAT)

# 🧠 Command routing via GPT-5
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_chat.id): return
    user_input = update.message.text
    prompt = f"""
You are Animous Core's command interpreter. Given a user message, map it to one of these actions:

- /platforms
- /recall [platform]
- /score [platform]
- /earnings
- /tasks
- /learn_auto on/off
- /retrain_auto on/off
- /reset
- /status

Respond ONLY with the command string.
User message: "{user_input}"
"""
    command = llm.predict(prompt).strip()
    update.message.text = command
    await application.process_update(update)

# 🎙️ Voice transcription
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_chat.id): return
    voice: Voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        file_path = f.name
        await file.download_to_drive(file_path)
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    update.message.text = transcript["text"]
    await handle_text(update, context)

# 🔧 Command handlers
async def cmd_platforms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🧠 Learned Platforms:\n" + "\n".join(f"- {n}" for n in platforms))

async def cmd_recall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = " ".join(context.args)
    plugin = platforms.get(name)
    await update.message.reply_text(f"🧠 Plugin for {name}:\n{plugin}" if plugin else "❌ Platform not found.")

async def cmd_score(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = " ".join(context.args)
    score = plugin_scores.get(name)
    if not score:
        await update.message.reply_text("❌ No score found.")
        return
    await update.message.reply_text(f"📊 Score for {name}:\n✅ {score['success']} | ❌ {score['fail']} | 💰 ${score['earned']}")

async def cmd_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = "📋 Recent Tasks:\n" + "\n".join(f"- {t['title']} (${t['pay']}) from {t['platform']}" for t in task_log[-10:])
    await update.message.reply_text(msg if task_log else "📭 No tasks completed yet.")

async def cmd_earnings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total = sum(t["pay"] for t in task_log if t["status"] == "completed")
    await update.message.reply_text(f"💰 Total earned: ${total}")

async def cmd_learn_auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_learn_enabled
    arg = " ".join(context.args).lower()
    auto_learn_enabled = arg == "on"
    await update.message.reply_text(f"🧠 Auto-learning is now {'ON' if auto_learn_enabled else 'OFF'}.")

async def cmd_retrain_auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_retrain_enabled
    arg = " ".join(context.args).lower()
    auto_retrain_enabled = arg == "on"
    await update.message.reply_text(f"🔁 Auto-retraining is now {'ON' if auto_retrain_enabled else 'OFF'}.")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = f"""🤖 Animous Core Status:
- Auto-learning: {"ON" if auto_learn_enabled else "OFF"}
- Auto-retraining: {"ON" if auto_retrain_enabled else "OFF"}
- Plugins learned: {len(platforms)}
- Wallet: {WALLET_ADDRESS}
"""
    await update.message.reply_text(msg)

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    plugin_scores.clear()
    platforms.clear()
    await update.message.reply_text("🔄 Memory and scores reset.")

# 🚀 Launch bot
application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
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

application.run_polling()
