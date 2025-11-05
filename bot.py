import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from memory import platforms, plugin_scores
from executor import task_log

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_CHAT = os.getenv("ADMIN_CHAT")
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")

auto_learn_enabled = False
auto_retrain_enabled = False
approved_plugins = set()
blocked_plugins = set()

def is_admin(chat_id):
    return str(chat_id) == str(ADMIN_CHAT)

async def cmd_platforms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_chat.id): return
    await update.message.reply_text("🧠 Learned Platforms:\n" + "\n".join(f"- {n}" for n in platforms))

async def cmd_recall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_chat.id): return
    name = " ".join(context.args)
    plugin = platforms.get(name)
    await update.message.reply_text(f"🧠 Plugin for {name}:\n{plugin}" if plugin else "❌ Platform not found.")

async def cmd_approve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    approved_plugins.add(" ".join(context.args))
    await update.message.reply_text(f"✅ Approved plugin: {' '.join(context.args)}")

async def cmd_block(update: Update, context: ContextTypes.DEFAULT_TYPE):
    blocked_plugins.add(" ".join(context.args))
    await update.message.reply_text(f"🚫 Blocked plugin: {' '.join(context.args)}")

async def cmd_score(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = " ".join(context.args)
    score = plugin_scores.get(name)
    if not score:
        await update.message.reply_text("❌ No score found.")
        return
    await update.message.reply_text(f"📊 Score for {name}:\n✅ {score['success']} | ❌ {score['fail']} | 💰 ${score['earned']}")

async def cmd_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_chat.id): return
    msg = "📋 Recent Tasks:\n" + "\n".join(f"- {t['title']} (${t['pay']}) from {t['platform']}" for t in task_log[-10:])
    await update.message.reply_text(msg if task_log else "📭 No tasks completed yet.")

async def cmd_earnings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total = sum(t["pay"] for t in task_log if t["status"] == "completed")
    await update.message.reply_text(f"💰 Total earned: ${total}")

async def cmd_learn_auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_learn_enabled
    auto_learn_enabled = " ".join(context.args).lower() == "on"
    await update.message.reply_text(f"🧠 Auto-learning is now {'ON' if auto_learn_enabled else 'OFF'}.")

async def cmd_retrain_auto(update: Update
