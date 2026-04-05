"""
飞书 + Claude（大脑）+ Seedance（视频生成）Bot
规则：
  1. 必须 @ 机器人才回复
  2. 必须明确说「确认」才开始生成视频
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import redis.asyncio as aioredis
from dotenv import load_dotenv
from fastapi import FastAPI, Request

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

FEISHU_APP_ID     = os.getenv("FEISHU_APP_ID")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET")
VOLCANO_API_KEY   = os.getenv("VOLCANO_API_KEY")
CLAUDE_API_KEY    = os.getenv("CLAUDE_API_KEY")
REDIS_URL         = os.getenv("REDIS_URL", "redis://localhost:6379")

SEEDANCE_MODEL    = "doubao-seedance-1-5-pro-251215"
VOLCANO_VIDEO_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
CLAUDE_API_URL    = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL      = "claude-sonnet-4-5"

# 会话状态
STATE_IDLE         = "idle"
STATE_CREATING     = "creating"
STATE_SCRIPT_READY = "script_ready"
STATE_GENERATING   = "generating"
STATE_REVIEW       = "reviewing"
STATE_MOD_CONFIRM  = "mod_confirming"

# 明确确认的关键词（严格匹配，避免误触发）
CONFIRM_KEYWORDS = ["确认", "确定", "开始生成", "开始吧", "可以生成", "生成吧"]
CANCEL_KEYWORDS  = ["取消", "不了", "算了", "不要", "停止"]
APPROVE_KEYWORDS = ["满意", "完成", "通过", "就这个", "好了"]

SYSTEM_PROMPT = """你是一个专业的视频创作助手，帮助用户创作视频脚本并生成视频。

你的工作流程：
1. 了解用户的视频需求（产品/主题、目标受众、风格、时长、横竖屏）
2. 根据需求创作专业的逐秒分镜脚本
3. 等用户明确说「确认」后，才开始生成视频

重要规则：
- 不要主动询问用户是否确认，等用户自己说「确认」
- 脚本展示完后，告诉用户：满意请回复「确认」开始生成，需要修改请直接告诉我
- 回复要简洁专业，用中文

创作脚本时要：
- 每秒都有具体的画面描述
- 包含镜头运动（推近、拉远、环绕等）
- 包含光线和氛围描述"""

http_client:  httpx.AsyncClient
redis_client: aioredis.Redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, redis_client
    http_client  = httpx.AsyncClient(timeout=60)
    redis_client = await aioredis.from_url(REDIS_URL, decode_responses=True)
    log.info("✅ 服务启动成功")
    yield
    await http_client.aclose()
    await redis_client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "time": int(time.time())}


@app.post("/feishu/event")
async def feishu_webhook(request: Request):
    body_bytes = await request.body()
    body = json.loads(body_bytes)
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge")}
    header = body.get("header", {})
    if header.get("event_type") == "im.message.receive_v1":
        asyncio.create_task(handle_message(body.get("event", {})))
    return {"code": 0}


# ══════════════════════════════════════════════════════════
#  消息处理主入口
# ══════════════════════════════════════════════════════════

async def handle_message(event: dict):
    msg = event.get("message", {})
    chat_id = msg.get("chat_id")

    if msg.get("message_type") != "text":
        return

    raw = json.loads(msg.get("content", "{}"))
    user_text = raw.get("text", "").strip()

    # ── 关键规则：必须 @ 机器人才回复 ──
    # 飞书群消息中，@ 机器人的消息 text 里会包含 @_user_1 或类似标记
    # 如果是单聊则不需要 @ 直接回复
    msg_type = msg.get("chat_type", "group")
    is_mentioned = "@_user_1" in user_text or msg_type == "p2p"

    if not is_mentioned:
        log.info(f"[chat={chat_id}] 未被@，忽略消息")
        return

    # 清除 @ 标记，获取纯文本
    user_text = user_text.replace("@_user_1", "").strip()
    if not user_text:
        return

    log.info(f"[chat={chat_id}] 收到：{user_text}")
    session = await load_session(chat_id)
    state = session.get("state", STATE_IDLE)

    if state == STATE_IDLE or state == STATE_CREATING:
        await handle_creating(chat_id, session, user_text)

    elif state == STATE_SCRIPT_READY:
        await handle_script_confirm(chat_id, session, user_text)

    elif state == STATE_REVIEW:
        await handle_review(chat_id, session, user_text)

    elif state == STATE_MOD_CONFIRM:
        await handle_mod_confirm(chat_id, session, user_text)

    elif state == STATE_GENERATING:
        await send_text(chat_id, "视频正在生成中，请稍候…")


# ══════════════════════════════════════════════════════════
#  状态1+2：对话创作脚本
# ══════════════════════════════════════════════════════════

async def handle_creating(chat_id: str, session: dict, user_text: str):
    history = session.get("chat_history", [])
    history.append({"role": "user", "content": user_text})

    # 判断信息是否足够
    judge_prompt = history + [{
        "role": "user",
        "content": "【系统判断】根据以上对话，你是否已经收集到足够信息可以创作完整的视频脚本了？只回答 YES 或 NO。"
    }]
    judge = await claude_chat(judge_prompt, system="只回答YES或NO，不要其他内容。")

    if "YES" in judge.upper():
        # 生成专业脚本
        script_prompt = history + [{
            "role": "user",
            "content": """【系统指令】请根据以上对话内容，创作专业的视频脚本。

格式要求：
逐秒分镜（每秒一行），然后一行写参数，然后写 ---PROMPT--- 分隔线，最后写Seedance提示词。

示例：
【第1秒】黑色背景，产品从中心渐现，光线汇聚
【第2秒】镜头推近展示产品细节
【第3秒】产品360°旋转，质感特写
【第4秒】镜头拉远，产品悬浮光粒子中
【第5秒】品牌感收尾，画面渐暗

风格：科技感 | 时长：5s | 比例：9:16

---PROMPT---
黑色背景产品渐现，镜头推近展示细节，产品旋转展示质感，拉远悬浮光粒子，科技感收尾。竖屏9:16。"""
        }]
        script_response = await claude_chat(script_prompt, system=SYSTEM_PROMPT)

        if "---PROMPT---" in script_response:
            parts = script_response.split("---PROMPT---")
            script_display = parts[0].strip()
            seedance_prompt = parts[1].strip()
        else:
            script_display = script_response
            seedance_prompt = script_response

        duration = 5
        ratio = "9:16"
        if "16:9" in script_response:
            ratio = "16:9"
        if "10s" in script_response or "10秒" in script_response:
            duration = 10

        history.append({"role": "assistant", "content": script_response})
        session["state"] = STATE_SCRIPT_READY
        session["script_display"] = script_display
        session["seedance_prompt"] = seedance_prompt
        session["duration"] = duration
        session["ratio"] = ratio
        session["chat_history"] = history
        session["version"] = 0
        session["video_url"] = None
        session["history"] = []
        await save_session(chat_id, session)

        await send_text(chat_id,
            f"好的！我为你创作了以下脚本：\n\n"
            f"{script_display}\n\n"
            f"────────────────\n"
            f"满意请回复「确认」开始生成视频\n"
            f"需要修改请直接告诉我哪里需要调整\n"
            f"想重来请回复「重新创作」"
        )
    else:
        # 继续收集信息
        reply = await claude_chat(history, system=SYSTEM_PROMPT)
        history.append({"role": "assistant", "content": reply})
        session["state"] = STATE_CREATING
        session["chat_history"] = history
        await save_session(chat_id, session)
        await send_text(chat_id, reply)


# ══════════════════════════════════════════════════════════
#  状态3：等待用户确认脚本
#  严格规则：只有明确说「确认」才生成，其他都当修改意见
# ══════════════════════════════════════════════════════════

async def handle_script_confirm(chat_id: str, session: dict, user_text: str):

    # 重新创作
    if any(k in user_text for k in ["重新创作", "重新来", "重写", "推倒重来"]):
        session["state"] = STATE_CREATING
        session["chat_history"] = []
        await save_session(chat_id, session)
        await send_text(chat_id, "好的，我们重新来！请告诉我你想做什么样的视频？")
        return

    # 严格判断是否是「确认」
    is_confirm = any(k in user_text for k in CONFIRM_KEYWORDS)

    if is_confirm:
        # 用户明确确认，开始生成
        session["state"] = STATE_GENERATING
        await save_session(chat_id, session)
        await send_text(chat_id, "好的！开始生成视频，大约需要 1～3 分钟，请稍候…")

        task_id = await submit_seedance(
            session["seedance_prompt"],
            duration=session.get("duration", 5),
            aspect_ratio=session.get("ratio", "9:16")
        )
        if not task_id:
            session["state"] = STATE_SCRIPT_READY
            await save_session(chat_id, session)
            await send_text(chat_id, "视频任务提交失败，请回复「确认」重试。")
            return

        asyncio.create_task(poll_and_notify(chat_id, task_id, session, is_first=True))

    else:
        # 不是确认，当作修改意见处理
        history = session.get("chat_history", [])
        history.append({
            "role": "user",
            "content": f"请根据以下意见修改脚本：{user_text}\n\n修改后按照原格式输出完整脚本，包含---PROMPT---分隔线和提示词。"
        })

        script_response = await claude_chat(history, system=SYSTEM_PROMPT)

        if "---PROMPT---" in script_response:
            parts = script_response.split("---PROMPT---")
            script_display = parts[0].strip()
            seedance_prompt = parts[1].strip()
        else:
            script_display = script_response
            seedance_prompt = script_response

        if "16:9" in script_response:
            session["ratio"] = "16:9"
        if "9:16" in script_response:
            session["ratio"] = "9:16"

        history.append({"role": "assistant", "content": script_response})
        session["script_display"] = script_display
        session["seedance_prompt"] = seedance_prompt
        session["chat_history"] = history
        await save_session(chat_id, session)

        await send_text(chat_id,
            f"好的，更新后的脚本：\n\n"
            f"{script_display}\n\n"
            f"────────────────\n"
            f"满意请回复「确认」开始生成\n"
            f"需要继续调整请直接告诉我"
        )


# ══════════════════════════════════════════════════════════
#  状态4：审核视频
# ══════════════════════════════════════════════════════════

async def handle_review(chat_id: str, session: dict, user_text: str):
    if any(k in user_text for k in APPROVE_KEYWORDS):
        await send_card(chat_id,
            title="视频已确认 ✓",
            body=f"太好了！第 {session['version']} 版视频已确认。\n\n**视频地址：**\n{session['video_url']}",
            color="green"
        )
        await delete_session(chat_id)
        return

    # 提出修改意见
    feedback = user_text
    old_prompt = session.get("seedance_prompt", "")

    update_messages = [{
        "role": "user",
        "content": f"原来的视频提示词是：{old_prompt}\n\n用户看了视频后说：{feedback}\n\n请根据反馈优化提示词，只输出新的提示词，不要其他内容。"
    }]
    new_prompt = await claude_chat(update_messages, system="你是视频提示词优化专家，根据用户反馈优化提示词，只输出提示词。")

    session["pending_prompt"] = new_prompt
    session["pending_feedback"] = feedback
    session["state"] = STATE_MOD_CONFIRM
    await save_session(chat_id, session)

    await send_text(chat_id,
        f"收到！\n\n"
        f"修改意见：「{feedback}」\n\n"
        f"已根据你的意见调整了方案。\n\n"
        f"回复「确认」生成第 {session['version'] + 1} 版\n"
        f"回复「取消」保留当前版本"
    )


# ══════════════════════════════════════════════════════════
#  状态5：确认修改后生成
# ══════════════════════════════════════════════════════════

async def handle_mod_confirm(chat_id: str, session: dict, user_text: str):
    if any(k in user_text for k in CONFIRM_KEYWORDS):
        new_prompt = session.get("pending_prompt", "")
        feedback = session.get("pending_feedback", "")

        session["history"].append({
            "version":   session["version"],
            "video_url": session["video_url"],
            "prompt":    session["seedance_prompt"],
            "feedback":  feedback,
        })
        session["seedance_prompt"] = new_prompt
        session["state"] = STATE_GENERATING
        await save_session(chat_id, session)

        await send_text(chat_id, f"好的！正在生成第 {session['version'] + 1} 版，请稍候…")

        task_id = await submit_seedance(
            new_prompt,
            duration=session.get("duration", 5),
            aspect_ratio=session.get("ratio", "9:16")
        )
        if not task_id:
            session["state"] = STATE_REVIEW
            await save_session(chat_id, session)
            await send_text(chat_id, "任务提交失败，请回复「确认」重试。")
            return

        asyncio.create_task(poll_and_notify(chat_id, task_id, session, is_first=False))

    elif any(k in user_text for k in CANCEL_KEYWORDS):
        session["state"] = STATE_REVIEW
        await save_session(chat_id, session)
        await send_text(chat_id, f"好的，保留当前第 {session['version']} 版视频。\n如需修改请继续告诉我。")

    else:
        await send_text(chat_id, "请回复「确认」开始生成，或回复「取消」保留当前版本。")


# ══════════════════════════════════════════════════════════
#  Claude API
# ══════════════════════════════════════════════════════════

async def claude_chat(messages: list, system: str = SYSTEM_PROMPT) -> str:
    headers = {
        "x-api-key":         CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type":      "application/json"
    }
    payload = {
        "model":      CLAUDE_MODEL,
        "max_tokens": 1000,
        "system":     system,
        "messages":   messages
    }
    try:
        resp = await http_client.post(CLAUDE_API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]
    except Exception as e:
        log.error(f"Claude 调用失败：{e}")
        return "抱歉，我暂时无法回复，请稍后再试。"


# ══════════════════════════════════════════════════════════
#  Seedance API
# ══════════════════════════════════════════════════════════

async def submit_seedance(prompt: str, duration: int = 5, aspect_ratio: str = "9:16") -> Optional[str]:
    payload = {
        "model":     SEEDANCE_MODEL,
        "content":   [{"type": "text", "text": prompt}],
        "ratio":     aspect_ratio,
        "duration":  duration,
        "watermark": False,
    }
    headers = {
        "Authorization": f"Bearer {VOLCANO_API_KEY}",
        "Content-Type":  "application/json"
    }
    try:
        resp = await http_client.post(VOLCANO_VIDEO_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        task_id = data.get("id")
        log.info(f"✅ 视频任务已提交 task_id={task_id}")
        return task_id
    except Exception as e:
        log.error(f"❌ 视频任务提交失败：{e}")
        return None


async def query_seedance(task_id: str) -> dict:
    headers = {"Authorization": f"Bearer {VOLCANO_API_KEY}"}
    url = f"{VOLCANO_VIDEO_URL}/{task_id}"
    try:
        resp = await http_client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.error(f"查询失败：{e}")
        return {}


async def poll_and_notify(chat_id: str, task_id: str, session: dict, is_first: bool):
    max_seconds = 600
    interval = 15
    elapsed = 0
    version = session["version"] + 1

    while elapsed < max_seconds:
        await asyncio.sleep(interval)
        elapsed += interval

        result = await query_seedance(task_id)
        status = result.get("status", "")
        log.info(f"[task={task_id}] 状态={status} 已等待={elapsed}s")

        if status == "succeeded":
            video_url = result.get("content", {}).get("video_url")
            if video_url:
                session["video_url"] = video_url
                session["version"]   = version
                session["state"]     = STATE_REVIEW
                await save_session(chat_id, session)
                await notify_done(chat_id, video_url, version, is_first)
            else:
                session["state"] = STATE_SCRIPT_READY
                await save_session(chat_id, session)
                await send_text(chat_id, "视频生成完成，但获取地址出错，请回复「确认」重试。")
            return

        elif status == "failed":
            err = result.get("error", {}).get("message", "未知错误")
            session["state"] = STATE_IDLE
            await save_session(chat_id, session)
            await send_text(chat_id, f"视频生成失败：{err}。请重新描述需求。")
            return

        elif status in ("running", "pending", "queued"):
            if elapsed == 60:
                await send_text(chat_id, "视频生成中，大约还需要 1 分钟，请继续等待…")

    session["state"] = STATE_IDLE
    await save_session(chat_id, session)
    await send_text(chat_id, "视频生成超时，请重新发送需求。")


# ══════════════════════════════════════════════════════════
#  飞书消息发送
# ══════════════════════════════════════════════════════════

_token_cache: dict = {"token": "", "expires": 0}


async def get_feishu_token() -> str:
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires"] - 60:
        return _token_cache["token"]
    resp = await http_client.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}
    )
    data = resp.json()
    _token_cache["token"]   = data.get("tenant_access_token", "")
    _token_cache["expires"] = now + data.get("expire", 7200)
    return _token_cache["token"]


async def send_text(chat_id: str, text: str):
    token = await get_feishu_token()
    await http_client.post(
        "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "receive_id": chat_id,
            "msg_type":   "text",
            "content":    json.dumps({"text": text})
        }
    )


async def send_card(chat_id: str, title: str, body: str, color: str = "blue"):
    token = await get_feishu_token()
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title":    {"tag": "plain_text", "content": title},
            "template": color
        },
        "elements": [{"tag": "markdown", "content": body}]
    }
    await http_client.post(
        "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "receive_id": chat_id,
            "msg_type":   "interactive",
            "content":    json.dumps(card)
        }
    )


async def notify_done(chat_id: str, video_url: str, version: int, is_first: bool):
    title = "首版视频已生成 🎬" if is_first else f"第 {version} 版视频已生成 ✨"
    color = "blue" if is_first else "turquoise"
    body = (
        f"**点击查看视频：**\n{video_url}\n\n"
        "---\n"
        "**满意了吗？**\n"
        "- 需要修改 → 直接告诉我修改意见，我会重新调整方案让你确认\n"
        "- 已经满意 → 回复「满意」完成流程"
    )
    await send_card(chat_id, title=title, body=body, color=color)


# ══════════════════════════════════════════════════════════
#  会话管理
# ══════════════════════════════════════════════════════════

SESSION_TTL = 86400


async def load_session(chat_id: str) -> dict:
    raw = await redis_client.get(f"session:{chat_id}")
    return json.loads(raw) if raw else {}


async def save_session(chat_id: str, session: dict):
    await redis_client.setex(
        f"session:{chat_id}",
        SESSION_TTL,
        json.dumps(session, ensure_ascii=False)
    )


async def delete_session(chat_id: str):
    await redis_client.delete(f"session:{chat_id}")
