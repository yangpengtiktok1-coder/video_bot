"""
飞书 + 豆包（意图理解）+ Seedance（视频生成）Bot
对话流程：
  1. 豆包理解意图
  2. 引导用户填写脚本
  3. 生成前展示脚本让用户确认
  4. 用户说「确认」才调用 Seedance
  5. 修改时也先确认再生成
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
REDIS_URL         = os.getenv("REDIS_URL", "redis://localhost:6379")

SEEDANCE_MODEL  = "doubao-seedance-1-5-pro-251215"
DOUBAO_MODEL    = "doubao-1-5-pro-256k-250115"
VOLCANO_VIDEO_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
VOLCANO_CHAT_URL  = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

# 会话状态
STATE_IDLE      = "idle"        # 待机
STATE_COLLECT   = "collecting"  # 收集脚本中
STATE_CONFIRM   = "confirming"  # 等待用户确认脚本
STATE_GENERATING = "generating" # 生成中
STATE_REVIEW    = "reviewing"   # 审核视频中
STATE_MOD_CONFIRM = "mod_confirming" # 等待用户确认修改脚本

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
        await send_text(chat_id, "目前只支持文字消息哦～")
        return

    raw = json.loads(msg.get("content", "{}"))
    user_text = raw.get("text", "").strip()
    user_text = user_text.replace("@_user_1", "").strip()
    if not user_text:
        return

    log.info(f"[chat={chat_id}] 收到：{user_text}")
    session = await load_session(chat_id)
    state = session.get("state", STATE_IDLE)

    # 根据当前状态分发处理
    if state == STATE_IDLE:
        await handle_idle(chat_id, session, user_text)

    elif state == STATE_COLLECT:
        await handle_collect(chat_id, session, user_text)

    elif state == STATE_CONFIRM:
        await handle_confirm(chat_id, session, user_text)

    elif state == STATE_REVIEW:
        await handle_review(chat_id, session, user_text)

    elif state == STATE_MOD_CONFIRM:
        await handle_mod_confirm(chat_id, session, user_text)


# ══════════════════════════════════════════════════════════
#  状态1：待机 - 豆包判断意图
# ══════════════════════════════════════════════════════════

async def handle_idle(chat_id: str, session: dict, user_text: str):
    intent = await detect_intent(user_text)

    if intent == "make_video":
        # 想做视频，进入收集脚本状态
        session["state"] = STATE_COLLECT
        session["script_draft"] = {}
        await save_session(chat_id, session)
        await send_text(chat_id,
            "好的！我来帮你生成视频。\n\n"
            "请告诉我以下信息（可以一次说完，也可以逐条回答）：\n\n"
            "1. 画面内容是什么？\n"
            "2. 风格偏好？（科技感 / 温馨 / 活力 / 写实 等）\n"
            "3. 时长？（5秒 或 10秒）\n"
            "4. 横屏（16:9）还是竖屏（9:16）？"
        )
    else:
        # 普通对话，豆包直接回复
        reply = await doubao_chat(user_text,
            system="你是一个视频生成助手，可以帮用户生成视频。"
                   "如果用户想做视频，引导他们描述视频需求。"
                   "其他问题正常回答，保持友好简洁。"
        )
        await send_text(chat_id, reply)


# ══════════════════════════════════════════════════════════
#  状态2：收集脚本 - 豆包提取信息
# ══════════════════════════════════════════════════════════

async def handle_collect(chat_id: str, session: dict, user_text: str):
    # 用豆包从用户描述中提取脚本要素
    extract_prompt = f"""用户想做一个视频，他说：「{user_text}」

请从中提取以下信息，用JSON格式返回，只返回JSON不要其他内容：
{{
  "scene": "画面内容描述（如果用户没说则为null）",
  "style": "风格（如果用户没说则为null）",
  "duration": 5或10（秒，如果用户没说则为null）,
  "ratio": "16:9或9:16（如果用户没说则为null）",
  "complete": true或false（四项信息是否都齐全）
}}"""

    result_str = await doubao_chat(extract_prompt, system="你是信息提取助手，只输出JSON。")

    try:
        result_str = result_str.strip()
        if result_str.startswith("```"):
            result_str = result_str.split("```")[1]
            if result_str.startswith("json"):
                result_str = result_str[4:]
        info = json.loads(result_str.strip())
    except Exception:
        info = {"complete": False}

    # 合并到已有草稿
    draft = session.get("script_draft", {})
    if info.get("scene"):
        draft["scene"] = info["scene"]
    if info.get("style"):
        draft["style"] = info["style"]
    if info.get("duration"):
        draft["duration"] = info["duration"]
    if info.get("ratio"):
        draft["ratio"] = info["ratio"]
    session["script_draft"] = draft

    # 检查缺少哪些信息
    missing = []
    if not draft.get("scene"):
        missing.append("画面内容是什么？")
    if not draft.get("style"):
        missing.append("风格偏好？（科技感 / 温馨 / 活力 / 写实）")
    if not draft.get("duration"):
        missing.append("时长？（5秒 或 10秒）")
    if not draft.get("ratio"):
        missing.append("横屏（16:9）还是竖屏（9:16）？")

    if missing:
        # 还有信息没收集到
        await save_session(chat_id, session)
        reply = "好的，还需要以下信息：\n\n"
        for i, q in enumerate(missing, 1):
            reply += f"{i}. {q}\n"
        await send_text(chat_id, reply)
    else:
        # 信息收集完整，展示脚本让用户确认
        session["state"] = STATE_CONFIRM
        await save_session(chat_id, session)

        ratio_label = "横屏 16:9" if draft.get("ratio") == "16:9" else "竖屏 9:16"
        await send_text(chat_id,
            f"好的！我整理了你的视频脚本：\n\n"
            f"────────────────\n"
            f"📽 画面：{draft.get('scene')}\n"
            f"🎨 风格：{draft.get('style')}\n"
            f"⏱ 时长：{draft.get('duration')}秒\n"
            f"📐 比例：{ratio_label}\n"
            f"────────────────\n\n"
            f"确认开始生成吗？\n"
            f"回复「确认」开始生成\n"
            f"回复「修改」重新描述"
        )


# ══════════════════════════════════════════════════════════
#  状态3：等待确认脚本
# ══════════════════════════════════════════════════════════

async def handle_confirm(chat_id: str, session: dict, user_text: str):
    if any(k in user_text for k in ["确认", "好的", "可以", "开始", "ok", "OK", "是", "没问题"]):
        # 用户确认，开始生成
        draft = session.get("script_draft", {})
        prompt = build_prompt(draft)
        session["state"] = STATE_GENERATING
        session["original_prompt"] = prompt
        session["current_prompt"] = prompt
        session["version"] = 0
        session["video_url"] = None
        session["history"] = []
        await save_session(chat_id, session)

        await send_text(chat_id, "好的！开始生成视频，大约需要 1～3 分钟，请稍候…")

        task_id = await submit_seedance(
            prompt,
            duration=draft.get("duration", 5),
            aspect_ratio=draft.get("ratio", "16:9")
        )
        if not task_id:
            session["state"] = STATE_IDLE
            await save_session(chat_id, session)
            await send_text(chat_id, "视频任务提交失败，请重试。")
            return

        asyncio.create_task(poll_and_notify(chat_id, task_id, session, is_first=True))

    elif any(k in user_text for k in ["修改", "重新", "不对", "不是", "改"]):
        # 用户要修改脚本
        session["state"] = STATE_COLLECT
        session["script_draft"] = {}
        await save_session(chat_id, session)
        await send_text(chat_id,
            "好的，重新来一次！\n\n"
            "请告诉我：\n"
            "1. 画面内容是什么？\n"
            "2. 风格偏好？\n"
            "3. 时长？（5秒 或 10秒）\n"
            "4. 横屏还是竖屏？"
        )
    else:
        await send_text(chat_id, "请回复「确认」开始生成，或回复「修改」重新填写脚本。")


# ══════════════════════════════════════════════════════════
#  状态4：审核视频 - 处理修改意见
# ══════════════════════════════════════════════════════════

async def handle_review(chat_id: str, session: dict, user_text: str):
    if any(k in user_text for k in ["满意", "好的", "不错", "可以", "确认", "完成", "棒", "赞", "通过"]):
        # 用户满意，结束
        await send_card(chat_id,
            title="视频已确认 ✓",
            body=f"太好了！第 {session['version']} 版视频已确认。\n\n**视频地址：**\n{session['video_url']}",
            color="green"
        )
        await delete_session(chat_id)
    else:
        # 用户提出修改意见，整理新脚本让用户确认
        feedback = user_text
        old_prompt = session.get("current_prompt", "")

        new_prompt = (
            f"{session.get('original_prompt', '')}。"
            f"在此基础上做以下调整：{feedback}。"
            f"保持整体风格不变，重点修改上述内容。"
        )

        session["pending_prompt"] = new_prompt
        session["pending_feedback"] = feedback
        session["state"] = STATE_MOD_CONFIRM
        await save_session(chat_id, session)

        await send_text(chat_id,
            f"收到修改意见！\n\n"
            f"────────────────\n"
            f"修改要求：{feedback}\n"
            f"────────────────\n\n"
            f"确认生成第 {session['version'] + 1} 版吗？\n"
            f"回复「确认」开始生成\n"
            f"回复「取消」保留当前版本"
        )


# ══════════════════════════════════════════════════════════
#  状态5：等待确认修改脚本
# ══════════════════════════════════════════════════════════

async def handle_mod_confirm(chat_id: str, session: dict, user_text: str):
    if any(k in user_text for k in ["确认", "好的", "可以", "开始", "ok", "OK", "是"]):
        new_prompt = session.get("pending_prompt", "")
        feedback = session.get("pending_feedback", "")

        session["history"].append({
            "version":   session["version"],
            "video_url": session["video_url"],
            "prompt":    session["current_prompt"],
            "feedback":  feedback,
        })
        session["current_prompt"] = new_prompt
        session["state"] = STATE_GENERATING
        await save_session(chat_id, session)

        await send_text(chat_id, f"好的！正在生成第 {session['version'] + 1} 版，请稍候…")

        task_id = await submit_seedance(new_prompt)
        if not task_id:
            session["state"] = STATE_REVIEW
            await save_session(chat_id, session)
            await send_text(chat_id, "任务提交失败，请重试。")
            return

        asyncio.create_task(poll_and_notify(chat_id, task_id, session, is_first=False))

    elif any(k in user_text for k in ["取消", "不了", "算了"]):
        session["state"] = STATE_REVIEW
        await save_session(chat_id, session)
        await send_text(chat_id, f"好的，保留当前第 {session['version']} 版视频。\n如需修改请继续告诉我。")
    else:
        await send_text(chat_id, "请回复「确认」开始生成，或回复「取消」保留当前版本。")


# ══════════════════════════════════════════════════════════
#  豆包 API
# ══════════════════════════════════════════════════════════

async def detect_intent(user_text: str) -> str:
    """判断用户意图：make_video 或 chat"""
    prompt = f"""判断用户消息的意图，只返回以下其中一个词：
- make_video：用户想要制作、生成、创建视频
- chat：其他所有情况（聊天、提问、闲聊等）

用户消息：「{user_text}」

只返回 make_video 或 chat，不要其他内容。"""

    result = await doubao_chat(prompt, system="你是意图识别助手，只输出指定的词。")
    result = result.strip().lower()
    if "make_video" in result:
        return "make_video"
    return "chat"


async def doubao_chat(user_msg: str, system: str = "你是一个helpful助手。") -> str:
    """调用豆包对话接口"""
    headers = {
        "Authorization": f"Bearer {VOLCANO_API_KEY}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model": DOUBAO_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg}
        ],
        "max_tokens": 500
    }
    try:
        resp = await http_client.post(VOLCANO_CHAT_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        log.error(f"豆包调用失败：{e}")
        return "抱歉，我暂时无法回复，请稍后再试。"


# ══════════════════════════════════════════════════════════
#  Seedance API
# ══════════════════════════════════════════════════════════

def build_prompt(draft: dict) -> str:
    """把脚本草稿拼成 Seedance prompt"""
    parts = []
    if draft.get("scene"):
        parts.append(draft["scene"])
    if draft.get("style"):
        parts.append(f"{draft['style']}风格")
    return "，".join(parts)


async def submit_seedance(prompt: str, duration: int = 5, aspect_ratio: str = "16:9") -> Optional[str]:
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
                session["state"] = STATE_IDLE
                await save_session(chat_id, session)
                await send_text(chat_id, "视频生成完成，但获取地址时出错，请重试。")
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
    await send_text(chat_id, "视频生成超时（超过 10 分钟），请重新发送需求。")


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
        "- 需要修改 → 直接告诉我修改意见\n"
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
