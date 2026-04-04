"""
飞书 + 火山方舟 + Seedance 视频生成 Bot
=========================================
在飞书群里说话 → 自动生成视频 → 发回飞书 → 支持多轮修改

启动命令：
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import redis.asyncio as aioredis
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  从 .env 读取配置
# ─────────────────────────────────────────────
FEISHU_APP_ID       = os.getenv("FEISHU_APP_ID")
FEISHU_APP_SECRET   = os.getenv("FEISHU_APP_SECRET")
FEISHU_VERIFY_TOKEN = os.getenv("FEISHU_VERIFY_TOKEN", "")
VOLCANO_API_KEY     = os.getenv("VOLCANO_API_KEY")
REDIS_URL           = os.getenv("REDIS_URL", "redis://localhost:6379")

# ─────────────────────────────────────────────
#  火山方舟 Seedance 接口
# ─────────────────────────────────────────────
SEEDANCE_MODEL   = "doubao-seedance-1-5-pro-250528"
VOLCANO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/content/generation/tasks"

# ─────────────────────────────────────────────
#  意图关键词
# ─────────────────────────────────────────────
GENERATE_KEYWORDS = ["生成", "制作", "创建", "做一个", "帮我做", "视频", "拍", "generate", "create", "make"]
APPROVE_KEYWORDS  = ["满意", "通过", "好的", "不错", "可以", "确认", "完成", "ok", "OK", "好", "棒", "赞"]

# ─────────────────────────────────────────────
#  全局客户端
# ─────────────────────────────────────────────
http_client:  httpx.AsyncClient
redis_client: aioredis.Redis

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, redis_client
    http_client  = httpx.AsyncClient(timeout=30)
    redis_client = await aioredis.from_url(REDIS_URL, decode_responses=True)
    log.info("✅ 服务启动成功")
    yield
    await http_client.aclose()
    await redis_client.aclose()

app = FastAPI(lifespan=lifespan)


# ══════════════════════════════════════════════════════════
#  一、接收飞书消息
# ══════════════════════════════════════════════════════════

@app.post("/feishu/event")
async def feishu_webhook(request: Request):
    body_bytes = await request.body()
    body       = json.loads(body_bytes)

    # 飞书 URL 验证（第一次配置时）
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge")}

   # 签名校验（已关闭）
    pass

    header = body.get("header", {})
    if header.get("event_type") == "im.message.receive_v1":
        asyncio.create_task(handle_message(body.get("event", {})))

    return {"code": 0}


def _check_signature(ts, nc, body_bytes, signature):
    token   = FEISHU_VERIFY_TOKEN
    content = (ts + nc + token).encode() + body_bytes
    digest  = hmac.new(token.encode(), content, hashlib.sha256).hexdigest()
    return digest == signature


# ══════════════════════════════════════════════════════════
#  二、判断意图，分发任务
# ══════════════════════════════════════════════════════════

async def handle_message(event: dict):
    msg     = event.get("message", {})
    chat_id = msg.get("chat_id")

    if msg.get("message_type") != "text":
        await send_text(chat_id, "目前只支持文字消息，请用文字描述视频需求或修改意见～")
        return

    raw_content = json.loads(msg.get("content", "{}"))
    user_text   = raw_content.get("text", "").strip()
    user_text   = user_text.replace("@_user_1", "").strip()
    if not user_text:
        return

    log.info(f"[chat={chat_id}] 收到：{user_text}")

    session = await load_session(chat_id)

    # 用户说满意 → 结束
    if session.get("video_url") and any(k in user_text for k in APPROVE_KEYWORDS):
        await send_card(
            chat_id,
            title="视频已确认 ✓",
            body=f"太好了！第 {session['version']} 版视频已确认。\n\n**视频地址：**\n{session['video_url']}",
            color="green"
        )
        await delete_session(chat_id)
        return

    # 已有视频 + 不像新需求 → 修改
    if session.get("video_url") and not any(k in user_text for k in GENERATE_KEYWORDS):
        await do_modify(chat_id, session, user_text)
        return

    # 其他情况 → 新建视频
    await do_generate(chat_id, user_text)


# ══════════════════════════════════════════════════════════
#  三、生成新视频
# ══════════════════════════════════════════════════════════

async def do_generate(chat_id: str, user_text: str):
    await send_text(chat_id, "收到！正在生成视频，大约需要 1～3 分钟，请稍候…")

    session = {
        "chat_id":         chat_id,
        "original_prompt": user_text,
        "current_prompt":  user_text,
        "video_url":       None,
        "version":         0,
        "history":         []
    }
    await save_session(chat_id, session)

    task_id = await submit_seedance(user_text)
    if not task_id:
        await send_text(chat_id, "视频任务提交失败，请稍后重试。")
        return

    asyncio.create_task(poll_and_notify(chat_id, task_id, session, is_first=True))


# ══════════════════════════════════════════════════════════
#  四、修改视频
# ══════════════════════════════════════════════════════════

async def do_modify(chat_id: str, session: dict, feedback: str):
    next_version = session["version"] + 1
    await send_text(chat_id, f"收到修改意见，正在生成第 {next_version} 版，请稍候…")

    new_prompt = (
        f"{session['original_prompt']}。"
        f"在此基础上做以下调整：{feedback}。"
        f"保持整体风格不变，重点修改上述内容。"
    )

    session["history"].append({
        "version":   session["version"],
        "video_url": session["video_url"],
        "prompt":    session["current_prompt"],
        "feedback":  feedback,
    })
    session["current_prompt"] = new_prompt
    await save_session(chat_id, session)

    task_id = await submit_seedance(new_prompt)
    if not task_id:
        await send_text(chat_id, "任务提交失败，请重试。")
        return

    asyncio.create_task(poll_and_notify(chat_id, task_id, session, is_first=False))


# ══════════════════════════════════════════════════════════
#  五、调用火山方舟 Seedance API
# ══════════════════════════════════════════════════════════

async def submit_seedance(
    prompt: str,
    duration: int = 5,
    aspect_ratio: str = "16:9",
    resolution: str = "1080p"
) -> Optional[str]:
    """提交生成任务，返回 task_id"""
    payload = {
        "model": SEEDANCE_MODEL,
        "content": [{"type": "text", "text": prompt}],
        "parameters": {
            "duration":   duration,
            "ratio":      aspect_ratio,
            "resolution": resolution,
            "watermark":  False,
        }
    }
    headers = {
        "Authorization": f"Bearer {VOLCANO_API_KEY}",
        "Content-Type":  "application/json"
    }
    try:
        resp = await http_client.post(VOLCANO_BASE_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data    = resp.json()
        task_id = data.get("id")
        log.info(f"✅ 任务已提交 task_id={task_id}")
        return task_id
    except Exception as e:
        log.error(f"❌ 任务提交失败：{e}")
        return None


async def query_seedance(task_id: str) -> dict:
    """查询任务状态"""
    headers = {"Authorization": f"Bearer {VOLCANO_API_KEY}"}
    url     = f"{VOLCANO_BASE_URL}/{task_id}"
    try:
        resp = await http_client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.error(f"查询失败：{e}")
        return {}


# ══════════════════════════════════════════════════════════
#  六、轮询等待 + 完成后通知飞书
# ══════════════════════════════════════════════════════════

async def poll_and_notify(chat_id: str, task_id: str, session: dict, is_first: bool):
    max_seconds = 600   # 最多等 10 分钟
    interval    = 15    # 每 15 秒查一次
    elapsed     = 0
    version     = session["version"] + 1

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
                await save_session(chat_id, session)
                await notify_done(chat_id, video_url, version, is_first)
            else:
                await send_text(chat_id, "视频生成完成，但获取地址时出错，请重试。")
            return

        elif status == "failed":
            err = result.get("error", {}).get("message", "未知错误")
            await send_text(chat_id, f"视频生成失败：{err}。请修改描述后重试。")
            return

        elif status in ("running", "pending", "queued"):
            if elapsed == 60:
                await send_text(chat_id, "视频生成中，大约还需要 1 分钟，请继续等待…")

    await send_text(chat_id, "视频生成超时（超过 10 分钟），请重新发送需求。")


# ══════════════════════════════════════════════════════════
#  七、发送飞书消息
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
    card  = {
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
    body  = (
        f"**点击查看视频：**\n{video_url}\n\n"
        "---\n"
        "**满意了吗？**\n"
        "- 需要修改 → 直接回复修改意见，例如：*「背景换成白色」*\n"
        "- 已经满意 → 回复 **「满意」** 完成流程"
    )
    await send_card(chat_id, title=title, body=body, color=color)


# ══════════════════════════════════════════════════════════
#  八、会话状态（存 Redis，24 小时有效）
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


# ══════════════════════════════════════════════════════════
#  九、健康检查
# ══════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "time": int(time.time())}
