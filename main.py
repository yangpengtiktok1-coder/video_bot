"""
飞书 + Claude（导演）+ Seedance（摄影机）Bot

核心架构：
- 整个创作过程就是和 Claude 的自然对话
- Claude 自己决定什么时候写脚本、什么时候调用 Seedance
- 只有两个状态：chatting（对话中）/ generating（生成中）
- 支持图片上传：Claude 分析图片融入脚本
- /重置 随时可用
"""

import asyncio
import base64
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

STATE_CHATTING   = "chatting"
STATE_GENERATING = "generating"

RESET_KEYWORDS = ["/重置", "/reset", "重置", "重新开始", "清除"]

# ══════════════════════════════════════════════════════════
#  Claude 的系统提示词（导演角色）
# ══════════════════════════════════════════════════════════

DIRECTOR_PROMPT = """你是一位专业的视频导演助手，通过飞书和用户对话，帮助他们创作视频。

## 你的工作方式

整个创作过程就是自然对话，你来主导：

1. **了解需求**：主动了解产品/主题、目标受众、风格、时长、横竖屏等
2. **分析素材**：用户上传图片时，仔细分析图片内容、风格、色调，融入创作
3. **创作脚本**：信息足够后，主动创作专业的逐秒分镜脚本
4. **打磨脚本**：根据用户反馈修改，直到用户满意
5. **等待确认**：脚本完成后告诉用户「满意请回复「确认拍摄」」
6. **执行拍摄**：用户确认后，你输出一段特殊格式让系统调用 Seedance

## 脚本格式（信息足够时主动创作）

【第1秒】具体画面描述，镜头运动
【第2秒】具体画面描述，镜头运动
...（每秒一行）

参数：风格 xxx | 时长 xs | 比例 16:9或9:16

## 确认拍摄后的输出格式（非常重要）

用户说「确认拍摄」后，你必须输出以下格式，系统会识别并执行：

[SEEDANCE_START]
（这里写给Seedance的完整拍摄指令，把所有分镜合并成一段连贯的中文描述，包含镜头、光线、风格、时长、比例）
[SEEDANCE_END]
[DURATION:5]
[RATIO:9:16]

## 重要原则

- 全程保持自然对话，不要太机械
- 用户没有明确说「确认拍摄」，绝对不输出 SEEDANCE_START 标记
- 用户上传图片后，要仔细描述你看到的内容，说明会如何融入视频
- 脚本修改时，完整输出修改后的脚本，不要只说「已修改」
- 对话要简洁专业，用中文
- 如果用户只是闲聊或提问，正常回答，不要强行引导做视频"""


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
    msg      = event.get("message", {})
    chat_id  = msg.get("chat_id")
    msg_type = msg.get("message_type", "text")
    chat_type = msg.get("chat_type", "group")

    raw_content = json.loads(msg.get("content", "{}"))

    # ── @ 规则：文字消息必须 @，图片和视频素材直接处理 ──
    if chat_type == "group" and msg_type == "text":
        text_check = raw_content.get("text", "")
        if "@_user_1" not in text_check:
            return  # 文字消息没有 @，忽略
    # 图片和视频消息不需要 @，直接处理

    session = await load_session(chat_id)
    state   = session.get("state", STATE_CHATTING)

    # ── 生成中：只接受重置 ──
    if state == STATE_GENERATING:
        text = raw_content.get("text", "").replace("@_user_1", "").strip()
        if any(k in text for k in RESET_KEYWORDS):
            await delete_session(chat_id)
            await send_text(chat_id, "已重置！视频生成任务已取消，我们重新开始吧。")
        else:
            await send_text(chat_id, "视频正在生成中，请稍候…\n发「/重置」可取消。")
        return

    # ── 重置命令 ──
    text = raw_content.get("text", "").replace("@_user_1", "").strip()
    if any(k in text for k in RESET_KEYWORDS):
        await delete_session(chat_id)
        await send_text(chat_id, "好的，已重置！重新开始吧，你想做什么样的视频？")
        return

    # ── 处理文字消息 ──
    if msg_type == "text":
        if not text:
            return
        log.info(f"[chat={chat_id}] 文字：{text[:50]}")
        await handle_text(chat_id, session, text)

    # ── 处理图片消息 ──
    elif msg_type == "image":
        log.info(f"[chat={chat_id}] 收到图片")
        await handle_image(chat_id, session, msg, raw_content)

    # ── 处理视频消息 ──
    elif msg_type == "media":
        log.info(f"[chat={chat_id}] 收到视频")
        await handle_video(chat_id, session, msg, raw_content)

    # ── 其他类型 ──
    else:
        await send_text(chat_id, "目前支持文字、图片和视频消息～")


# ══════════════════════════════════════════════════════════
#  处理文字消息
# ══════════════════════════════════════════════════════════

async def handle_text(chat_id: str, session: dict, text: str):
    history = session.get("history", [])
    history.append({"role": "user", "content": text})

    # 调用 Claude
    reply = await claude_chat(history, system=DIRECTOR_PROMPT)
    history.append({"role": "assistant", "content": reply})

    # 检查 Claude 是否输出了 Seedance 指令
    if "[SEEDANCE_START]" in reply and "[SEEDANCE_END]" in reply:
        await handle_seedance_trigger(chat_id, session, history, reply)
    else:
        # 普通对话回复
        session["history"] = history[-20:]  # 保留最近20条
        session["state"]   = STATE_CHATTING
        await save_session(chat_id, session)

        # 去掉可能残留的标记，发送干净的回复
        clean_reply = reply.replace("[SEEDANCE_START]", "").replace("[SEEDANCE_END]", "").strip()
        await send_text(chat_id, clean_reply)


# ══════════════════════════════════════════════════════════
#  处理图片消息
# ══════════════════════════════════════════════════════════

async def handle_image(chat_id: str, session: dict, msg: dict, raw_content: dict):
    await send_text(chat_id, "收到图片，正在分析…")

    # 从飞书下载图片
    image_key = raw_content.get("image_key", "")
    image_data = await download_feishu_image(msg.get("message_id", ""), image_key)

    history = session.get("history", [])

    if image_data:
        # 把图片以 base64 形式发给 Claude 分析
        message_content = [
            {
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": "image/jpeg",
                    "data":       image_data
                }
            },
            {
                "type": "text",
                "text": "用户上传了这张图片作为视频参考素材。请仔细分析图片内容、风格、色调、主体，告诉用户你看到了什么，以及你打算如何将它融入视频创作中。"
            }
        ]
        history.append({"role": "user", "content": message_content})
    else:
        # 图片下载失败，告知用户
        history.append({
            "role": "user",
            "content": "用户上传了一张参考图片（下载失败，请用文字描述图片内容）。"
        })

    reply = await claude_chat(history, system=DIRECTOR_PROMPT)
    history.append({"role": "assistant", "content": reply})

    session["history"] = history[-20:]
    session["state"]   = STATE_CHATTING
    await save_session(chat_id, session)

    await send_text(chat_id, reply)


async def download_feishu_image(message_id: str, image_key: str) -> Optional[str]:
    """从飞书下载图片，返回 base64 字符串"""
    try:
        token = await get_feishu_token()
        url   = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/resources/{image_key}?type=image"
        resp  = await http_client.get(url, headers={"Authorization": f"Bearer {token}"})
        if resp.status_code == 200:
            return base64.b64encode(resp.content).decode("utf-8")
        log.error(f"图片下载失败：{resp.status_code}")
        return None
    except Exception as e:
        log.error(f"图片下载异常：{e}")
        return None


# ══════════════════════════════════════════════════════════
#  检测到 Seedance 指令，执行拍摄
# ══════════════════════════════════════════════════════════

async def handle_video(chat_id: str, session: dict, msg: dict, raw_content: dict):
    """处理视频消息：抽帧给 Claude 分析，同时保存视频 URL 供 Seedance 参考"""
    await send_text(chat_id, "收到视频，正在处理…")

    message_id = msg.get("message_id", "")
    file_key   = raw_content.get("file_key", "")

    # 从飞书下载视频文件
    video_bytes = await download_feishu_file(message_id, file_key)
    history = session.get("history", [])

    if video_bytes:
        # 用 ffmpeg 抽取 3 帧关键画面
        frames = await extract_video_frames(video_bytes)

        if frames:
            # 把帧图片发给 Claude 分析
            content = []
            for i, frame_b64 in enumerate(frames):
                content.append({
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": "image/jpeg",
                        "data":       frame_b64
                    }
                })
            content.append({
                "type": "text",
                "text": f"用户上传了一段参考视频，我已从中提取了 {len(frames)} 帧关键画面（开头、中间、结尾）。请仔细分析视频的画面风格、色调、镜头语言、主体内容，告诉用户你看到了什么，以及你打算如何将这个视频的风格或内容融入新的视频创作中。"
            })
            history.append({"role": "user", "content": content})

            # 同时保存视频信息供 Seedance 参考
            session["reference_video"] = f"用户上传的参考视频（已分析{len(frames)}帧）"
        else:
            # ffmpeg 抽帧失败，降级处理
            history.append({
                "role": "user",
                "content": "用户上传了一段参考视频（视频帧提取失败）。请告知用户视频已收到但无法预览，请用文字描述视频的风格和内容，我会按照描述来创作。"
            })
    else:
        history.append({
            "role": "user",
            "content": "用户上传了一段参考视频（下载失败）。请告知用户视频下载失败，可以尝试重新发送，或用文字描述视频内容。"
        })

    reply = await claude_chat(history, system=DIRECTOR_PROMPT)
    history.append({"role": "assistant", "content": reply})
    session["history"] = history[-20:]
    session["state"]   = STATE_CHATTING
    await save_session(chat_id, session)
    await send_text(chat_id, reply)


async def download_feishu_file(message_id: str, file_key: str) -> Optional[bytes]:
    """从飞书下载文件，返回原始字节"""
    try:
        token = await get_feishu_token()
        url   = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/resources/{file_key}?type=file"
        resp  = await http_client.get(url, headers={"Authorization": f"Bearer {token}"})
        if resp.status_code == 200:
            return resp.content
        log.error(f"视频下载失败：{resp.status_code}")
        return None
    except Exception as e:
        log.error(f"视频下载异常：{e}")
        return None


async def extract_video_frames(video_bytes: bytes) -> list:
    """用 ffmpeg 从视频中抽取 3 帧，返回 base64 列表"""
    import tempfile, subprocess, os
    frames = []
    try:
        # 写入临时文件
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vf:
            vf.write(video_bytes)
            video_path = vf.name

        # 抽取 3 帧：开头、中间、结尾
        for i, timestamp in enumerate(["00:00:00", "50%", "99%"]):
            out_path = f"{video_path}_frame{i}.jpg"
            cmd = [
                "ffmpeg", "-y",
                "-ss", timestamp if "%" not in timestamp else "0",
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                out_path
            ]
            if "%" in timestamp:
                # 用 -sseof 取结尾帧
                cmd = ["ffmpeg", "-y", "-sseof", "-1", "-i", video_path, "-vframes", "1", "-q:v", "2", out_path]

            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode == 0 and os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    frames.append(base64.b64encode(f.read()).decode("utf-8"))
                os.remove(out_path)

        os.remove(video_path)
        log.info(f"成功提取 {len(frames)} 帧")
    except Exception as e:
        log.error(f"ffmpeg 抽帧失败：{e}")

    return frames


async def handle_seedance_trigger(chat_id: str, session: dict, history: list, reply: str):
    """Claude 输出了拍摄指令，解析并提交 Seedance 任务"""
    try:
        # 解析 prompt
        start = reply.index("[SEEDANCE_START]") + len("[SEEDANCE_START]")
        end   = reply.index("[SEEDANCE_END]")
        prompt = reply[start:end].strip()

        # 解析时长
        duration = 5
        if "[DURATION:" in reply:
            d_start = reply.index("[DURATION:") + len("[DURATION:")
            d_end   = reply.index("]", d_start)
            duration = int(reply[d_start:d_end])

        # 解析比例
        ratio = "9:16"
        if "[RATIO:" in reply:
            r_start = reply.index("[RATIO:") + len("[RATIO:")
            r_end   = reply.index("]", r_start)
            ratio = reply[r_start:r_end].strip()

        log.info(f"Seedance 指令解析成功 duration={duration} ratio={ratio} prompt={prompt[:50]}")

    except Exception as e:
        log.error(f"Seedance 指令解析失败：{e}")
        await send_text(chat_id, "脚本格式有点问题，Claude 正在重新整理，请稍候…")
        return

    # 保存当前版本信息
    version = session.get("version", 0) + 1
    session["history"]        = history[-20:]
    session["state"]          = STATE_GENERATING
    session["current_prompt"] = prompt
    session["duration"]       = duration
    session["ratio"]          = ratio
    session["version"]        = version
    if "video_url" not in session:
        session["video_url"] = None
    await save_session(chat_id, session)

    is_first = version == 1
    await send_text(chat_id,
        f"好！开始拍摄{'首版' if is_first else f'第{version}版'}视频，大约需要 1～3 分钟，请稍候…"
    )

    # 提交任务
    task_id = await submit_seedance(prompt, duration=duration, aspect_ratio=ratio)
    if not task_id:
        session["state"] = STATE_CHATTING
        await save_session(chat_id, session)
        await send_text(chat_id, "视频任务提交失败，请告诉我重新生成。")
        return

    asyncio.create_task(poll_and_notify(chat_id, task_id, session, is_first=is_first))


# ══════════════════════════════════════════════════════════
#  Claude API
# ══════════════════════════════════════════════════════════

async def claude_chat(messages: list, system: str = DIRECTOR_PROMPT) -> str:
    headers = {
        "x-api-key":         CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type":      "application/json"
    }
    payload = {
        "model":      CLAUDE_MODEL,
        "max_tokens": 1500,
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
        data    = resp.json()
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
    interval    = 15
    elapsed     = 0
    version     = session.get("version", 1)

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
                session["state"]     = STATE_CHATTING
                await save_session(chat_id, session)

                # 把视频结果告知 Claude，让它来回复用户
                history = session.get("history", [])
                history.append({
                    "role": "user",
                    "content": f"【系统通知】{'首版' if is_first else f'第{version}版'}视频已生成成功！视频地址：{video_url}\n\n请告知用户视频已完成，提醒他们查看，并询问是否满意或需要修改。"
                })
                reply = await claude_chat(history, system=DIRECTOR_PROMPT)
                history.append({"role": "assistant", "content": reply})
                session["history"] = history[-20:]
                await save_session(chat_id, session)

                await send_card(chat_id,
                    title=f"{'首版' if is_first else f'第{version}版'}视频已生成 🎬",
                    body=f"**点击查看视频：**\n{video_url}\n\n{reply}",
                    color="blue" if is_first else "turquoise"
                )
            else:
                session["state"] = STATE_CHATTING
                await save_session(chat_id, session)
                await send_text(chat_id, "视频生成完成，但获取地址出错，请告诉我重新生成。")
            return

        elif status == "failed":
            err = result.get("error", {}).get("message", "未知错误")
            session["state"] = STATE_CHATTING
            await save_session(chat_id, session)
            await send_text(chat_id, f"视频生成失败：{err}\n\n请告诉我需要调整什么，我们重新来。")
            return

        elif status in ("running", "pending", "queued"):
            if elapsed == 60:
                await send_text(chat_id, "视频生成中，大约还需要 1 分钟…")

    session["state"] = STATE_CHATTING
    await save_session(chat_id, session)
    await send_text(chat_id, "视频生成超时，请告诉我重新生成。")


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
