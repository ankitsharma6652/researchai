"""
FastAPI backend for AI Research multi-agent system.
Streams progress updates via Server-Sent Events (SSE).

Run:
  cd "AI Research"
  uvicorn server:app --reload --port 8000
"""

import os
import sys
import json
import asyncio
import traceback
import re
import requests as _requests
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load env from parent directories
for env_path in [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
]:
    if env_path.exists():
        load_dotenv(env_path)

sys.path.insert(0, str(Path(__file__).parent))
from agent import build_reporter_agent, get_llm

app = FastAPI(title="AI Research API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Serve HTML frontend ──────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)


# ── Request/response schemas ─────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    topic: str
    provider: str = "groq"          # "groq" | "gemini" | "openai" | "openai-compatible" | "euron"
    model: Optional[str] = None     # override default model
    api_base: Optional[str] = None  # custom base URL for OpenAI-compatible endpoints
    # API keys supplied from browser (optional, fall back to .env)
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    euron_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None


# ── Key check endpoint ───────────────────────────────────────────────────────
@app.get("/api/keys")
async def check_keys():
    """Returns which keys are available in server .env."""
    return {
        "groq":   bool(os.getenv("GROQ_API_KEY")),
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
        "tavily": bool(os.getenv("TAVILY_API_KEY")),
        "devto":  bool(os.getenv("DEVTO_API_KEY")),
    }


# ── Research endpoint (SSE) ──────────────────────────────────────────────────
@app.post("/api/research")
async def research(req: ResearchRequest):
    """Stream research progress via Server-Sent Events."""

    # Resolve keys: request payload > environment variable
    groq_key   = req.groq_api_key   or os.getenv("GROQ_API_KEY",   "")
    gemini_key = req.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    openai_key = req.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    euron_key  = req.euron_api_key  or os.getenv("EURON_API_KEY",  "")
    tavily_key = req.tavily_api_key or os.getenv("TAVILY_API_KEY", "")

    async def event_generator():
        def sse(event_type: str, data: dict) -> str:
            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

        try:
            # ── Validate keys ────────────────────────────────────────────────
            if req.provider == "groq" and not groq_key:
                yield sse("error", {"message": "Groq API key not found. Please enter it in Settings or add GROQ_API_KEY to your .env file."})
                return
            if req.provider == "gemini" and not gemini_key:
                yield sse("error", {"message": "Gemini API key not found. Please enter it in Settings or add GEMINI_API_KEY to your .env file."})
                return
            if req.provider in ("openai", "openai-compatible") and not openai_key:
                yield sse("error", {"message": "OpenAI API key not found. Please enter it in Settings or add OPENAI_API_KEY to your .env file."})
                return
            if req.provider == "euron" and not euron_key:
                yield sse("error", {"message": "Euron API key not found. Please enter it in Settings (get one at euron.one)."})
                return
            if not tavily_key:
                yield sse("error", {"message": "Tavily API key not found. Please enter it in Settings or add TAVILY_API_KEY to your .env file."})
                return

            # Export keys into environment so sub-libraries can pick them up
            if groq_key:   os.environ["GROQ_API_KEY"]   = groq_key
            if gemini_key: os.environ["GEMINI_API_KEY"] = gemini_key
            if openai_key: os.environ["OPENAI_API_KEY"] = openai_key
            if tavily_key: os.environ["TAVILY_API_KEY"] = tavily_key
            # Also set GOOGLE_API_KEY (some langchain-google-genai versions need it)
            if gemini_key: os.environ["GOOGLE_API_KEY"] = gemini_key
            if req.api_base: os.environ["OPENAI_API_BASE"] = req.api_base

            yield sse("status", {"phase": "init", "message": f"Initializing {req.provider.upper()} LLM…"})
            await asyncio.sleep(0.05)

            llm   = get_llm(req.provider, req.model, api_base=req.api_base, euron_key=euron_key)
            agent = build_reporter_agent(llm)

            yield sse("status", {"phase": "planning", "message": "🧠 Planning report — searching the web for context…"})
            await asyncio.sleep(0.05)

            config          = {"recursion_limit": 60}
            sections_logged = set()
            final_report    = None
            plan_done       = False

            # ── Stream agent events ──────────────────────────────────────────
            async for event in agent.astream({'topic': req.topic}, config, stream_mode="values"):

                if 'sections' in event and not plan_done:
                    plan_done = True
                    names = [s.name for s in event['sections']]
                    yield sse("plan", {
                        "phase":    "plan_ready",
                        "message":  f"✅ Plan ready — {len(names)} sections identified",
                        "sections": names,
                    })
                    await asyncio.sleep(0.05)

                if 'completed_sections' in event:
                    for section in event['completed_sections']:
                        if section.name not in sections_logged:
                            sections_logged.add(section.name)
                            preview = (section.content or "")
                            if len(preview) > 200:
                                preview = preview[:200] + "…"
                            yield sse("section_complete", {
                                "phase":        "section_research" if getattr(section, 'research', True) else "section_final",
                                "message":      f"✍️  Written: **{section.name}**",
                                "section_name": section.name,
                                "preview":      preview,
                            })
                            await asyncio.sleep(0.05)

                if 'report_sections_from_research' in event and 'final_report' not in event:
                    yield sse("status", {"phase": "formatting", "message": "📐 Formatting & writing intro/conclusion…"})
                    await asyncio.sleep(0.05)

                if 'final_report' in event and event['final_report']:
                    final_report = event['final_report']

            if final_report:
                yield sse("complete", {
                    "phase":   "done",
                    "message": "🎉 Research report complete!",
                    "report":  final_report,
                })
            else:
                yield sse("error", {"message": "Agent completed but no report was generated. Try a different topic."})

        except Exception as exc:
            yield sse("error", {"message": str(exc), "details": traceback.format_exc()})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ── LinkedIn helpers ─────────────────────────────────────────────────────────

BOLD_MAP = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    "𝗔𝗕𝗖𝗗𝗘𝗙𝗚𝗛𝗜𝗝𝗞𝗟𝗠𝗡𝗢𝗣𝗤𝗥𝗦𝗧𝗨𝗩𝗪𝗫𝗬𝗭𝗮𝗯𝗰𝗱𝗲𝗳𝗴𝗵𝗶𝗷𝗸𝗹𝗺𝗻𝗼𝗽𝗾𝗿𝘀𝘁𝘂𝘃𝘄𝘅𝘆𝘇𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵"
)

ITALIC_MAP = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "𝘈𝘉𝘊𝘋𝘌𝘍𝘎𝘏𝘐𝘑𝘒𝘓𝘔𝘕𝘖𝘗𝘘𝘙𝘚𝘛𝘜𝘝𝘞𝘟𝘠𝘡𝘢𝘣𝘤𝘥𝘦𝘧𝘨𝘩𝘪𝘫𝘬𝘭𝘮𝘯𝘰𝘱𝘲𝘳𝘴𝘵𝘶𝘷𝘸𝘹𝘺𝘻"
)


def md_to_linkedin(text: str) -> str:
    """Convert markdown-ish LLM output to LinkedIn-safe unicode formatted text."""
    lines = text.split("\n")
    out = []
    for line in lines:
        # ## Heading → bold unicode
        if re.match(r"^#{1,3}\s+", line):
            heading = re.sub(r"^#{1,3}\s+", "", line).strip()
            out.append(heading.translate(BOLD_MAP))
        # **bold** → unicode bold
        elif "**" in line:
            def bold_repl(m):
                return m.group(1).translate(BOLD_MAP)
            line = re.sub(r"\*\*(.+?)\*\*", bold_repl, line)
            # _italic_ → unicode italic
            def italic_repl(m):
                return m.group(1).translate(ITALIC_MAP)
            line = re.sub(r"_(.+?)_", italic_repl, line)
            out.append(line)
        # - bullet → emoji bullet
        elif re.match(r"^[-*]\s+", line):
            content = re.sub(r"^[-*]\s+", "", line)
            out.append(f"🔹 {content}")
        else:
            def italic_repl2(m):
                return m.group(1).translate(ITALIC_MAP)
            line = re.sub(r"_(.+?)_", italic_repl2, line)
            out.append(line)
    return "\n".join(out)


LINKEDIN_FORMAT_PROMPT = """You are a friendly LinkedIn writer who explains complex topics in simple, everyday language that anyone can understand — not just experts.

Transform the research report below into a LinkedIn post that feels like advice from a smart friend, not a textbook.

STRICT RULES:
1. Total length: 2200–2600 characters MAX
2. Write like you're talking to a curious person, not a PhD — avoid acronyms, buzzwords, and jargon. If you must use a technical term, explain it in plain words right after.
3. Use short sentences. Max 20 words per sentence.
4. Structure:
   - 1 attention-grabbing opening line (a surprising fact, bold statement, or relatable question)
   - blank line
   - 2-3 short paragraphs explaining the key ideas simply (2-3 lines each), separated by blank lines
   - blank line
   - "Here's what matters:" heading, then 4-5 bullet points in plain language (use - prefix)
   - blank line
   - 1-2 line closing thought that feels warm and human, with a simple question to invite comments
   - blank line
   - 5-7 relevant hashtags on the last line
5. Use **bold** for key ideas only (not every other word)
6. Use _italic_ sparingly for one or two important phrases
7. Tone: conversational, warm, curious — like a LinkedIn post people actually enjoy reading
8. Do NOT use words like: leverage, utilize, paradigm, synergy, robust, scalable, ecosystem, holistic, framework, methodology, cutting-edge, state-of-the-art
9. Do NOT start with "Introduction", "In this report", or "I recently read"

Research Report:
{report}"""


class LinkedInFormatRequest(BaseModel):
    report: str
    provider: str = "groq"
    model: Optional[str] = None
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    euron_api_key: Optional[str] = None
    api_base: Optional[str] = None


class LinkedInPostRequest(BaseModel):
    text: str


@app.post("/api/linkedin/format")
async def linkedin_format(req: LinkedInFormatRequest):
    """Use the configured LLM to summarise + format the report for LinkedIn."""
    # resolve keys
    gk = req.groq_api_key   or os.getenv("GROQ_API_KEY", "")
    mk = req.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    ok = req.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    ek = req.euron_api_key  or os.getenv("EURON_API_KEY", "")
    if gk: os.environ["GROQ_API_KEY"]   = gk
    if mk: os.environ["GEMINI_API_KEY"] = mk
    if ok: os.environ["OPENAI_API_KEY"] = ok

    try:
        llm = get_llm(req.provider, req.model, api_base=req.api_base, euron_key=ek)
        from langchain_core.messages import HumanMessage
        prompt = LINKEDIN_FORMAT_PROMPT.format(report=req.report[:12000])
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
        linkedin_text = md_to_linkedin(raw.strip())
        linkedin_text += "\n\n🔬 Researched and published by ResearchAI — deep research, fully automated.\nhttps://researchai-3706.onrender.com"
        return {"text": linkedin_text, "char_count": len(linkedin_text)}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/linkedin/post")
async def linkedin_post(req: LinkedInPostRequest):
    """Post text to LinkedIn using stored credentials."""
    token  = os.getenv("LINKEDIN_ACCESS_TOKEN", "")
    author = os.getenv("LINKEDIN_AUTHOR_URN", "")
    if not token or not author:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="LinkedIn credentials not configured in .env")

    # Use the v2 UGC Posts API — stable, no versioning header required
    payload = {
        "author": author,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": req.text},
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0"
    }
    resp = _requests.post("https://api.linkedin.com/v2/ugcPosts", json=payload, headers=headers)
    if resp.status_code in (200, 201):
        return {"success": True, "message": "Posted to LinkedIn successfully!"}
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=resp.status_code, detail=resp.text)


# ── Serve Blog UI ────────────────────────────────────────────────────────────
@app.get("/blog", response_class=HTMLResponse)
async def blog_ui():
    html_path = Path(__file__).parent / "blog.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)


# ── Blog endpoints ────────────────────────────────────────────────────────────

BLOG_WRITER_PROMPT = """You are an expert technical blogger. Convert the research report below into a detailed, engaging blog post for developers on dev.to.

REQUIREMENTS:
1. Length: 1500–2500 words
2. Title: SEO-friendly, compelling (return as first line: # Title)
3. Structure (use ## for sections):
   - Hook intro paragraph — why this topic matters RIGHT NOW
   - 5–7 main sections covering the key ideas
   - "Key Takeaways" section at the end (bullet list)
   - Brief closing paragraph with a question to readers
4. Code examples:
   - Include at least 2–3 real, working code snippets
   - Use fenced code blocks with the correct language tag (```python, ```javascript, etc.)
   - Make code practical and copy-pasteable
5. Diagrams:
   - Include 1 Mermaid diagram where it adds clarity (architecture, flow, or sequence)
   - Use ```mermaid fenced blocks
6. Formatting:
   - Use **bold** for key terms on first use
   - Use `inline code` for technical terms, function names, file names
   - Use blockquotes (>) for important callouts or tips
   - Use tables where comparing options
7. Tone: Clear, friendly, practical — like a senior engineer sharing what they learned
8. No buzzwords: avoid "leverage", "utilize", "paradigm", "synergy", "robust", "cutting-edge"
9. Return ONLY the markdown blog post. No preamble or explanation.
10. After the post, on a new line write: TAGS: tag1, tag2, tag3, tag4
    CRITICAL: Tags must be lowercase alphanumeric ONLY — no hyphens, no spaces, no special chars.
    Good: ai, python, webdev, tutorial, machinelearning, javascript, beginners, devops
    Bad:  artificial-intelligence, machine-learning, future-of-tech

Research Report:
{report}"""


class BlogGenerateRequest(BaseModel):
    report: str
    title: Optional[str] = None
    provider: str = "groq"
    model: Optional[str] = None
    api_base: Optional[str] = None
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    euron_api_key: Optional[str] = None


class BlogPublishRequest(BaseModel):
    markdown: str
    title: str
    tags: list = []
    cover_image: Optional[str] = None
    devto_api_key: str
    published: bool = True


@app.post("/api/blog/generate")
async def blog_generate(req: BlogGenerateRequest):
    """Use the configured LLM to convert the research report into a full blog post."""
    gk = req.groq_api_key   or os.getenv("GROQ_API_KEY", "")
    mk = req.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    ok = req.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    ek = req.euron_api_key  or os.getenv("EURON_API_KEY", "")
    if gk: os.environ["GROQ_API_KEY"]   = gk
    if mk: os.environ["GEMINI_API_KEY"] = mk
    if ok: os.environ["OPENAI_API_KEY"] = ok

    try:
        llm = get_llm(req.provider, req.model, api_base=req.api_base, euron_key=ek)
        from langchain_core.messages import HumanMessage
        prompt = BLOG_WRITER_PROMPT.format(report=req.report[:14000])
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
        raw = raw.strip()

        # Extract TAGS line if present
        tags = []
        if "\nTAGS:" in raw:
            parts = raw.rsplit("\nTAGS:", 1)
            raw = parts[0].strip()
            tags = [t.strip().lower().replace(" ", "") for t in parts[1].split(",")][:4]

        # Extract title from first # heading
        title = req.title or ""
        lines = raw.split("\n")
        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip()

        word_count = len(raw.split())
        return {"markdown": raw, "title": title, "tags": tags, "word_count": word_count}

    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/blog/publish")
async def blog_publish(req: BlogPublishRequest):
    """Publish a blog post to dev.to via their API."""
    # dev.to tags: lowercase alphanumeric only, no hyphens/spaces
    clean_tags = [re.sub(r'[^a-z0-9]', '', t.lower()) for t in req.tags]
    clean_tags = [t for t in clean_tags if t][:4]

    _attribution = "\n\n---\n*🔬 Researched and published by [ResearchAI](https://researchai-3706.onrender.com) — deep research, fully automated.*"
    article_body = {
        "title":         req.title,
        "body_markdown": req.markdown + _attribution,
        "published":     req.published,
        "tags":          clean_tags,
    }
    if req.cover_image:
        article_body["main_image"] = req.cover_image

    payload = {"article": article_body}
    headers = {
        "api-key":      req.devto_api_key,
        "Content-Type": "application/json",
        "User-Agent":   "ResearchAI-BlogPublisher/1.0",
    }

    # fall back to .env key if none supplied
    if not req.devto_api_key:
        req.devto_api_key = os.getenv("DEVTO_API_KEY", "")
    if not req.devto_api_key:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="dev.to API key not configured")

    resp = _requests.post("https://dev.to/api/articles", json=payload, headers=headers)

    if resp.status_code in (200, 201):
        data = resp.json()
        return {
            "success": True,
            "url":     data.get("url", "https://dev.to"),
            "id":      data.get("id"),
            "slug":    data.get("slug"),
        }
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=resp.status_code, detail=resp.text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
