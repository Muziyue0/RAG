import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from app.schemas import ChatRequest, ChatResponse
from app.rag_service import rag_service
from app.config import settings


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Domain Manual RAG Service",
    description="基于重排与滑动窗口记忆的高精度检索后端服务",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check():
    if rag_service is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")
    return {"status": "ready"}


@app.get("/", response_class=HTMLResponse)
def chat_page():
    return """
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>RAG 对话</title>
      <style>
        :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                 BlinkMacSystemFont, "Segoe UI", sans-serif; }
        * { box-sizing: border-box; }
        body { min-height: 100vh; margin: 0; padding: 32px 16px;
               color: #172033; background: linear-gradient(135deg, #eef4ff, #f8fafc 45%, #eefbf8); }
        main { width: min(920px, 100%); margin: 0 auto; }
        header { margin-bottom: 20px; }
        .eyebrow { margin: 0 0 6px; color: #2563eb; font-size: 12px;
                   font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
        h1 { margin: 0; color: #0f172a; font-size: clamp(26px, 5vw, 38px); letter-spacing: -.03em; }
        .subtitle { margin: 8px 0 0; color: #64748b; }
        #messages { display: flex; min-height: 440px; max-height: 68vh; padding: 22px;
                    flex-direction: column; gap: 14px; overflow-y: auto; border: 1px solid #dbe5f0;
                    border-radius: 20px; background: rgba(255,255,255,.82);
                    box-shadow: 0 18px 50px rgba(30, 64, 175, .10); }
        .empty-state { margin: auto; color: #94a3b8; text-align: center; }
        .empty-state strong { display: block; margin-bottom: 6px; color: #475569; font-size: 18px; }
        .message { max-width: 86%; padding: 13px 16px; border: 1px solid transparent;
                   border-radius: 16px; line-height: 1.7; overflow-wrap: anywhere; }
        .message.user { align-self: flex-end; color: #eff6ff; background: #2563eb;
                        border-bottom-right-radius: 5px; }
        .message.assistant { align-self: flex-start; color: #273449; background: #f8fafc;
                             border-color: #e2e8f0; border-bottom-left-radius: 5px; }
        .role { display: block; margin-bottom: 3px; font-size: 12px; font-weight: 700;
                opacity: .72; }
        .content p { margin: 0 0 8px; }
        .content p:last-child { margin-bottom: 0; }
        .content h1, .content h2, .content h3, .content h4, .content h5, .content h6 {
          margin: 14px 0 8px; color: #0f766e; line-height: 1.35;
        }
        .content h1 { font-size: 1.45em; }
        .content h2 { font-size: 1.3em; }
        .content h3 { font-size: 1.18em; }
        .content hr { margin: 14px 0; border: 0; border-top: 1px solid #cbd5e1; }
        .content table { width: 100%; margin: 10px 0; border-collapse: collapse; font-size: .95em; }
        .content th, .content td { padding: 8px 10px; border: 1px solid #cbd5e1;
                                   text-align: left; vertical-align: top; }
        .content th { color: #0f766e; background: #ecfeff; font-weight: 750; }
        .content strong { color: #0f766e; font-weight: 750; }
        .user .content strong { color: #fff; }
        .content ul { margin: 5px 0 8px; padding-left: 22px; }
        .content li { margin: 2px 0; }
        .content code { padding: 2px 5px; border-radius: 5px; color: #be123c; background: #ffe4e6;
                        font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .92em; }
        form { display: flex; gap: 10px; margin-top: 14px; }
        input { min-width: 0; flex: 1; padding: 14px 16px; color: #172033; border: 1px solid #cbd5e1;
                border-radius: 12px; outline: none; background: white; font-size: 16px;
                transition: border-color .2s, box-shadow .2s; }
        input:focus { border-color: #60a5fa; box-shadow: 0 0 0 4px rgba(96,165,250,.2); }
        button { padding: 0 22px; border: 0; border-radius: 12px; color: white;
                 background: #2563eb; font-size: 15px; font-weight: 700; cursor: pointer;
                 transition: background .2s, transform .2s; }
        button:hover:not(:disabled) { background: #1d4ed8; transform: translateY(-1px); }
        button:disabled { background: #94a3b8; cursor: wait; }
        @media (max-width: 560px) {
          body { padding: 20px 10px; }
          #messages { min-height: 55vh; padding: 14px; border-radius: 16px; }
          .message { max-width: 94%; }
          form { flex-direction: column; }
          button { min-height: 48px; }
        }
      </style>
    </head>
    <body>
      <main>
        <header>
          <p class="eyebrow">Domain Manual RAG</p>
          <h1>智能知识库问答</h1>
          <p class="subtitle">基于参考资料检索，为你的问题生成可靠回答</p>
        </header>
        <div id="messages">
          <div class="empty-state"><strong>开始一段对话</strong>输入问题，模型会根据知识库内容回答。</div>
        </div>
      <form id="chat-form">
        <input id="question" autocomplete="off" placeholder="请输入问题..." required>
        <button id="send" type="submit">发送</button>
      </form>
      </main>
      <script>
        const messages = document.getElementById("messages");
        const question = document.getElementById("question");
        const send = document.getElementById("send");
        const sessionId = "web_session";

        function escapeHtml(text) {
          return text.replace(/[&<>"']/g, (character) => ({
            "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
          }[character]));
        }

        function renderMarkdown(text) {
          const lines = escapeHtml(text).split("\\n");
          const output = [];
          let listItems = [];
          const flushList = () => {
            if (listItems.length) {
              output.push(`<ul>${listItems.join("")}</ul>`);
              listItems = [];
            }
          };
          const inline = (line) => line
            .replace(/\\*\\*(.+?)\\*\\*/g, "<strong>$1</strong>")
            .replace(/`([^`]+)`/g, "<code>$1</code>");
          const tableCells = (line) => line.trim().replace(/^\\||\\|$/g, "").split("|")
            .map((cell) => inline(cell.trim()));
          const isTableSeparator = (line) => /^\\s*\\|?\\s*:?-{3,}:?\\s*(\\|\\s*:?-{3,}:?\\s*)+\\|?\\s*$/.test(line);
          const renderTable = (header, rows) => {
            const head = tableCells(header).map((cell) => `<th>${cell}</th>`).join("");
            const body = rows.map((row) => {
              const cells = tableCells(row).map((cell) => `<td>${cell}</td>`).join("");
              return `<tr>${cells}</tr>`;
            }).join("");
            return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
          };

          for (let index = 0; index < lines.length; index += 1) {
            const line = lines[index];
            const listMatch = line.match(/^\\s*[-*]\\s+(.+)/);
            if (line.includes("|") && isTableSeparator(lines[index + 1] || "")) {
              flushList();
              const rows = [];
              index += 2;
              while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
                rows.push(lines[index]);
                index += 1;
              }
              output.push(renderTable(line, rows));
              index -= 1;
            } else if (listMatch) {
              listItems.push(`<li>${inline(listMatch[1])}</li>`);
            } else if (/^#{1,6}\\s+/.test(line)) {
              flushList();
              const heading = line.match(/^(#{1,6})\\s+(.+)/);
              const level = heading[1].length;
              output.push(`<h${level}>${inline(heading[2])}</h${level}>`);
            } else if (/^\\s*((\\*\\s*){3,}|(-\\s*){3,}|(_\\s*){3,})$/.test(line)) {
              flushList();
              output.push("<hr>");
            } else if (line.trim()) {
              flushList();
              output.push(`<p>${inline(line)}</p>`);
            } else {
              flushList();
            }
          }
          flushList();
          return output.join("");
        }

        function appendMessage(role, text) {
          const item = document.createElement("div");
          item.className = `message ${role}`;
          const roleLabel = document.createElement("span");
          roleLabel.className = "role";
          roleLabel.textContent = role === "user" ? "你" : "模型";
          const content = document.createElement("div");
          content.className = "content";
          content.innerHTML = role === "assistant" ? renderMarkdown(text) : escapeHtml(text);
          item.append(roleLabel, content);
          messages.appendChild(item);
          messages.scrollTop = messages.scrollHeight;
          return content;
        }

        document.getElementById("chat-form").addEventListener("submit", async (event) => {
          event.preventDefault();
          const text = question.value.trim();
          if (!text) return;
          messages.querySelector(".empty-state")?.remove();
          appendMessage("user", text);
          question.value = "";
          send.disabled = true;
          const pending = appendMessage("assistant", "正在思考...");
          try {
            const response = await fetch("/api/v1/chat", {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({question: text, session_id: sessionId})
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "请求失败");
            pending.innerHTML = renderMarkdown(data.answer);
          } catch (error) {
            pending.textContent = `请求失败：${error.message}`;
          } finally {
            send.disabled = false;
            question.focus();
          }
        });
      </script>
    </body>
    </html>
    """


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if len(request.question.strip()) > settings.MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"问题长度不能超过 {settings.MAX_QUESTION_LENGTH} 个字符。",
        )
    if len(request.session_id) > settings.MAX_SESSION_ID_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"会话标识长度不能超过 {settings.MAX_SESSION_ID_LENGTH} 个字符。",
        )
    try:
        answer = rag_service.execute_query(
            question=request.question, session_id=request.session_id
        )
        return ChatResponse(session_id=request.session_id, answer=answer)
    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=500, detail="服务暂时不可用，请稍后重试。"
        )
