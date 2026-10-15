/*
 * Embeddable support chat widget.
 *
 *   <script src="https://support.example.com/widget.js" data-title="Help"></script>
 *
 * Attributes: data-api (defaults to the origin the script was loaded from), data-title,
 * data-greeting, data-accent.
 */
(function () {
  "use strict";

  var script = document.currentScript;
  if (!script) return;

  var api = (script.dataset.api || new URL(script.src).origin).replace(/\/$/, "");
  var title = script.dataset.title || "Support";
  var greeting = script.dataset.greeting || "Hi! Ask me anything about our services.";
  var POLL_MS = 15000;
  var store = window.localStorage;

  var sessionId = store.getItem("sa-session");
  if (!sessionId) {
    sessionId = crypto.randomUUID().replace(/-/g, "");
    store.setItem("sa-session", sessionId);
  }
  var lastReplyId = Number(store.getItem("sa-last-reply") || 0);
  var pollTimer = null;
  var busy = false;

  var link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = api + "/widget.css";
  document.head.appendChild(link);

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  var root = el("div", "sa-widget");
  if (script.dataset.accent) root.style.setProperty("--sa-accent", script.dataset.accent);

  var launcher = el("button", "sa-launcher", "Chat");
  launcher.type = "button";
  launcher.setAttribute("aria-label", "Open support chat");
  launcher.setAttribute("aria-expanded", "false");

  var panel = el("section", "sa-panel");
  panel.hidden = true;
  panel.setAttribute("aria-label", title);

  var header = el("header", "sa-header");
  var closeButton = el("button", "sa-close", "×");
  closeButton.type = "button";
  closeButton.setAttribute("aria-label", "Close chat");
  header.append(el("span", "sa-title", title), closeButton);

  var log = el("div", "sa-messages");
  log.setAttribute("role", "log");
  log.setAttribute("aria-live", "polite");

  var form = el("form", "sa-form");
  var input = el("textarea", "sa-input");
  input.rows = 1;
  input.maxLength = 1000;
  input.placeholder = "Type your question";
  input.setAttribute("aria-label", "Your message");
  var send = el("button", "sa-send", "Send");
  send.type = "submit";
  form.append(input, send);

  panel.append(header, log, form);
  root.append(panel, launcher);
  document.body.appendChild(root);

  function addMessage(kind, text) {
    var bubble = el("div", "sa-message sa-" + kind);
    bubble.appendChild(el("div", "sa-text", text));
    log.appendChild(bubble);
    log.scrollTop = log.scrollHeight;
    return bubble;
  }

  function setCitations(bubble, citations) {
    if (!citations || !citations.length) return;
    var list = el("ul", "sa-sources");
    citations.forEach(function (c) {
      var label = "[" + c.index + "] " + c.title + (c.page ? ", p. " + c.page : "");
      list.appendChild(el("li", "", label));
    });
    bubble.appendChild(list);
  }

  function toggle(open) {
    panel.hidden = !open;
    launcher.setAttribute("aria-expanded", String(open));
    if (open) input.focus();
    else launcher.focus();
  }

  function parseEvent(block) {
    var name = "message";
    var data = "";
    block.split("\n").forEach(function (line) {
      if (line.indexOf("event: ") === 0) name = line.slice(7);
      else if (line.indexOf("data: ") === 0) data += line.slice(6);
    });
    return { name: name, data: data ? JSON.parse(data) : {} };
  }

  async function ask(text) {
    var bubble = addMessage("bot", "");
    var textNode = bubble.firstChild;
    bubble.classList.add("sa-pending");
    var received = false;

    try {
      var response = await fetch(api + "/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });
      if (!response.ok) {
        var problem = await response.json().catch(function () { return {}; });
        throw new Error(problem.detail || "Something went wrong. Please try again.");
      }

      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";
      for (;;) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        var blocks = buffer.split("\n\n");
        buffer = blocks.pop();
        blocks.forEach(function (block) {
          var event = parseEvent(block);
          if (event.name === "token") {
            received = true;
            bubble.classList.remove("sa-pending");
            textNode.textContent += event.data.text;
          } else if (event.name === "done") {
            textNode.textContent = event.data.answer;
            setCitations(bubble, event.data.citations);
            if (event.data.ticket_id) startPolling();
          } else if (event.name === "error") {
            throw new Error(event.data.detail);
          }
        });
        log.scrollTop = log.scrollHeight;
      }
    } catch (error) {
      if (!received) textNode.textContent = "";
      bubble.classList.add("sa-error");
      textNode.textContent += (textNode.textContent ? "\n" : "") + error.message;
    } finally {
      bubble.classList.remove("sa-pending");
    }
  }

  async function poll() {
    try {
      var url = api + "/api/sessions/" + sessionId + "/operator-replies?after=" + lastReplyId;
      var response = await fetch(url);
      if (!response.ok) return;
      var replies = await response.json();
      replies.forEach(function (reply) {
        addMessage("operator", reply.text);
        lastReplyId = Math.max(lastReplyId, reply.id);
      });
      store.setItem("sa-last-reply", String(lastReplyId));
    } catch (error) {
      /* transient network failure: try again on the next tick */
    }
  }

  function startPolling() {
    store.setItem("sa-handoff", "1");
    if (!pollTimer) pollTimer = setInterval(poll, POLL_MS);
  }

  launcher.addEventListener("click", function () { toggle(panel.hidden); });
  closeButton.addEventListener("click", function () { toggle(false); });
  panel.addEventListener("keydown", function (event) {
    if (event.key === "Escape") toggle(false);
  });
  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    var text = input.value.trim();
    if (!text || busy) return;
    input.value = "";
    busy = true;
    send.disabled = true;
    addMessage("user", text);
    await ask(text);
    busy = false;
    send.disabled = false;
    input.focus();
  });

  addMessage("bot", greeting);
  if (store.getItem("sa-handoff")) {
    startPolling();
    poll();
  }
})();
