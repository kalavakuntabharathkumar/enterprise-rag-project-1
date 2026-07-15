"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { askQuestion, uploadPdf, ApiClientError } from "@/lib/api";
import type { AskResponse } from "@/types/api";

// ── Types ────────────────────────────────────────────────────────────────

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources?: string[];
  confidence?: number;
  intent?: string;
  query_type?: string;
  error?: boolean;
}

// ── Helpers ──────────────────────────────────────────────────────────────

function confidenceLabel(score: number): { label: string; color: string; bg: string } {
  if (score >= 0.75) return { label: `${Math.round(score * 100)}%`, color: "#166534", bg: "#dcfce7" };
  if (score >= 0.45) return { label: `${Math.round(score * 100)}%`, color: "#92400e", bg: "#fef3c7" };
  return { label: `${Math.round(score * 100)}%`, color: "#991b1b", bg: "#fee2e2" };
}

function shortSource(src: string): string {
  // Show only the filename, not the full path
  return src.split("/").pop() ?? src;
}

// ── Sub-components ────────────────────────────────────────────────────────

function UserBubble({ text }: { text: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 16 }}>
      <div
        style={{
          maxWidth: 600,
          padding: "10px 14px",
          background: "#0070f3",
          color: "#fff",
          borderRadius: "16px 16px 4px 16px",
          fontSize: 14,
          lineHeight: 1.55,
          wordBreak: "break-word",
        }}
      >
        {text}
      </div>
    </div>
  );
}

function AssistantBubble({ message }: { message: Message }) {
  const conf =
    message.confidence !== undefined ? confidenceLabel(message.confidence) : null;

  return (
    <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 16 }}>
      <div style={{ maxWidth: 660 }}>
        <div
          style={{
            padding: "12px 16px",
            background: message.error ? "#fff5f5" : "#fff",
            border: `1px solid ${message.error ? "#fca5a5" : "#e5e5e5"}`,
            borderRadius: "16px 16px 16px 4px",
            fontSize: 14,
            lineHeight: 1.65,
            color: message.error ? "#991b1b" : "#111",
            wordBreak: "break-word",
            whiteSpace: "pre-wrap",
          }}
        >
          {message.text}
        </div>

        {/* Sources + confidence row */}
        {((message.sources && message.sources.length > 0) || conf) && (
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              alignItems: "center",
              gap: 6,
              marginTop: 6,
              paddingLeft: 4,
            }}
          >
            {message.sources && message.sources.length > 0 && (
              <>
                <span style={{ fontSize: 11, color: "#9ca3af", marginRight: 2 }}>
                  Sources:
                </span>
                {message.sources.map((src, i) => (
                  <span
                    key={i}
                    title={src}
                    style={{
                      display: "inline-block",
                      padding: "2px 8px",
                      background: "#f0f0f0",
                      borderRadius: 4,
                      fontSize: 11,
                      color: "#555",
                      fontFamily: "monospace",
                      maxWidth: 220,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {shortSource(src)}
                  </span>
                ))}
              </>
            )}

            {conf && (
              <span
                title={`Confidence: ${message.confidence}`}
                style={{
                  marginLeft: "auto",
                  padding: "2px 8px",
                  background: conf.bg,
                  color: conf.color,
                  borderRadius: 4,
                  fontSize: 11,
                  fontWeight: 600,
                }}
              >
                {conf.label} confidence
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 16 }}>
      <div
        style={{
          padding: "12px 16px",
          background: "#fff",
          border: "1px solid #e5e5e5",
          borderRadius: "16px 16px 16px 4px",
          display: "flex",
          gap: 5,
          alignItems: "center",
        }}
      >
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: "#9ca3af",
              display: "inline-block",
              animation: `bounce 1.2s infinite`,
              animationDelay: `${i * 0.2}s`,
            }}
          />
        ))}
        <style>{`
          @keyframes bounce {
            0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
            40% { transform: translateY(-5px); opacity: 1; }
          }
        `}</style>
      </div>
    </div>
  );
}

function EmptyState({ onUpload }: { onUpload: (file: File) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        gap: 12,
        color: "#9ca3af",
        padding: 32,
        textAlign: "center",
      }}
    >
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
      <p style={{ fontSize: 15, fontWeight: 500, color: "#6b7280" }}>
        No documents uploaded yet
      </p>
      <p style={{ fontSize: 13, maxWidth: 320 }}>
        Upload a PDF to index it, then ask questions about its content.
      </p>
      <button
        onClick={() => inputRef.current?.click()}
        style={{
          marginTop: 4,
          padding: "8px 18px",
          background: "#0070f3",
          color: "#fff",
          border: "none",
          borderRadius: 7,
          fontSize: 13,
          fontWeight: 500,
          cursor: "pointer",
        }}
      >
        Upload PDF
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onUpload(file);
        }}
      />
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [sessionId] = useState<string>(() =>
    typeof crypto !== "undefined" ? crypto.randomUUID() : Math.random().toString(36).slice(2),
  );

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const addMessage = (msg: Omit<Message, "id">) =>
    setMessages((prev) => [...prev, { ...msg, id: crypto.randomUUID() }]);

  const handleSend = useCallback(async () => {
    const question = input.trim();
    if (!question || loading) return;

    setInput("");
    addMessage({ role: "user", text: question });
    setLoading(true);

    try {
      const resp: AskResponse = await askQuestion(question, sessionId);
      addMessage({
        role: "assistant",
        text: resp.answer,
        sources: resp.sources,
        confidence: resp.confidence,
        intent: resp.intent,
        query_type: resp.query_type,
      });
    } catch (err) {
      const detail =
        err instanceof ApiClientError
          ? err.detail
          : "Could not reach the backend. Is the API server running?";
      addMessage({ role: "assistant", text: detail, error: true });
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [input, loading, sessionId]);

  const handleUpload = async (file: File) => {
    setUploadStatus("Uploading…");
    try {
      const result = await uploadPdf(file);
      setUploadStatus(`✓ Indexed ${result.chunks_indexed} chunks from ${file.name}`);
      addMessage({
        role: "assistant",
        text: `Indexed **${file.name}** — ${result.chunks_indexed} chunk${result.chunks_indexed !== 1 ? "s" : ""} ready. Ask me anything about it.`,
      });
    } catch (err) {
      const detail =
        err instanceof ApiClientError ? err.detail : "Upload failed";
      setUploadStatus(`✗ ${detail}`);
    }
    setTimeout(() => setUploadStatus(null), 4000);
  };

  const hasMessages = messages.length > 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>

      {/* Upload bar */}
      <div
        style={{
          background: "#fff",
          borderBottom: "1px solid #e5e5e5",
          padding: "8px 20px",
          display: "flex",
          alignItems: "center",
          gap: 12,
          flexShrink: 0,
        }}
      >
        <button
          onClick={() => fileInputRef.current?.click()}
          style={{
            padding: "5px 14px",
            border: "1px solid #d1d5db",
            borderRadius: 6,
            background: "#fff",
            fontSize: 12,
            fontWeight: 500,
            cursor: "pointer",
            color: "#374151",
            display: "flex",
            alignItems: "center",
            gap: 5,
          }}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Upload PDF
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          style={{ display: "none" }}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleUpload(file);
            e.target.value = "";
          }}
        />
        {uploadStatus && (
          <span
            style={{
              fontSize: 12,
              color: uploadStatus.startsWith("✓") ? "#166534" : uploadStatus.startsWith("✗") ? "#991b1b" : "#555",
            }}
          >
            {uploadStatus}
          </span>
        )}
        <span style={{ marginLeft: "auto", fontSize: 11, color: "#9ca3af" }}>
          Session: {sessionId.slice(0, 8)}…
        </span>
      </div>

      {/* Messages */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: hasMessages ? "20px" : 0,
        }}
      >
        <div style={{ maxWidth: 720, margin: "0 auto", height: hasMessages ? "auto" : "100%" }}>
          {!hasMessages ? (
            <EmptyState onUpload={handleUpload} />
          ) : (
            <>
              {messages.map((msg) =>
                msg.role === "user" ? (
                  <UserBubble key={msg.id} text={msg.text} />
                ) : (
                  <AssistantBubble key={msg.id} message={msg} />
                ),
              )}
              {loading && <TypingIndicator />}
              <div ref={bottomRef} />
            </>
          )}
        </div>
      </div>

      {/* Input bar */}
      <div
        style={{
          borderTop: "1px solid #e5e5e5",
          background: "#fff",
          padding: "12px 20px",
          flexShrink: 0,
        }}
      >
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          style={{
            display: "flex",
            gap: 8,
            maxWidth: 720,
            margin: "0 auto",
          }}
        >
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Ask a question about your documents…"
            disabled={loading}
            autoFocus
            style={{
              flex: 1,
              padding: "10px 14px",
              border: "1px solid #d1d5db",
              borderRadius: 8,
              fontSize: 14,
              outline: "none",
              background: loading ? "#f9fafb" : "#fff",
              color: "#111",
              transition: "border-color 0.15s",
            }}
            onFocus={(e) => (e.currentTarget.style.borderColor = "#0070f3")}
            onBlur={(e) => (e.currentTarget.style.borderColor = "#d1d5db")}
          />
          <button
            type="submit"
            disabled={!input.trim() || loading}
            style={{
              padding: "10px 20px",
              background: "#0070f3",
              color: "#fff",
              border: "none",
              borderRadius: 8,
              fontSize: 14,
              fontWeight: 500,
              cursor: !input.trim() || loading ? "not-allowed" : "pointer",
              opacity: !input.trim() || loading ? 0.45 : 1,
              transition: "opacity 0.15s",
              whiteSpace: "nowrap",
            }}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
