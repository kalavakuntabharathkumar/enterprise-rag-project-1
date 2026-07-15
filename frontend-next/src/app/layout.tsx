import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Document Assistant",
  description: "Ask questions about your uploaded documents.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
        <header
          style={{
            height: 56,
            minHeight: 56,
            borderBottom: "1px solid #e5e5e5",
            background: "#fff",
            display: "flex",
            alignItems: "center",
            padding: "0 24px",
            gap: 24,
            flexShrink: 0,
          }}
        >
          <Link
            href="/"
            style={{
              fontWeight: 600,
              fontSize: 15,
              letterSpacing: "-0.01em",
              color: "#111",
            }}
          >
            RAG Document Assistant
          </Link>

          <nav style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
            <Link href="/" className="nav-link">Chat</Link>
            <Link href="/analytics" className="nav-link">Analytics</Link>
          </nav>
        </header>

        <main style={{ flex: 1, minHeight: 0 }}>{children}</main>
      </body>
    </html>
  );
}
