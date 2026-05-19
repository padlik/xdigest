#!/usr/bin/env python3
"""
md_to_email.py — Converts a Markdown file to a professional, email-optimized HTML file.
Usage: python md_to_email.py input.md [output.html]
"""

import re
import sys
import os
from pathlib import Path
from datetime import datetime


# ── Colour palette & design tokens ──────────────────────────────────────────
PALETTE = {
    "bg":           "#f4f5f7",
    "card":         "#ffffff",
    "primary":      "#1a1f36",      # deep navy
    "accent":       "#3d5afe",      # vivid indigo
    "accent_light": "#e8ecff",
    "muted":        "#6b7280",
    "border":       "#e5e7eb",
    "code_bg":      "#f1f3f9",
    "bullet":       "#3d5afe",
    "h2_bar":       "#3d5afe",
    "tag_bg":       "#e8ecff",
    "tag_fg":       "#3730a3",
}


# ── Inline Markdown → HTML ───────────────────────────────────────────────────
def inline_md(text: str) -> str:
    """Convert inline Markdown (bold, italic, code, links) to HTML."""
    # Escape HTML entities first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Bold + italic
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # Inline code
    text = re.sub(
        r"`([^`]+)`",
        lambda m: (
            f'<code style="font-family:\'SF Mono\',\'Fira Mono\',monospace;'
            f'background:{PALETTE["code_bg"]};color:{PALETTE["accent"]};'
            f'padding:2px 6px;border-radius:4px;font-size:0.88em;">'
            f'{m.group(1)}</code>'
        ),
        text,
    )
    # Links  [text](url)
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^\)]+)\)",
        lambda m: (
            f'<a href="{m.group(2)}" style="color:{PALETTE["accent"]};'
            f'text-decoration:none;border-bottom:1px solid {PALETTE["accent"]};">'
            f'{m.group(1)}</a>'
        ),
        text,
    )
    return text


# ── Block-level Markdown parser ──────────────────────────────────────────────
def parse_blocks(md: str) -> list[dict]:
    """
    Tokenise Markdown into a list of block dicts:
      {type: h1|h2|h3|hr|blockquote|ul|ol|code_block|p, ...}
    """
    lines = md.splitlines()
    blocks: list[dict] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Fenced code block
        if line.startswith("```"):
            lang = line[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            blocks.append({"type": "code_block", "lang": lang, "text": "\n".join(code_lines)})
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^[-*_]{3,}\s*$", line):
            blocks.append({"type": "hr"})
            i += 1
            continue

        # Headings
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            blocks.append({"type": f"h{level}", "text": m.group(2).strip()})
            i += 1
            continue

        # Blockquote
        if line.startswith(">"):
            bq_lines = []
            while i < len(lines) and lines[i].startswith(">"):
                bq_lines.append(lines[i].lstrip("> ").strip())
                i += 1
            blocks.append({"type": "blockquote", "text": " ".join(bq_lines)})
            continue

        # Unordered list
        if re.match(r"^[\*\-\+]\s+", line):
            items = []
            while i < len(lines) and re.match(r"^[\*\-\+]\s+", lines[i]):
                items.append(re.sub(r"^[\*\-\+]\s+", "", lines[i]))
                i += 1
            blocks.append({"type": "ul", "items": items})
            continue

        # Ordered list
        if re.match(r"^\d+\.\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i]))
                i += 1
            blocks.append({"type": "ol", "items": items})
            continue

        # Blank line → skip
        if not line.strip():
            i += 1
            continue

        # Paragraph (accumulate until blank or special)
        para_lines = []
        while i < len(lines):
            l = lines[i]
            if (not l.strip()
                    or l.startswith("#")
                    or re.match(r"^[-*_]{3,}\s*$", l)
                    or l.startswith(">")
                    or re.match(r"^[\*\-\+]\s+", l)
                    or re.match(r"^\d+\.\s+", l)
                    or l.startswith("```")):
                break
            para_lines.append(l)
            i += 1
        blocks.append({"type": "p", "text": " ".join(para_lines)})

    return blocks


# ── Render blocks → HTML snippets ────────────────────────────────────────────
def render_block(block: dict) -> str:
    t = block["type"]
    c = PALETTE

    if t == "h1":
        return (
            f'<h1 style="font-family:Georgia,\'Times New Roman\',serif;'
            f'font-size:28px;font-weight:700;color:{c["primary"]};'
            f'margin:0 0 4px 0;line-height:1.25;">{inline_md(block["text"])}</h1>'
        )

    if t == "h2":
        return (
            f'<table width="100%" cellpadding="0" cellspacing="0" style="margin:32px 0 12px 0;">'
            f'<tr><td style="font-family:Georgia,\'Times New Roman\',serif;'
            f'font-size:18px;font-weight:700;color:{c["primary"]};'
            f'padding-bottom:8px;border-bottom:2px solid {c["h2_bar"]};">'
            f'{inline_md(block["text"])}</td></tr></table>'
        )

    if t == "h3":
        return (
            f'<p style="font-family:Georgia,\'Times New Roman\',serif;'
            f'font-size:15px;font-weight:700;color:{c["primary"]};'
            f'margin:20px 0 6px 0;letter-spacing:0.02em;">'
            f'{inline_md(block["text"])}</p>'
        )

    if t == "h4":
        return (
            f'<p style="font-family:\'Helvetica Neue\',Arial,sans-serif;'
            f'font-size:13px;font-weight:700;color:{c["muted"]};'
            f'text-transform:uppercase;letter-spacing:0.08em;margin:16px 0 4px 0;">'
            f'{inline_md(block["text"])}</p>'
        )

    if t == "hr":
        return f'<hr style="border:none;border-top:1px solid {c["border"]};margin:28px 0;" />'

    if t == "blockquote":
        return (
            f'<table cellpadding="0" cellspacing="0" width="100%" style="margin:16px 0;">'
            f'<tr>'
            f'<td width="4" style="background:{c["accent"]};border-radius:2px;">&nbsp;</td>'
            f'<td style="padding:10px 16px;font-family:\'Helvetica Neue\',Arial,sans-serif;'
            f'font-size:14px;color:{c["muted"]};font-style:italic;">'
            f'{inline_md(block["text"])}</td>'
            f'</tr></table>'
        )

    if t == "ul":
        rows = ""
        for item in block["items"]:
            rows += (
                f'<tr><td valign="top" style="padding:3px 8px 3px 0;color:{c["bullet"]};'
                f'font-size:16px;line-height:1.5;">&#8226;</td>'
                f'<td style="font-family:\'Helvetica Neue\',Arial,sans-serif;'
                f'font-size:14px;color:{c["primary"]};line-height:1.6;padding:3px 0;">'
                f'{inline_md(item)}</td></tr>'
            )
        return f'<table cellpadding="0" cellspacing="0" style="margin:8px 0 12px 8px;">{rows}</table>'

    if t == "ol":
        rows = ""
        for n, item in enumerate(block["items"], 1):
            rows += (
                f'<tr><td valign="top" style="padding:3px 10px 3px 0;color:{c["accent"]};'
                f'font-family:Georgia,serif;font-weight:700;font-size:14px;'
                f'line-height:1.5;white-space:nowrap;">{n}.</td>'
                f'<td style="font-family:\'Helvetica Neue\',Arial,sans-serif;'
                f'font-size:14px;color:{c["primary"]};line-height:1.6;padding:3px 0;">'
                f'{inline_md(item)}</td></tr>'
            )
        return f'<table cellpadding="0" cellspacing="0" style="margin:8px 0 12px 8px;">{rows}</table>'

    if t == "code_block":
        escaped = (block["text"]
                   .replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;"))
        return (
            f'<table width="100%" cellpadding="0" cellspacing="0" style="margin:16px 0;">'
            f'<tr><td style="background:{c["code_bg"]};border:1px solid {c["border"]};'
            f'border-radius:6px;padding:16px;">'
            f'<pre style="margin:0;font-family:\'SF Mono\',\'Fira Mono\',Consolas,monospace;'
            f'font-size:12px;color:{c["primary"]};line-height:1.6;white-space:pre-wrap;'
            f'word-break:break-all;">{escaped}</pre>'
            f'</td></tr></table>'
        )

    if t == "p":
        text = block["text"]
        # Key-value metadata lines like **Key:** Value → styled badge row
        kv = re.match(r"^\*\*(.+?):\*\*\s*(.*)", text)
        if kv:
            return (
                f'<p style="font-family:\'Helvetica Neue\',Arial,sans-serif;'
                f'font-size:13px;color:{c["muted"]};margin:4px 0;">'
                f'<span style="font-weight:700;color:{c["primary"]};">{kv.group(1)}:</span>'
                f'&nbsp;{inline_md(kv.group(2))}</p>'
            )
        return (
            f'<p style="font-family:\'Helvetica Neue\',Arial,sans-serif;'
            f'font-size:14px;color:{c["primary"]};line-height:1.7;margin:8px 0;">'
            f'{inline_md(text)}</p>'
        )

    return ""


# ── Full HTML wrapper ─────────────────────────────────────────────────────────
HEADER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="X-UA-Compatible" content="IE=edge" />
  <title>{title}</title>
</head>
<body style="margin:0;padding:0;background-color:{bg};font-size:16px;">
  <!-- Outer wrapper -->
  <table width="100%" cellpadding="0" cellspacing="0"
         style="background:{bg};padding:32px 16px;">
    <tr><td align="center">
      <!-- Card -->
      <table width="620" cellpadding="0" cellspacing="0"
             style="max-width:620px;background:{card};border-radius:10px;
                    box-shadow:0 2px 12px rgba(0,0,0,.08);overflow:hidden;">

        <!-- Top accent bar -->
        <tr><td height="4" style="background:{accent};font-size:0;">&nbsp;</td></tr>

        <!-- Header -->
        <tr><td style="padding:32px 40px 20px 40px;
                        border-bottom:1px solid {border};">
          <table width="100%" cellpadding="0" cellspacing="0"><tr>
            <td>
              {h1_block}
              <p style="margin:6px 0 0 0;font-family:'Helvetica Neue',Arial,sans-serif;
                         font-size:12px;color:{muted};letter-spacing:0.04em;
                         text-transform:uppercase;">{subtitle}</p>
            </td>
            <td align="right" valign="top">
              <span style="display:inline-block;background:{tag_bg};color:{tag_fg};
                           font-family:'Helvetica Neue',Arial,sans-serif;
                           font-size:11px;font-weight:700;letter-spacing:0.06em;
                           text-transform:uppercase;padding:4px 10px;
                           border-radius:20px;">{badge}</span>
            </td>
          </tr></table>
        </td></tr>

        <!-- Body -->
        <tr><td style="padding:28px 40px 8px 40px;">
          {body}
        </td></tr>

        <!-- Footer -->
        <tr><td style="padding:20px 40px 28px 40px;border-top:1px solid {border};
                        font-family:'Helvetica Neue',Arial,sans-serif;
                        font-size:11px;color:{muted};text-align:center;">
          Generated by <strong>md_to_email.py</strong> &nbsp;·&nbsp; {date}
        </td></tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


# ── Orchestrator ──────────────────────────────────────────────────────────────
def convert(md_path: str, html_path: str | None = None) -> str:
    md_path = Path(md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"Input file not found: {md_path}")

    md = md_path.read_text(encoding="utf-8")
    blocks = parse_blocks(md)

    # Extract title from first H1 (or filename)
    title = md_path.stem
    h1_block_html = ""
    subtitle = ""
    body_blocks = []

    first_h1_used = False
    for b in blocks:
        if b["type"] == "h1" and not first_h1_used:
            title = b["text"]
            h1_block_html = render_block(b)
            first_h1_used = True
        elif b["type"] == "p" and not body_blocks and first_h1_used and not subtitle:
            # Use first paragraph after H1 as subtitle if it looks like metadata
            if re.match(r"^\*\*Generated", b["text"]) or len(b["text"]) < 120:
                subtitle = inline_md(b["text"])
        else:
            body_blocks.append(render_block(b))

    body_html = "\n".join(body_blocks)
    badge = "AI Digest"
    date_str = datetime.utcnow().strftime("%B %d, %Y")

    html = HEADER_HTML.format(
        title=title,
        h1_block=h1_block_html,
        subtitle=subtitle,
        body=body_html,
        badge=badge,
        date=date_str,
        **PALETTE,
    )

    if html_path is None:
        html_path = md_path.with_suffix(".html")
    else:
        html_path = Path(html_path)

    html_path.write_text(html, encoding="utf-8")
    print(f"✓  Written → {html_path}")
    return str(html_path)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python md_to_email.py <input.md> [output.html]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) >= 3 else None
    convert(sys.argv[1], out)
