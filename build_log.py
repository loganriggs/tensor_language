"""Render LOG.md to log_live.html with real markdown formatting (figures inlined as data URIs).

Usage: python build_log.py
"""

import base64
import re
from pathlib import Path

import markdown

text = Path("LOG.md").read_text()

body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])

# inline any local figure references as data URIs so the page is self-contained
def inline(m):
    p = Path(m.group(1))
    if p.exists():
        return 'src="data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode() + '"'
    return m.group(0)

body = re.sub(r'src="((?:figures/)?[\w.-]+\.png)"', inline, body)

HTML = f"""<title>Experiment log — live</title>
<style>
  .log {{
    --surface: #fcfcfb; --ink: #0b0b0b; --secondary: #52514e; --muted: #898781;
    --hairline: rgba(11,11,11,0.10); --accent: #2a78d6;
    background: var(--surface); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height: 1.55; padding: 36px 20px 64px; box-sizing: border-box;
  }}
  @media (prefers-color-scheme: dark) {{
    .log {{ --surface: #191918; --ink: #ebeae7; --secondary: #b3b1ac; --muted: #8a8883;
            --hairline: rgba(235,234,231,0.14); --accent: #6aa5e8; }}
    .log code, .log pre, .log th {{ background: #242422; }}
  }}
  :root[data-theme="dark"] .log {{ --surface: #191918; --ink: #ebeae7; --secondary: #b3b1ac;
    --muted: #8a8883; --hairline: rgba(235,234,231,0.14); --accent: #6aa5e8; }}
  :root[data-theme="dark"] .log code, :root[data-theme="dark"] .log pre,
  :root[data-theme="dark"] .log th {{ background: #242422; }}
  :root[data-theme="light"] .log {{ --surface: #fcfcfb; --ink: #0b0b0b; --secondary: #52514e;
    --muted: #898781; --hairline: rgba(11,11,11,0.10); --accent: #2a78d6; }}
  :root[data-theme="light"] .log code, :root[data-theme="light"] .log pre,
  :root[data-theme="light"] .log th {{ background: #f3f2ee; }}
  .log .col {{ max-width: 880px; margin: 0 auto; }}
  .log h1 {{ font-size: 24px; font-weight: 650; margin: 0 0 16px; text-wrap: balance; }}
  .log h2 {{ font-size: 18px; font-weight: 650; margin: 36px 0 10px; padding-top: 16px;
             border-top: 1px solid var(--hairline); }}
  .log h3 {{ font-size: 15px; font-weight: 650; margin: 24px 0 6px; }}
  .log p, .log li {{ font-size: 14px; max-width: 78ch; }}
  .log p {{ margin: 8px 0; }}
  .log ul, .log ol {{ padding-left: 22px; }}
  .log li {{ margin: 4px 0; }}
  .log code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px;
               background: #f3f2ee; border-radius: 3px; padding: 0 4px; }}
  .log pre {{ background: #f3f2ee; border: 1px solid var(--hairline); border-radius: 6px;
              padding: 10px 14px; overflow-x: auto; }}
  .log pre code {{ background: none; padding: 0; }}
  .log img {{ max-width: 100%; border: 1px solid var(--hairline); border-radius: 6px; margin: 10px 0; }}
  .log table {{ border-collapse: collapse; font-size: 13px; margin: 12px 0; display: block;
                overflow-x: auto; max-width: 100%; }}
  .log th, .log td {{ border: 1px solid var(--hairline); padding: 5px 12px; text-align: right;
                      font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .log th:first-child, .log td:first-child {{ text-align: left; }}
  .log th {{ background: #f3f2ee; font-weight: 600; }}
  .log blockquote {{ border-left: 3px solid var(--accent); margin: 10px 0; padding: 2px 14px;
                     color: var(--secondary); }}
  .log hr {{ border: none; border-top: 1px solid var(--hairline); margin: 24px 0; }}
  .log a {{ color: var(--accent); }}
</style>
<div class="log"><div class="col">
{body}
</div></div>
"""

Path("log_live.html").write_text(HTML)
print(f"wrote log_live.html ({len(HTML) / 1024:.0f} kB)")
