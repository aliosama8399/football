import os
import sys
import subprocess
import markdown
from pygments.formatters import HtmlFormatter

def convert_md_to_pdf():
    workspace_dir = r"d:\SASUniversityEdition\Machine\MODEL\football"
    md_file = os.path.join(workspace_dir, "football_project_book.md")
    html_file = os.path.join(workspace_dir, "football_project_book.html")
    pdf_file = os.path.join(workspace_dir, "football_project_book.pdf")

    with open(md_file, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Pygments syntax highlighting CSS (monokai / dark theme)
    pygments_css = HtmlFormatter(style="monokai").get_style_defs(".codehilite")

    # Render Markdown to HTML with extensions
    html_body = markdown.markdown(
        md_text,
        extensions=[
            "fenced_code",
            "codehilite",
            "tables",
            "toc",
            "attr_list",
            "def_list"
        ]
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Football Analysis Platform — Technical Book</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500;600&display=swap');

@page {{
    size: A4;
    margin: 15mm 15mm 15mm 15mm;
}}

* {{
    box-sizing: border-box;
}}

body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
    line-height: 1.6;
    padding: 25px 35px;
    font-size: 13.5px;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
}}

h1, h2, h3, h4, h5, h6 {{
    color: #58a6ff;
    font-weight: 600;
    margin-top: 1.8em;
    margin-bottom: 0.6em;
    line-height: 1.25;
}}

h1 {{
    font-size: 26px;
    color: #79c0ff;
    border-bottom: 2px solid #1f6feb;
    padding-bottom: 12px;
    margin-top: 0;
    text-align: center;
}}

h2 {{
    font-size: 19px;
    color: #58a6ff;
    border-bottom: 1px solid #30363d;
    padding-bottom: 6px;
    margin-top: 32px;
    page-break-after: avoid;
}}

h3 {{
    font-size: 15px;
    color: #1f6feb;
    margin-top: 24px;
    page-break-after: avoid;
}}

h4 {{
    font-size: 14px;
    color: #d2a8ff;
    margin-top: 18px;
}}

p {{
    margin-top: 0;
    margin-bottom: 12px;
}}

code {{
    font-family: 'Fira Code', Consolas, 'Courier New', monospace;
    background-color: rgba(110, 118, 129, 0.25);
    color: #f0883e;
    padding: 0.2em 0.4em;
    border-radius: 4px;
    font-size: 85%;
}}

pre {{
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px;
    overflow-x: auto;
    line-height: 1.45;
    margin: 14px 0;
    page-break-inside: avoid;
}}

pre code {{
    background: transparent;
    color: #c9d1d9;
    padding: 0;
    font-size: 11.5px;
}}

blockquote {{
    border-left: 4px solid #1f6feb;
    background-color: #161b22;
    color: #8b949e;
    padding: 12px 18px;
    margin: 16px 0;
    border-radius: 0 6px 6px 0;
    page-break-inside: avoid;
}}

blockquote p:last-child {{
    margin-bottom: 0;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    background-color: #161b22;
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid #30363d;
    page-break-inside: avoid;
}}

th, td {{
    padding: 10px 14px;
    border: 1px solid #30363d;
    text-align: left;
}}

th {{
    background-color: #21262d;
    color: #f0f6fc;
    font-weight: 600;
    font-size: 13px;
}}

td {{
    font-size: 12.5px;
}}

tr:nth-child(even) {{
    background-color: #0d1117;
}}

hr {{
    height: 1px;
    background-color: #30363d;
    border: none;
    margin: 32px 0;
}}

ul, ol {{
    padding-left: 24px;
    margin-top: 0;
    margin-bottom: 14px;
}}

li {{
    margin-bottom: 6px;
}}

a {{
    color: #58a6ff;
    text-decoration: none;
}}

/* Pygments Syntax Highlighting */
{pygments_css}

.codehilite {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 16px 0;
    page-break-inside: avoid;
}}

.codehilite pre {{
    border: none;
    padding: 0;
    margin: 0;
    background: transparent;
}}

/* Additional Syntax Colors */
.codehilite .k {{ color: #ff7b72; font-weight: bold; }} /* Keyword */
.codehilite .nf {{ color: #d2a8ff; font-weight: bold; }} /* Function name */
.codehilite .nc {{ color: #f0883e; font-weight: bold; }} /* Class name */
.codehilite .s {{ color: #a5d6ff; }} /* String */
.codehilite .c1 {{ color: #8b949e; font-style: italic; }} /* Comment */
.codehilite .mi {{ color: #79c0ff; }} /* Integer */

@media print {{
    body {{
        background-color: #0d1117 !important;
        color: #c9d1d9 !important;
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
    }}
    .codehilite, pre, blockquote, table {{
        page-break-inside: avoid;
    }}
    h1, h2, h3 {{
        page-break-after: avoid;
    }}
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"[OK] Generated styled HTML: {html_file}")

    edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not os.path.exists(edge_exe):
        edge_exe = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

    cmd = [
        edge_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_file}",
        f"file:///{html_file.replace('\\', '/')}"
    ]

    print("Executing headless Edge PDF generator...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(pdf_file) and os.path.getsize(pdf_file) > 0:
        print(f"[SUCCESS] PDF successfully created: {pdf_file} ({os.path.getsize(pdf_file):,} bytes)")
    else:
        print(f"[ERROR] PDF compilation failed. Exit code: {result.returncode}")
        print("Stderr:", result.stderr)

if __name__ == "__main__":
    convert_md_to_pdf()
