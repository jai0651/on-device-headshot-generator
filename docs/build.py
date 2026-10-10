"""Wrap each docs/src/<page>.html fragment in the shared site frame -> docs/<page>.html.

    python3 docs/build.py

A fragment starts with a metadata comment:
    <!-- title: Models | eyebrow: DML · under the hood | lede: One sentence under the heading. -->
Everything after it is the page body. Navigation, header and footer live here, once.
"""

import re
from pathlib import Path


def split_arrow_paths(html):
    """An SVG <path d="M.. M.."> with marker-end only draws a head on its LAST subpath.
    Split every multi-subpath arrow into one <path> per subpath so each gets its head."""
    def split(m):
        d, rest = m.group(1), m.group(2)
        subs = re.findall(r"M[^M]+", d)
        if len(subs) < 2 or "marker-end" not in rest:
            return m.group(0)
        return "".join(f'<path d="{x.strip()}" {rest}>' for x in subs)
    return re.sub(r'<path d="([^"]+)" ([^>]*)>', split, html)

ROOT = Path(__file__).parent
PAGES = [  # (file stem, nav label)
    ("index", "Start here"),
    ("models", "Models"),
    ("training", "Training"),
    ("generation", "Generation"),
    ("evaluation", "Evaluation"),
    ("infra", "Where it runs"),
    ("results", "Results"),
]
NOTES = "../on-device-headshot-generator.html"

FRAME = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{title} · DML headshot generator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css">
</head>
<body>
<nav class="sitenav"><div class="wrap">
  <a class="brand" href="index.html">DML · headshots</a>
{nav}
  <a class="pg" href="{notes}">Study notes ↗</a>
</div></nav>
<header class="hero">
  <div class="wrap">
    <p class="eyebrow">{eyebrow}</p>
    <h1>{title}</h1>
    <p class="standfirst">{lede}</p>
  </div>
</header>
<div class="wrap"><div class="single"><article>
{body}
{prevnext}
</article></div></div>
<footer><div class="wrap">DML · on-device headshot generator · {title}</div></footer>
</body>
</html>
"""


def main():
    for i, (stem, label) in enumerate(PAGES):
        path = ROOT / "src" / f"{stem}.html"
        if not path.exists():
            print(f"  (skip {stem}: no fragment yet)")
            continue
        src = path.read_text()
        meta = re.match(r"\s*<!--(.*?)-->", src, re.S)
        fields = dict(part.split(":", 1) for part in meta.group(1).split(" | "))
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        body = split_arrow_paths(src[meta.end():])
        nav = "\n".join(f'  <a class="pg{" on" if s == stem else ""}" href="{s}.html">{lab}</a>' for s, lab in PAGES)
        links = []
        if i > 0:
            s, lab = PAGES[i - 1]
            links.append(f'<a href="{s}.html"><span class="t">← {lab}</span><span class="d">previous page</span></a>')
        if i + 1 < len(PAGES):
            s, lab = PAGES[i + 1]
            links.append(f'<a href="{s}.html"><span class="t">{lab} →</span><span class="d">next page</span></a>')
        prevnext = f'<hr><div class="pagelinks">{"".join(links)}</div>'
        html = FRAME.format(title=fields["title"], eyebrow=fields["eyebrow"], lede=fields["lede"],
                            nav=nav, notes=NOTES, body=body, prevnext=prevnext)
        (ROOT / f"{stem}.html").write_text(html)
        print(f"  docs/{stem}.html")


if __name__ == "__main__":
    main()
