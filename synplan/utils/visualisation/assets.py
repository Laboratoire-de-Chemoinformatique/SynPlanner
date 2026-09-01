"""The chrome a report page carries: its stylesheet, its script, its legend.

Strings only. Nothing here reads a route, so restyling a page never touches the
code that draws one.
"""

from __future__ import annotations

import json

from synplan.utils.routedraw import ROUTE_CSS

#: Page chrome for :func:`routes_report_html`. The route drawings bring their own
#: rules through :data:`synplan.utils.routedraw.ROUTE_CSS`.
REPORT_CSS = """
:root{--ink:#0f1419;--ink2:#38414a;--ink3:#6b7480;--ink4:#9ba3ad;
--rule:#e6e8eb;--surface:#ffffff;--bg:#fafbfc;--accent:#1e3a8a;--ok:#1f4d3d}
*,::before,::after{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-size:14px;line-height:1.5;
font-family:"Inter Tight",system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:44px 28px 96px}
.eyebrow{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--ink3)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:3px}
header.page{padding:26px 26px 24px}
header.page h1{margin:0;font-size:21px;font-weight:600;letter-spacing:-.01em}
.target{display:flex;flex-wrap:wrap;gap:20px;align-items:center;margin-top:20px}
.tile{flex:0 1 auto;max-width:100%;border:1px solid var(--accent);border-radius:3px;
background:#fff;padding:9px 11px}
.tile svg{display:block;max-width:100%;height:auto}
.target>div+div{flex:1 1 240px;min-width:0}
.target .smi{font-size:13px;color:var(--ink2);word-break:break-all;margin-top:5px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);margin-top:24px;
border:1px solid var(--rule);border-radius:3px;overflow:hidden;background:var(--surface)}
.stat{padding:13px 16px 14px;border-left:1px solid var(--rule)}
.stat:first-child{border-left:0}
.stat .v{display:block;margin-top:5px;font-size:27px;font-weight:600;line-height:1.05;letter-spacing:-.02em}
.stat .u{font-size:12px;font-weight:400;color:var(--ink4);letter-spacing:0}
.legend{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
.chip{display:inline-flex;align-items:center;gap:7px;padding:4px 10px;background:var(--surface);
border:1px solid var(--rule);border-radius:3px;font-size:11px;font-weight:600;
text-transform:uppercase;letter-spacing:.1em;color:var(--ink2)}
.sw{width:12px;height:12px;border-radius:2px;flex:0 0 auto}
.route{margin-top:20px}
.rhead{display:flex;flex-wrap:wrap;gap:34px;padding:13px 18px;border-bottom:1px solid var(--rule)}
.kv .v{font-size:15px;font-weight:600;margin-top:2px}
.kv .v.id{color:var(--accent)}
.draw{padding:20px 18px 22px;overflow-x:auto;display:flex;justify-content:center;
background:radial-gradient(circle,#e9ecef .9px,transparent .9px) 0 0/30px 30px,#fff}
.draw > svg{display:block;max-width:100%;height:auto;flex:0 0 auto}
.step{display:grid;grid-template-columns:22px minmax(0,1fr);gap:0 13px;
align-items:start;padding:11px 18px;border-top:1px solid var(--rule)}
.disc{width:22px;height:22px;border-radius:50%;background:#2b3440;color:#fff;
font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;
line-height:1;margin-top:1px}
.lab{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--ok)}
.rxn{font-size:12px;color:var(--ink2);word-break:break-all;line-height:1.6}
.acts{margin-left:auto;align-self:center;display:flex;gap:6px}
.act{font:inherit;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;
line-height:1;color:var(--ink2);background:var(--surface);border:1px solid var(--rule);
border-radius:3px;padding:6px 9px;cursor:pointer}
.act:hover{color:var(--accent);border-color:var(--accent)}
.draw{cursor:zoom-in}
.zoom{display:none;position:fixed;inset:0;z-index:9;overflow:hidden;cursor:grab;
background:var(--surface);touch-action:none}
.zoom.on{display:block}
.zoom.drag{cursor:grabbing}
.zstage{position:absolute;top:0;left:0;transform-origin:0 0}
.zstage>svg{display:block}
.zbar{position:fixed;top:16px;right:18px;display:flex;gap:6px}
@media (max-width:760px){
.wrap{padding:26px 14px 64px}
header.page{padding:20px 16px 18px}
.rhead{gap:18px;padding:12px 16px}
.draw{padding:16px 12px 18px}
.stats{grid-template-columns:repeat(2,1fr)}
.stat:nth-child(odd){border-left:0}
.stat:nth-child(n+3){border-top:1px solid var(--rule)}
}
@media (max-width:430px){
.stats{grid-template-columns:1fr}
.stat{border-left:0;border-top:1px solid var(--rule)}
.stat:first-child{border-top:0}
}
"""

#: Export and zoom for :func:`routes_report_html`. The page pools every molecule in
#: one hidden ``<defs>``, so an exported drawing has to carry back the definitions
#: its own references reach for, and :data:`ROUTE_CSS` with them.
REPORT_JS = (
    "const SVG_CSS="
    + json.dumps(ROUTE_CSS)
    + ";"
    + r"""
const NS = "http://www.w3.org/2000/svg";
let zoom = null, stage = null, k = 1, tx = 0, ty = 0, drag = null;

function standalone(svg) {
  const out = svg.cloneNode(true), ids = new Set();
  for (const el of out.querySelectorAll("*")) {
    for (const a of el.attributes) {
      if (a.localName === "href") ids.add(a.value.slice(1));
      const m = /^url\(#(.+)\)$/.exec(a.value);
      if (m) ids.add(m[1]);
    }
  }
  const defs = document.createElementNS(NS, "defs");
  for (const id of ids) {
    const def = document.getElementById(id);
    if (def) defs.appendChild(def.cloneNode(true));
  }
  const style = document.createElementNS(NS, "style");
  style.textContent = SVG_CSS;
  out.insertBefore(defs, out.firstChild);
  out.insertBefore(style, out.firstChild);
  return new XMLSerializer().serializeToString(out);
}

function save(name, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

function png(svg, name) {
  const image = new Image();
  image.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = svg.width.baseVal.value * 2;
    canvas.height = svg.height.baseVal.value * 2;
    const pen = canvas.getContext("2d");
    pen.fillStyle = "#fff";
    pen.fillRect(0, 0, canvas.width, canvas.height);
    pen.drawImage(image, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => save(name, blob), "image/png");
  };
  // A data: URI leaves the canvas untainted; a blob: one from file:// does not.
  const text = standalone(svg);
  image.src = "data:image/svg+xml;base64," +
    btoa(unescape(encodeURIComponent(text)));
}

function place() {
  stage.style.transform = "translate(" + tx + "px," + ty + "px) scale(" + k + ")";
}

function zoomAt(factor, cx, cy) {
  const next = Math.min(12, Math.max(0.1, k * factor));
  tx = cx - (cx - tx) * (next / k);
  ty = cy - (cy - ty) * (next / k);
  k = next;
  place();
}

function build() {
  zoom = document.createElement("div");
  zoom.className = "zoom";
  zoom.innerHTML = '<div class="zstage"></div><div class="zbar">' +
    '<button class="act" data-z="out">&minus;</button>' +
    '<button class="act" data-z="in">+</button>' +
    '<button class="act" data-z="close">Close</button></div>';
  document.body.appendChild(zoom);
  stage = zoom.firstChild;
  zoom.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoomAt(Math.exp(-e.deltaY * 0.002), e.clientX, e.clientY);
  }, { passive: false });
  zoom.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".zbar")) return;
    zoom.setPointerCapture(e.pointerId);
    drag = [e.clientX - tx, e.clientY - ty];
    zoom.classList.add("drag");
  });
  zoom.addEventListener("pointermove", (e) => {
    if (!drag) return;
    tx = e.clientX - drag[0];
    ty = e.clientY - drag[1];
    place();
  });
  zoom.addEventListener("pointerup", () => {
    drag = null;
    zoom.classList.remove("drag");
  });
  zoom.addEventListener("click", (e) => {
    const button = e.target.closest("[data-z]");
    if (!button) return;
    if (button.dataset.z === "close") zoom.classList.remove("on");
    else zoomAt(button.dataset.z === "in" ? 1.3 : 1 / 1.3,
                innerWidth / 2, innerHeight / 2);
  });
}

function showZoom(svg) {
  if (!zoom) build();
  stage.replaceChildren(svg.cloneNode(true));
  // Natural size: the page fits a drawing to its card, so 1:1 is already a zoom in.
  k = 1;
  tx = (innerWidth - svg.width.baseVal.value) / 2;
  ty = (innerHeight - svg.height.baseVal.value) / 2;
  place();
  zoom.classList.add("on");
}

addEventListener("keydown", (e) => {
  if (e.key === "Escape" && zoom) zoom.classList.remove("on");
});

document.addEventListener("click", (e) => {
  const button = e.target.closest("[data-act]");
  const section = e.target.closest(".route");
  if (!section) return;
  const act = button ? button.dataset.act
            : e.target.closest(".draw") ? "zoom" : null;
  if (!act) return;
  const svg = section.querySelector(".draw > svg");
  const name = "route-" + section.querySelector(".v.id").textContent.trim();
  if (act === "zoom") showZoom(svg);
  else if (act === "svg")
    save(name + ".svg",
         new Blob([standalone(svg)], { type: "image/svg+xml" }));
  else png(svg, name + ".png");
});
"""
)

#: Every role the drawing tints, in reading order.
ROLE_LEGEND = (
    ("target", "Target molecule"),
    ("int", "Intermediate"),
    ("oos", "Not in stock"),
    ("bb", "In stock"),
)


#: Head and tail of a clustering report page, ``{title}`` left to fill in.
#: These pages fetch Bootstrap, so unlike the routes report they need a network.
BOOTSTRAP_PAGE_HEAD = """
    <!doctype html>
    <html lang="en">
    <head>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
    rel="stylesheet"
    integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3"
    crossorigin="anonymous">
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>
        /* Optional: Add some basic styling */
        .table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        tr:nth-child(even) {{ background-color: #ffffff; }}
        caption {{ caption-side: top; font-size: 1.5em; margin: 1em 0; }}
        svg {{ max-width: 100%; height: auto; }}
    </style>
    </head>
    <body>
    <div class="container"> """

BOOTSTRAP_PAGE_TAIL = """
    </div> <script
    src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"
    integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p"
    crossorigin="anonymous">
    </script>
    </body>
    </html>
    """

#: The tinted disc a clustering report's legend puts before a role. The caller
#: replaces ``rgb()`` with the role's colour.
BOX_MARK = """
    <svg width="30" height="30" viewBox="0 0 1 1" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle; margin-right: 5px;">
    <circle cx="0.5" cy="0.5" r="0.5" fill="rgb()" fill-opacity="0.35" />
    </svg>
    """
