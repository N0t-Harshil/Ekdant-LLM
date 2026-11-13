"""Toy run of the L2 equations in 2 dimensions (instead of 576), for illustration.
Same steps as sandbox.py: diverse start -> 4 debate rounds -> 8 anchored descent
steps -> average -> gate. Weights are hand-picked so the mechanics are visible.
Writes frames.json (for the interactive page) and l2_filmstrip.png."""
import json, math, sys
import numpy as np

out_dir = sys.argv[1]
fig_path = sys.argv[2]

def rot(deg, gain=1.0):
    t = math.radians(deg)
    return gain * np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])

def rmsnorm(v):
    return v / math.sqrt((v ** 2).mean() + 1e-9)

h = np.array([1.2, 0.7])                      # one word's state (2 of the 576 numbers)
gains = np.logspace(math.log10(0.8), math.log10(1.25), 4)
W = [rot(a, g) for a, g in zip((25, 115, 205, 295), gains)]   # four different angles
b = [np.array([0.05, -0.05]), np.array([-0.04, 0.03]), np.array([0.02, 0.06]), np.array([-0.03, -0.02])]
T = [W[k] @ h + b[k] for k in range(4)]

frames = []
def snap(key, title, text, extra=None, trails=None):
    f = {"key": key, "title": title, "text": text, "h": h.round(4).tolist(),
         "T": [t.round(4).tolist() for t in T]}
    if extra:
        f.update({k: (v.round(4).tolist() if isinstance(v, np.ndarray) else v) for k, v in extra.items()})
    if trails:
        f["trails"] = [[p.round(4).tolist() for p in tr] for tr in trails]
    frames.append(f)

frames.append({"key": "h", "title": "The input: one word's state h",
               "text": "SmolLM2 has read the question and abstract. For this word its understanding is the state h (here 2 numbers instead of 576). Without L2, the answer would come straight from h.",
               "h": h.round(4).tolist(), "T": []})
snap("start", "Step 1 · four viewpoints",
     "Four learned transformations turn h into four different views T1–T4. They start far apart on purpose — four identical reviewers would have nothing to discuss.")

gamma = 0.12
trails = [[t.copy()] for t in T]
for r in range(1, 5):
    x = [rmsnorm(t) for t in T]
    scale = np.mean([np.linalg.norm(t) for t in T])
    new = []
    for k in range(4):
        others = [j for j in range(4) if j != k]
        sc = np.array([x[k] @ x[j] / math.sqrt(2) for j in others])
        w = np.exp(sc - sc.max()); w /= w.sum()
        cross = sum(wi * x[j] for wi, j in zip(w, others))
        proposal = cross / (np.linalg.norm(cross) + 1e-9) * scale     # MLP(cross), toy: keep direction, model scale
        new.append((1 - gamma) * T[k] + gamma * proposal)
    T = new
    for k in range(4):
        trails[k].append(T[k].copy())
    snap(f"round{r}", f"Step 2 · discussion, round {r} of 4",
         ("Each reviewer reads the other three (never itself), forms a proposal, and the repair gate γ ≈ 0.12 lets it take only a little of it. "
          "So the views move towards each other slowly and a reviewer can keep its own position.") if r == 1 else
         f"Round {r}: the same again, with this round's own weights. The views keep drifting closer, but stay different.",
         trails=[list(tr) for tr in trails])

a = rot(-12, 0.95) @ h                         # anchor computed from h
s_pull = [0.9, 0.45, 0.7, 0.3]                 # each reviewer's own pull strength (the descent gate)
eta = 0.0625
trails = [[t.copy()] for t in T]
for step in range(1, 9):
    T = [T[k] - eta * s_pull[k] * 2.0 * (T[k] - a) for k in range(4)]   # x2 so the toy movement is visible
    for k in range(4):
        trails[k].append(T[k].copy())
    if step in (1, 4, 8):
        snap(f"descent{step}", f"Step 3 · staying grounded, step {step} of 8",
             ("An anchor a is computed from the original state h — “what the text said”. Every reviewer takes a small step towards it, each with its own pull strength."
              if step == 1 else
              "After 8 steps all four are closer to the anchor, by different amounts — they do not collapse into one point." if step == 8 else
              "Halfway: the reviewers with a strong pull (R1, R3) move faster than the others."),
             extra={"a": a}, trails=[list(tr) for tr in trails])

c = sum(T) / 4
snap("average", "Step 4 · the verdict",
     "The four views are averaged into one verdict c.", extra={"a": a, "c": c})
y = c / np.linalg.norm(c) * np.linalg.norm(h)  # output norm, toy: verdict at the size of h
g_trained = 0.027
hp = h + g_trained * (y - h)
hp_big = h + 0.5 * (y - h)
snap("gate", "Step 4 · the gate decides how much is used",
     "Final state = h + g·(verdict − h). In the trained model g ≈ 0.027, so the final state (●) moves only about 3 % of the way from h towards the verdict. The dotted point shows g = 0.5 for comparison.",
     extra={"y": y, "hp": hp, "hp_big": hp_big})

json.dump(frames, open(f"{out_dir}/frames.json", "w"), indent=1)

# ------------------------------------------------------------------ filmstrip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
COL = ["#2A78D6", "#7B4FB8", "#1F8A80", "#C2185B"]
INK, MUT = "#1B2430", "#5A6472"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
pick = ["h", "start", "round1", "round4", "descent8", "average", "gate"]
fs = {f["key"]: f for f in frames}
titles = ["0  Input state h", "1  Four viewpoints", "2  Discussion, round 1", "2  After 4 rounds",
          "3  After 8 grounding steps", "4  Verdict = average", "4  Gate: g ≈ 0.027"]
for layout in ("row", "grid"):
  if layout == "row":
    fig, axs = plt.subplots(1, 7, figsize=(17.5, 3.0)); axs = list(axs)
  else:
    fig, axg = plt.subplots(2, 4, figsize=(12, 6.6)); axs = list(axg.flat)
    axs[-1].axis("off")
    axs[-1].text(0.02, 0.85, "Legend", fontsize=10, weight="bold", color=INK, transform=axs[-1].transAxes)
    for k in range(4):
        axs[-1].scatter([0.08], [0.7 - k * 0.11], s=55, color=COL[k], transform=axs[-1].transAxes)
        axs[-1].text(0.16, 0.7 - k * 0.11, f"reviewer R{k+1}", va="center", fontsize=9, color=INK, transform=axs[-1].transAxes)
    axs[-1].scatter([0.08], [0.26], s=60, marker="s", color=INK, transform=axs[-1].transAxes)
    axs[-1].text(0.16, 0.26, "h = the word's state", va="center", fontsize=9, color=INK, transform=axs[-1].transAxes)
    axs[-1].scatter([0.08], [0.15], s=110, marker="*", color=INK, transform=axs[-1].transAxes)
    axs[-1].text(0.16, 0.15, "anchor", va="center", fontsize=9, color=INK, transform=axs[-1].transAxes)
    axs[-1].scatter([0.08], [0.04], s=70, marker="X", color="#E9A23B", edgecolor=INK, linewidth=0.6, transform=axs[-1].transAxes)
    axs[-1].text(0.16, 0.04, "verdict", va="center", fontsize=9, color=INK, transform=axs[-1].transAxes)
  for ax, key, ttl in zip(axs, pick, titles):
      f = fs[key]
      ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.75, 1.75); ax.set_aspect("equal")
      ax.set_xticks([]); ax.set_yticks([])
      for sp in ax.spines.values():
          sp.set_color("#DCE2E8")
      ax.set_facecolor("#F7F9FB")
      ax.axhline(0, color="#E3E7EC", lw=0.8); ax.axvline(0, color="#E3E7EC", lw=0.8)
      if "trails" in f:
          for k, tr in enumerate(f["trails"]):
              tr = np.array(tr); ax.plot(tr[:, 0], tr[:, 1], color=COL[k], lw=1.2, alpha=0.55)
      hh = f["h"]
      ax.scatter([hh[0]], [hh[1]], s=60, marker="s", color=INK, zorder=5)
      ax.annotate("h", hh, xytext=(5, 5), textcoords="offset points", color=INK, fontsize=9, weight="bold")
      for k, t in enumerate(f["T"]):
          if key in ("gate",):
              break
          ax.scatter([t[0]], [t[1]], s=55, color=COL[k], zorder=6, edgecolor="white", linewidth=1)
          ax.annotate(f"R{k+1}", t, xytext=[(-12, 9), (-20, 5), (-20, -12), (6, -12)][k], textcoords="offset points", color=COL[k], fontsize=8, weight="bold")
      if "a" in f:
          ax.scatter([f["a"][0]], [f["a"][1]], s=110, marker="*", color=INK, zorder=7)
          ax.annotate("anchor", f["a"], xytext=(8, -3), textcoords="offset points", color=INK, fontsize=8)
      if "c" in f:
          ax.scatter([f["c"][0]], [f["c"][1]], s=70, marker="X", color="#E9A23B", zorder=8, edgecolor=INK, linewidth=0.6)
          ax.annotate("verdict", f["c"], xytext=(-20, -16), textcoords="offset points", color="#B07A12", fontsize=8, weight="bold")
      if "hp" in f:
          ax.plot([hh[0], f["y"][0]], [hh[1], f["y"][1]], color=MUT, lw=1, ls=":")
          ax.scatter([f["y"][0]], [f["y"][1]], s=50, marker="X", color="#E9A23B", zorder=6, edgecolor=INK, linewidth=0.6)
          ax.scatter([f["hp_big"][0]], [f["hp_big"][1]], s=40, facecolor="white", edgecolor=MUT, zorder=6, linestyle=":")
          ax.scatter([f["hp"][0]], [f["hp"][1]], s=45, color="#2A78D6", zorder=9)
          ax.annotate("final state\n(3 % of the way)", f["hp"], xytext=(-70, -34), textcoords="offset points", color="#2A78D6", fontsize=7.5, arrowprops=dict(arrowstyle="->", color="#2A78D6", lw=0.8))
          ax.annotate("verdict", f["y"], xytext=(6, -12), textcoords="offset points", color="#B07A12", fontsize=8, weight="bold")
      ax.set_title(ttl, fontsize=9.5, color=INK, loc="left")
  fig.text(0.005, 0.01, "Toy run: 2 numbers instead of 576, hand-picked weights, same steps and formulas as the code. Dots = the four reviewers R1–R4, lines = their paths.",
           fontsize=8.5, color=MUT)
  fig.tight_layout(rect=(0, 0.04, 1, 1))
  fig.savefig(fig_path if layout == "row" else fig_path.replace(".png", "_grid.png"), dpi=200, facecolor="white")
  plt.close(fig)
print("frames", len(frames))
