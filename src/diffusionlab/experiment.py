"""Train three seeds and replay a real reverse-diffusion trajectory offline."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .core import sample, shapes, sliced_wasserstein, train
from .visual import save, scatter_svg, table


def run(out, quick=False):
    torch.set_num_threads(1)
    started = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [17] if quick else [17, 29, 41]
    updates = 30 if quick else 4000
    n = 64 if quick else 768
    records, traces, animation = [], {}, {}
    for seed in seeds:
        model, trace = train(seed, updates)
        traces[str(seed)] = trace
        torch.save(
            {
                "state_dict": model.state_dict(),
                "steps": model.steps,
                "width": 96,
                "seed": seed,
                "updates": updates,
            },
            out / f"model-{seed}.pt",
        )
        target, ty = shapes(n * 8, 9000 + seed)
        for steps, guidance in [(32, 0.0), (32, 1.0), (32, 2.0), (128, 1.0)]:
            per_class = []
            for cls in range(3):
                generated, trajectory = sample(
                    model, torch.full((n,), cls), 5000 + seed, steps, guidance
                )
                reference = target[ty == cls][:n]
                per_class.append(sliced_wasserstein(generated.numpy(), reference.numpy()))
                if seed == seeds[0] and cls == 1 and steps == 32:
                    animation[str(guidance)] = [p[:64].tolist() for p in trajectory]
            records.append(
                {
                    "seed": seed,
                    "steps": steps,
                    "guidance": guidance,
                    "class_swd": per_class,
                    "mean_swd": float(np.mean(per_class)),
                }
            )
        if seed == seeds[0]:
            generated, _ = sample(model, torch.full((n,), 1), 5000 + seed, 32, 1.0)
            image = scatter_svg(
                [
                    ("Held-out spiral", target[ty == 1][:n].numpy()),
                    ("DDIM samples", generated.numpy()),
                ],
                "Noise becomes a spiral",
            )
            (out / "preview.svg").write_text(image)
    summary = []
    for steps, guidance in [(32, 0.0), (32, 1.0), (32, 2.0), (128, 1.0)]:
        scores = [
            r["mean_swd"] for r in records if r["steps"] == steps and r["guidance"] == guidance
        ]
        summary.append(
            {
                "steps": steps,
                "guidance": guidance,
                "mean SWD ↓": round(float(np.mean(scores)), 4),
                "seed SD": round(float(np.std(scores)), 4),
            }
        )
    payload = {
        "quick": quick,
        "seeds": seeds,
        "updates": updates,
        "samples_per_class": n,
        "seconds": time.perf_counter() - started,
        "records": records,
        "summary": summary,
        "loss_trace": traces,
        "animation": animation,
        "torch": torch.__version__,
        "reference_seed_rule": "9000 + training seed",
    }
    body = (
        '<div class="card"><label>Guidance <select id="guide">'
        "<option>0.0</option><option selected>1.0</option><option>2.0</option></select>"
        '</label><label>Denoising step <input id="step" type="range" min="0" max="32" '
        'value="0"></label><button id="play">Play / pause</button>'
        '<p id="frame"></p><canvas id="plot" width="900" height="460" '
        'aria-label="Reverse diffusion samples" role="img"></canvas></div>'
        "<h2>Measured conditional distribution distance</h2>"
        + table(summary, ["steps", "guidance", "mean SWD ↓", "seed SD"])
        + "<p>SWD is mean one-dimensional Wasserstein-1 distance over 128 fixed "
        "directions, averaged across three classes. Lower is better. Seed SD is "
        "population standard deviation across training seeds, not a confidence interval. "
        "Guidance zero is unconditional sampling; guidance one is ordinary conditioning.</p>"
    )
    script = """
    const canvas=document.getElementById('plot'),ctx=canvas.getContext('2d');
    const step=document.getElementById('step'),guide=document.getElementById('guide');
    function draw(){const t=Number(step.value),points=DATA.animation[guide.value][t];
      ctx.clearRect(0,0,900,460);ctx.fillStyle='#66e4cc';
      for(const [x,y] of points){ctx.beginPath();ctx.arc(450+x*110,230-y*95,3,0,Math.PI*2);ctx.fill()}
      document.getElementById('frame').textContent=`Step ${t}/32 · guidance ${guide.value} · measured spiral samples`;
    } step.oninput=draw;guide.onchange=draw;let playing=false;
    document.getElementById('play').onclick=()=>{playing=!playing};
    setInterval(()=>{if(playing){step.value=(Number(step.value)+1)%33;draw()}},130);draw();
    """
    save(
        out,
        payload,
        "Diffusion Playground",
        "Watch a small conditional PyTorch model turn Gaussian noise into a synthetic spiral. "
        "Scrub the actual DDIM trajectory and compare guidance settings.",
        body,
        script,
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/reference")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args.out, args.quick)
