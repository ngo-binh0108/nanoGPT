"""
Plot the train/val loss curves that train.py writes to out_dir/log.csv
$ python plot_loss.py --out_dir=out-shakespeare-char
saves out_dir/loss.png. Needs matplotlib (pip install matplotlib)
"""
import os
import csv

import matplotlib
matplotlib.use('Agg') # no display needed, we only save the figure
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
out_dir = 'out'
log_file = 'log.csv'
image_file = 'loss.png'
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

log_path = os.path.join(out_dir, log_file)
with open(log_path, newline='') as f:
    rows = list(csv.DictReader(f))
assert rows, f"no evaluations logged in {log_path} yet"
iters = [int(r['iter']) for r in rows]
train_loss = [float(r['train_loss']) for r in rows]
val_loss = [float(r['val_loss']) for r in rows]

best = min(range(len(rows)), key=lambda i: val_loss[i])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(iters, train_loss, label='train')
ax.plot(iters, val_loss, label='val')
ax.scatter([iters[best]], [val_loss[best]], color='C1', zorder=3)
ax.annotate(f"best val {val_loss[best]:.4f} @ {iters[best]}", (iters[best], val_loss[best]),
            textcoords='offset points', xytext=(0, 10), ha='center')
ax.set_xlabel('iteration')
ax.set_ylabel('loss')
ax.set_title(out_dir)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
image_path = os.path.join(out_dir, image_file)
fig.savefig(image_path, dpi=120)
print(f"best val loss {val_loss[best]:.4f} at iter {iters[best]}, saved {image_path}")
