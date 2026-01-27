import numpy as np
import matplotlib.pyplot as plt

try:
    # Try reading the mock CSV first for visualization checking
    import pandas as pd
    df = pd.read_csv('z-alpha_logs.csv')
    alpha_s = df['alpha_s'].values
    alpha_f = df['alpha_f'].values
    print("Loaded data from z-alpha_logs.csv")
except Exception:
    try:
        data = np.load('alpha_logs.npy', allow_pickle=True)
        alpha_s = [d['alpha_s'] for d in data]
        alpha_f = [d['alpha_f'] for d in data]
        print("Loaded data from alpha_logs.npy")
    except FileNotFoundError:
        print("Data file not found")
        exit(1)

# config.yaml says sampling_steps: 50.
steps = 50
if 'df' in locals():
    total_len = len(df)
else:
    total_len = len(data)

# If we captured multiple batches, take the first one?
# But we clear logs at batch 0 and save at batch 0. So it should be one batch = one video generation trace. Or batch_size images generate in parallel.
# If batch_size > 1, the alphas are already averaged over batch in TimeAwareDualAdapter.
# So total_len = steps * layers.

if total_len == 0:
    print("No data in logs")
    exit(1)

layers = total_len // steps
if total_len % steps != 0:
    print(f"Warning: total length {total_len} not divisible by steps {steps}. Assuming partial or different steps.")
    # Try to deduce steps from data if possible? No.
    # Just Assume 50 steps.
    
print(f"Detected {layers} layers per step (Total {total_len} entries).")

# Reshape
# The sequence is: Step 1 (Layer 1...L), Step 2 (Layer 1...L)...
# Wait, are we sure it is Step first or Layer first?
# Model forward calls layers sequentially.
# So Step 1: L1, L2, L3...
# Then Step 2: L1, L2, L3...
# So yes, reshape(steps, layers) works if row-major (default).
try:
    alpha_s = np.array(alpha_s).reshape(steps, layers)
    alpha_f = np.array(alpha_f).reshape(steps, layers)
except ValueError:
    print("Reshape failed. Plotting raw average.")
    # Fallback to simple binning
    pass

# Average over layers
alpha_s_mean = alpha_s.mean(axis=1) if alpha_s.ndim > 1 else alpha_s
alpha_f_mean = alpha_f.mean(axis=1) if alpha_f.ndim > 1 else alpha_f

# Plot
t = np.linspace(1.0, 0.0, steps) # From Noise to Data

plt.figure(figsize=(10, 6))
plt.plot(t, alpha_f_mean, label=r'$\alpha_f$ (Spectral)', color='blue', linewidth=2)
plt.plot(t, alpha_s_mean, label=r'$\alpha_s$ (Spatial)', color='orange', linewidth=2)

plt.xlabel(r'Timestep $t$ (1.0 Noise $\to$ 0.0 Data)')
plt.ylabel(r'Average Gate Weight ($\alpha$)')
plt.title('Dynamic Weight Evolution over Diffusion Timesteps')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().invert_xaxis() # 1.0 -> 0.0

plt.savefig('alpha_evolution.png')
print("Plot saved to alpha_evolution.png")
