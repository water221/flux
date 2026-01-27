import numpy as np
import pandas as pd

def generate_mock_data():
    steps = 50
    layers = 64
    
    data = []

    # Params
    layer_std = 0.15 
    step_std = 0.08 
    
    np.random.seed(42)
    
    # Layer specific biases to simulate that some layers are consistently different
    layer_bias_s = np.random.normal(0, layer_std, layers)
    layer_bias_f = np.random.normal(0, layer_std, layers)
    
    for step in range(steps):
        # Map step to t: step 0 -> t=1.0, step 49 -> t=0.0
        t = 1.0 - (step / (steps - 1))
        
        # --- Curve A (Spectral): 1.5 -> Drops ---
        # Starts high (approx 1.6), stays relatively high until t=0.5, then drops to ~0.8
        # Using a slight power curve + base offset
        mean_f = 0.85 + 0.75 * (t ** 0.8)
        
        # --- Curve B (Spatial): Low -> 1.8 ---
        # Starts low (approx 0.6), rises sharply as t -> 0 (1-t gets larger)
        mean_s = 0.65 + 1.15 * ((1 - t) ** 2.5)
        
        for layer in range(layers):
            index = step * layers + layer
            
            # Add noise components
            val_s = mean_s + layer_bias_s[layer] + np.random.normal(0, step_std)
            val_f = mean_f + layer_bias_f[layer] + np.random.normal(0, step_std)
            
            # Clamp to valid sigmoid*2 range [0, 2]
            val_s = np.clip(val_s, 0.01, 1.99)
            val_f = np.clip(val_f, 0.01, 1.99)
            
            data.append({
                "index": int(index),
                "step_estimated": int(step),
                "layer_estimated": int(layer),
                "alpha_s": float(f"{val_s:.8f}"),
                "alpha_f": float(f"{val_f:.8f}")
            })
            
    df = pd.DataFrame(data)
    output_path = "z-alpha_logs.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {len(df)} rows.")

if __name__ == "__main__":
    generate_mock_data()
