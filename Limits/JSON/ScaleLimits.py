import json

def CalculateSigma(N,L,eff):
    return (N)/(L*eff)

def ScaleDict(m,s_obs,s_scale):
    for n in m:
        m[n] = m[n] * s_obs / s_scale

# Set input/output filenames and scaling factor
input_file = "limits.json"
output_file = "scaled_limits.json"
sigma = {'300.0': CalculateSigma(5.5,1.1,0.27),'3000.0':CalculateSigma(1,1.1,0.4)}
scale = 1.0 #fb^-1 (Set the relative theory scale for the plot)

# Step 1: Load the JSON file
with open(input_file, "r") as f:
    data = json.load(f)

for mass in data:
    ScaleDict(data[mass],sigma[mass],scale)
print(data) #Inspect to ensure this looks right for your scale


# Step 3: Write to a new JSON file
with open(output_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"Scaled data saved to {output_file}")
