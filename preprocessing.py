import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
# Load the dataset
df = pd.read_csv("/Users/mac/btp/dataset(in).csv")

# Clean column names
df.columns = df.columns.str.strip()

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Convert Pressure (hPa → Pa)
df['PS_Pa'] = df['PS'] * 100

# Convert Temperature (°C → Kelvin)
df['T_K'] = df['T2M'] + 273.15

# Compute Air Density (ρ = P / RT)
R = 287  # specific gas constant for dry air, J/(kg·K)
df['Air_Density'] = df['PS_Pa'] / (R * df['T_K'])

# Compute Wind Speed Cubed (v³)
df = df[df['WS50M'] <= 20]  # Remove unrealistic wind speeds
df['WS50M_Cubed'] = df['WS50M'] ** 3

# Compute Power Output using Suzlon S144 Rotor
# Rotor diameter = 144 meters → radius = 72 m
# Swept area A = π × r²
A = np.pi * (72 ** 2)  # ≈ 16286.45 m²
df['Power'] = 0.5 * df['Air_Density'] * A * df['WS50M_Cubed']  # in watts

# Suzlon S144 has a rated capacity of 3 MW
df['Power'] = df['Power'].clip(upper=3_000_000)

# Plot power output
'''plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Power'], color='darkgreen')
plt.xlabel('Date')
plt.ylabel('Simulated Power Output (W)')
plt.title('Simulated Wind Power Output Over Time (Suzlon S144)')
plt.grid(True)
plt.tight_layout()
plt.show()'''
sns.boxplot(x=df['WS50M'])

# Optional: Save the updated dataset
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x='WS50M', color='skyblue')
plt.title('Boxplot of Wind Speed at 50m')
plt.xlabel('Wind Speed (m/s)')
plt.tight_layout()
plt.show()
