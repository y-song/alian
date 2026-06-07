import matplotlib.pyplot as plt
import pandas as pd

eecab = {
    'setting': ['2.76 TeV, recoil on', '5.02 TeV, recoil on'],
    '0.1': [0.050, 0.044],
    '0.2': [0.044, 0.040],
    '0.3': [0.035, 0.032]
}
cab = {
    'setting': ['2.76 TeV, recoil on', '5.02 TeV, recoil on'],
    '0.1': [0.080, 0.070],
    '0.2': [0.076, 0.068],
    '0.3': [0.071, 0.066]
}

df_eecab = pd.DataFrame(eecab)
df_cab = pd.DataFrame(cab)

print("EEC(AxB) dataframe:\n", df_eecab)
print("Cab dataframe:\n", df_cab)

# 1. Define X-axis points
zcut = [0.1, 0.2, 0.3]

# 2. Extract Y-values from df_cab rows
# row 0 corresponds to '2.76 TeV, recoil on'
# row 1 corresponds to '5.02 TeV, recoil on'
eecab_276 = df_eecab.loc[0, ['0.1', '0.2', '0.3']].values
eecab_502 = df_eecab.loc[1, ['0.1', '0.2', '0.3']].values
cab_276 = df_cab.loc[0, ['0.1', '0.2', '0.3']].values
cab_502 = df_cab.loc[1, ['0.1', '0.2', '0.3']].values

# 3. Create the plot
plt.figure(figsize=(6, 4))
plt.plot(zcut, cab_276, marker='o', label='Cab, 2.76 TeV')
plt.plot(zcut, cab_502, marker='s', label='Cab, 5.02 TeV')
plt.plot(zcut, eecab_276, marker='o', label='EEC(AxB), 2.76 TeV')
plt.plot(zcut, eecab_502, marker='s', label='EEC(AxB), 5.02 TeV')

# 4. Format the plot
plt.xticks([0.1, 0.2, 0.3])  # Forces X-axis to only show 1, 2, and 3
plt.xlabel('$z_{\mathrm{cut}}$')
plt.ylabel('$R_L$ crossing point')
plt.ylim(0, 0.1)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 5. Show the plot
plt.savefig('output/eecab_cab_summary.png', bbox_inches='tight')