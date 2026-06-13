import os
import matplotlib.pyplot as plt
import pandas as pd

# X-axis points and the matching column labels
zcut = [0.1, 0.2, 0.3]
zcols = ['0.1', '0.2', '0.3']

settings = ['2.76 TeV, 100-120 GeV', '5.02 TeV, 100-120 GeV', '5.02 TeV, 150-200 GeV']

# Setting as the index -> rows are addressable by label, and
# the printed tables show the setting names down the left side.
df_eecab = pd.DataFrame(
    {'0.1': [0.050, 0.044, 0.040], '0.2': [0.044, 0.040, 0.035], '0.3': [0.035, 0.032, 0.028]},
    index=settings,
)
df_cab = pd.DataFrame(
    {'0.1': [0.080, 0.070, 0.064], '0.2': [0.076, 0.068, 0.059], '0.3': [0.071, 0.066, 0.057]},
    index=settings,
)

print("EEC(AxB) dataframe:\n", df_eecab)
print("Cab dataframe:\n", df_cab)

# Encode observable with color, energy with marker
observables = {'Cab': df_cab, 'EEC(AxB)': df_eecab}
colors = {'Cab': 'C0', 'EEC(AxB)': 'C1'}
markers = {'2.76 TeV, 100-120 GeV': 'o', '5.02 TeV, 100-120 GeV': 's', '5.02 TeV, 150-200 GeV': 's'}

fig, ax = plt.subplots(figsize=(6, 4))
for obs, df in observables.items():
    for setting, row in df.iterrows():           # label-based, position-independent
        energy = setting.split(',')[0]           # '2.76 TeV'
        ax.plot(zcut, row[zcols].to_numpy(), marker=markers[setting],
                label=setting)

ax.set_xticks(zcut)
ax.set_xlabel(r'$z_{\mathrm{cut}}$')
ax.set_ylabel(r'$R_L$ crossing point')
ax.set_ylim(0, 0.1)
# ax.legend(framealpha=1)
ax.grid(True, linestyle='--', alpha=0.6)

os.makedirs('output', exist_ok=True)
fig.savefig('output/eecab_cab_summary.png', dpi=300, bbox_inches='tight')
plt.close(fig)