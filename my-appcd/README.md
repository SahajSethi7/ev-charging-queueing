# EV queue simulator frontend

Requires Node.js 20.19+ or 22.12+.

The application reads `public/simulator-data.json` at runtime. Generate that
file after rerunning Weeks 8 and 9:

```powershell
python ..\Code\export_simulator_data.py
npm ci
npm run dev
```

Only exact simulated fleet sizes are exposed. The UI deliberately does not
interpolate nonlinear waiting probabilities between fleet sizes.
