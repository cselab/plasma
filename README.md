# torax

```
git clone git@github.com:google-deepmind/torax.git
```

```
python -m pip install git+https://github.com/google-deepmind/torax.git
```

```
python -m pip install seaborn
```

```
run_torax --config=basic_config.py --quit
run_torax --config=iterhybrid_rampup.py --quit
```

# hacking

```
git clone --depth 1 -b v1.1.1 git@github.com:google-deepmind/torax.git torax0
rm -rf torax0/.git
git add torax0 -A
```

```
python3 -m pip install -e ./torax0
python3 -m pip install coverage
```

```
python -m coverage  run --source . run.py
python -m coverage html
```

# Refs

Van Mulders, S., Felici, F., Sauter, O., Citrin, J., Ho, A., Marin,
M., & Van De Plassche, K. L. (2021). Rapid optimization of stationary
tokamak plasmas in RAPTOR: demonstration for the ITER hybrid scenario
with neural network surrogate transport model QLKNN. Nuclear Fusion,
61(8), 086019.

Felici, F. (2011). Real-time control of tokamak plasmas: from control
of physics to physics-based control.
