# NoGIL-Swarm

Watch demo video

## INSTALLATION!!!!!!!!!!

This project requires Python 3.14+ (Free-Threaded build) to demonstrate the removal of the Global Interpreter Lock (GIL). 

Open your terminal and follow.

Assuming you have Python Install Manager, and **NOT** the old Python Launcher,

```bash
py install 3.14t
```
then do this just in case, if it's installed then use that Python and you're good to go
```bash
py --list
```

### Install Dependencies:
```bash
pip install zengl glfw numpy
```
### Run with GIL Disabled: 
To unlock the 6 physics worker threads, you **MUST** use the -X gil=0 flag:
```bash
python -X gil=0 main.py
```
Verify No-GIL Status: If running correctly, your terminal should not show any GIL warnings, and your CPU usage should be distributed across multiple cores (check Task Manager/HTOP).

Note: If you run this on a standard (non-t) Python build, the simulation will still work, but the physics workers will be bottlenecked by the GIL, resulting in lower performance and potential stuttering at high entity counts.

## Features?

* Python 3.14t (Experimental): Fully utilizes the --threading build to execute physics logic without the Global Interpreter Lock (GIL).

* High-Frequency Physics: 6 dedicated worker threads running independent Euler integration at ~200Hz.

* Zero-Copy GPU Pipeline: Leverages ZenGL for hardware-accelerated instancing of 2,000+ entities with a single draw call.

* Non-Linear Vector Fields: Real-time switching between tangential vortex (Whirlpool) and radial inverse-distance forces (Shockwave).

Perhaps Python will finally be used in rendering? God knows. All I know is that I wrestled with the pipeline of ZenGL, and I had to use ZenGL as my first renderer.