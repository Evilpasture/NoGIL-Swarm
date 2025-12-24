# NoGIL-Swarm

Built with [ZenGL](https://github.com/szabolcsdombi/zengl)



## INSTALLATION!!!!!!!!!!

This project requires Python 3.14+ (Free-Threaded build) to demonstrate the removal of the Global Interpreter Lock (GIL). 

Open your terminal and follow.

Assuming you have Python Install Manager working, and **NOT** the old Python Launcher,

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
There wouldn't be any dependency problems with these in 3.14.2t.
### Run with GIL Disabled: 
To unlock the 6 physics worker threads, you **MUST** use the -X gil=0 flag:
```bash
python -X gil=0 main.py
```
Verify No-GIL Status: If running correctly, your terminal should not show any GIL warnings, and your CPU usage should be distributed across multiple cores (check Task Manager/HTOP).

Note: If you run this on a standard (non-t) Python build, the simulation will still work, but the physics workers will be bottlenecked by the GIL, resulting in lower performance and potential stuttering at high entity counts.

## Features?

This project demonstrates two distinct approaches to high-performance particle simulations in Python. 
It tracks the journey from optimizing CPU threads in a 
Free-Threaded (No-GIL) environment to leveraging GPGPU (General-Purpose computing on Graphics Processing Units) to 
handle tens of millions of entities. See [main.py](main.py) and [main-GPGPU.py](main-GPGPU.py)

Perhaps Python will finally be used in rendering? 
God knows. All I know is that I wrestled with the pipeline of ZenGL, and I had to use ZenGL as my first renderer.
One day I just decided I wanted to learn rendering but using the 3.14.2t, discovered the only working renderer that is ZenGL,
suffering with the first static triangle. But guess what? It worked. And I'm proud of myself.