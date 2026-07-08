#!/usr/bin/env python3

import os
import platform
import subprocess

# run app with platform specific arguments
# cmd = ["bin/python3", "src/posture_monitor.py", "os"]
cmd = ["bin/python3", "src/test_monitor.py", "os"] # test and debug
opSys = platform.system()

# activate virtual environment for specific platforms
# try:
#     if opSys == "Linux": subprocess.run( ["source",f"{os.getcwd()}/bin/activate"] )
#     # elif opSys == "Windows": subprocess.run( ['''.\/bin\Activate.ps1'''] )
#     # elif opSys == "Darwin": subprocess.run( [".", "/bin/activate"] )
# except  subprocess.CalledProcessError as e: print(e.stderr)

# run
cmd[2] = str(opSys)
subprocess.run(cmd)

# # deactivate environment, not specific
# try: subprocess.run( ["deactivate"] )
# except  subprocess.CalledProcessError as e: print(e.stderr)
