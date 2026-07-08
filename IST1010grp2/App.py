#!/usr/bin/env python3;

import platform as os;
import subprocess as subp;

cmd = ["python", "utils/posture_monitor.py", "os"]
cmd[2] = str(os.system())
subp.run(cmd)
