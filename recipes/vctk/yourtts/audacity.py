#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests the audacity pipe.

Keep pipe_test.py short!!
You can make more complicated longer tests to test other functionality
or to generate screenshots etc in other scripts.

Make sure Audacity is running first and that mod-script-pipe is enabled
before running this script.

Requires Python 2.7 or later. Python 3 is strongly recommended.

"""

import os
import sys


class AudacityScript:
    def __init__(self):
        if sys.platform == 'win32':
            print("pipe-test.py, running on windows")
            self.TONAME = '\\\\.\\pipe\\ToSrvPipe'
            self.FROMNAME = '\\\\.\\pipe\\FromSrvPipe'
            self.EOL = '\r\n\0'
        else:
            print("pipe-test.py, running on linux or mac")
            self.TONAME = '/tmp/audacity_script_pipe.to.' + str(os.getuid())
            self.FROMNAME = '/tmp/audacity_script_pipe.from.' + str(os.getuid())
            self.EOL = '\n'

        print("Write to  \"" + self.TONAME +"\"")
        if not os.path.exists(self.TONAME):
            print(" ..does not exist.  Ensure Audacity is running with mod-script-pipe.")
            sys.exit()

        print("Read from \"" + self.FROMNAME +"\"")
        if not os.path.exists(self.FROMNAME):
            print(" ..does not exist.  Ensure Audacity is running with mod-script-pipe.")
            sys.exit()

        print("-- Both pipes exist.  Good.")

        self.TOFILE = open(self.TONAME, 'w')
        print("-- File to write to has been opened")
        self.FROMFILE = open(self.FROMNAME, 'rt')
        print("-- File to read from has now been opened too\r\n")

    def send_command(self, command):
        """Send a single command."""
        print("Send: >>> \n"+command)
        self.TOFILE.write(command + self.EOL)
        self.TOFILE.flush()

    def get_response(self):
        """Return the command response."""
        result = ''
        line = ''
        while True:
            result += line
            line = self.FROMFILE.readline()
            if line == '\n' and len(result) > 0:
                break
        return result

    def do_command(self, command):
        """Send one command, and return the response."""
        self.send_command(command)
        response = self.get_response()
        print("Rcvd: <<< \n" + response)
        return response

def quick_test(aud):
    """Example list of commands."""
    aud.do_command('Help: Command=Help')
    aud.do_command('Help: Command="GetInfo"')
    #do_command('SetPreference: Name=GUI/Theme Value=classic Reload=1')

if __name__ == "__main__":
    aud = AudacityScript()
    quick_test(aud)