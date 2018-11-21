import sys
import os
import threading

def run_script(cmd):                                                             
    os.system('python {}'.format(cmd))    

def run_kill():                                                             
    os.system('"ping google.com&"')    

if __name__ == '__main__':
    threads = []
    for i in range(8):
        t = threading.Thread(target=run_kill)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
