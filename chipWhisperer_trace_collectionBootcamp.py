import chipwhisperer as cw
import os
import numpy as np
import time
import csv

# Initialize the ChipWhisperer scope (the capture hardware)
scope = cw.scope()
scope.default_setup()

target = cw.target(scope)
output_csv = 'trace_data_collectionBootcamp.csv'

# Initialize the programmer
prog = cw.programmers.STM32FProgrammer

cw.program_target(scope, prog, "simpleserial-aes-CW308_STM32F3.hex")


key = bytearray([0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                 0xab, 0xf7, 0x4f, 0x3c, 0x4f, 0x3c, 0x4f, 0x1b])
# Initialize an empty list to store the traces
traces = []
N=5000

project = cw.create_project("AES_Adv_1.cwp", overwrite=True)

file=open(output_csv, mode='w', newline='')
writer = csv.writer(file)
for i in range(N):
    textin = os.urandom(16)
    trace=cw.capture_trace(scope,target,textin,key)
    if trace is None:
        print(f"No trace collected for input {i}")
        continue
    row = [trace.textin.hex()] + [trace.textout.hex()] + list(trace.wave)
    writer.writerow(row)
    project.traces.append(trace)

file.close()
project.save()
