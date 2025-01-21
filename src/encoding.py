import tiktoken as tk
import struct as st

encd = tk.encoding_for_model('gpt2') # gets the encoding scheme for gpt2 wieghts depend on this encoding
f = open('data/tiny.txt', 'r')
tokens = encd.encode(f.read()) # encodes the text

structs = []
strings = b""
for i in range(50257):
    s = encd.decode([i])
    offset = len(strings)
    structs.append(offset)
    size = len(s)
    structs.append(size)
    strings += bytes(s, "utf-8")

with open('data/enc', 'wb') as file_out:
    for e in structs:
        value = st.pack("I", e)
        file_out.write(value)
    file_out.write(strings)
